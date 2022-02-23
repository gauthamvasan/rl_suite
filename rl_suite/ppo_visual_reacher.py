import sys
import argparse
import os
import torch


import numpy as np
import matplotlib.pyplot as plt

from rl_suite.algo.ppo_rad import PPO_RAD
from rl_suite.algo.replay_buffer import VisuomotorReplayBuffer
from rl_suite.envs.visual_reacher import VisualMujocoReacher2D
from rl_suite.plot import smoothed_curve
from sys import platform
if platform == "darwin":    # For MacOS
    import matplotlib as mpl
    mpl.use("TKAgg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Spatial softmax encoder
ss_config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],

    'latent': 50,

    'mlp': [
        [-1, 1024],
        [1024, 1024],
        [1024, -1]
    ],
}

def parse_args():
    parser = argparse.ArgumentParser()
    # Task
    parser.add_argument('--seed', default=0, type=int, help="Seed for random number generator")
    parser.add_argument('--tol', default=0.018, type=float, help="Target size in [0.09, 0.018, 0.036, 0.072]")
    parser.add_argument('--image_period', default=1, type=int, help="Update image obs only every 'image_period' steps")
    parser.add_argument('--max_timesteps', default=500000, type=int, help="# timesteps for the run")
    parser.add_argument('--timeout', default=500, type=int, help="Timeout for the env")
    # Algorithm
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--opt_batch_size', default=256, type=int, help="Optimizer batch size")
    parser.add_argument('--n_epochs', default=10, type=int, help="Number of learning epochs per PPO update")
    parser.add_argument('--actor_lr', default=0.0003, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--lmbda', default=0.97, type=float, help="Lambda return coefficient")
    parser.add_argument('--clip_epsilon', default=0.2, type=float, help="Clip epsilon for KL divergence in PPO actor loss")
    parser.add_argument('--l2_reg', default=0, type=float, help="L2 regularization coefficient")
    parser.add_argument('--bootstrap_terminal', default=0, type=int, help="Bootstrap on terminal state")
    # RAD
    parser.add_argument('--rad_offset', default=0.01, type=float)
    parser.add_argument('--freeze_cnn', default=0, type=int)
    # Misc
    parser.add_argument('--work_dir', default='./results', type=str)
    parser.add_argument('--checkpoint', default=5000, type=int, help="Save plots and rets every checkpoint")
    args = parser.parse_args()
    return args

def save_returns(rets, ep_lens, fname):
    data = np.zeros((2, len(rets)))
    data[0] = ep_lens
    data[1] = rets
    np.savetxt(fname, data)

def main():
    args = parse_args()
    seed = args.seed

    # Task setup block starts
    # Do not change
    env = VisualMujocoReacher2D(tol=args.tol)
    env.seed(seed)
    # Task setup block end

    # Learner setup block
    ####### Start
    torch.manual_seed(seed)
    np.random.seed(seed)
    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.net_params = ss_config

    buffer = VisuomotorReplayBuffer(env.image_space.shape, env.proprioception_space.shape, env.action_space.shape,
                                    args.batch_size + 1000, store_lprob=True)
    learner = PPO_RAD(cfg=args, buffer=buffer, device=device)
    ####### End

    # Experiment block starts
    fname = os.path.join(args.work_dir, "ppo_visual_reacher_bs-{}_{}.txt".format(args.batch_size, seed))
    plt_fname = os.path.join(args.work_dir, "ppo_visual_reacher_bs-{}_{}.png".format(args.batch_size, seed))
    ret = 0
    step = 0
    rets = []
    ep_lens = []
    obs = env.reset()
    i_episode = 0
    for t in range(args.max_timesteps):

        # Select an action
        ####### Start
        # Replace the following statement with your own code for
        # selecting an action
        # a = np.random.randint(a_dim)
        img = torch.as_tensor(obs.images.astype(np.float32))[None, :, :, :]
        prop = torch.as_tensor(obs.proprioception.astype(np.float32))[None, :]
        action, lprob = learner.sample_action(img, prop)
        ####### End

        # Observe
        next_obs, r, done, infos = env.step(action)

        # Learn
        ####### Start
        learner.push_and_update(images=img, proprioception=prop, action=action,
                                reward=r, log_prob=lprob, done=done)
        if t % 100 == 0:
            print("Step: {}, Obs: {}, Action: {}, Reward: {:.2f}, Done: {}".format(
                t, obs.proprioception[:2], action, r, done))
        obs = next_obs
        ####### End

        # Log
        ret += r
        step += 1
        if done or step == args.timeout:
            i_episode += 1
            rets.append(ret)
            ep_lens.append(step)
            print("Episode {} ended after {} steps with return {}".format(i_episode, step, ret))
            ret = 0
            step = 0
            obs = env.reset()

        if (t+1) % args.checkpoint == 0:
            plot_rets, plot_x = smoothed_curve(rets, ep_lens, x_tick=args.checkpoint, window_len=args.checkpoint)
            if plot_rets.any():
                plt.clf()
                plt.plot(plot_x, plot_rets)
                plt.pause(0.001)
                plt.savefig(plt_fname)
            save_returns(rets, ep_lens, fname)

    save_returns(rets, ep_lens, fname)
    # plt.show()


if __name__ == "__main__":
    main()
