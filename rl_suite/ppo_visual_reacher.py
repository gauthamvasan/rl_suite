import sys
import argparse
import gym
import torch


import numpy as np
import matplotlib.pyplot as plt

from rl_suite.algo.ppo_rad import PPO_RAD
from rl_suite.algo.replay_buffer import VisuomotorReplayBuffer
from rl_suite.envs.visual_reacher import VisualMujocoReacher2D
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
    # Algorithm
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--opt_batch_size', default=128, type=int, help="Optimizer batch size")
    parser.add_argument('--n_epochs', default=10, type=int, help="Number of learning epochs per PPO update")
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--lmbda', default=0.995, type=float, help="Lambda return coefficient")
    parser.add_argument('--clip_epsilon', default=0.2, type=float, help="Clip epsilon for KL divergence in PPO actor loss")
    parser.add_argument('--l2_reg', default=1e-4, type=float, help="L2 regularization coefficient")
    parser.add_argument('--bootstrap_terminal', default=0, type=int, help="Bootstrap on terminal state")
    # RAD
    parser.add_argument('--rad_offset', default=0.01, type=float)
    parser.add_argument('--freeze_cnn', default=0, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    seed = args.seed

    # Task setup block starts
    # Do not change
    env = VisualMujocoReacher2D(tol=args.tol)
    env.seed(seed)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    # Task setup block end

    # Learner setup block
    torch.manual_seed(seed)
    ####### Start
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
    ret = 0
    rets = []
    avgrets = []
    obs = env.reset()
    num_steps = 500000
    checkpoint = 10000
    for steps in range(num_steps):

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
        if steps % 100 == 0:
            print("Step: {}, Obs: {}, Action: {}, Reward: {:.2f}, Done: {}".format(
                steps, obs.proprioception[:2], action, r, done))
        obs = next_obs
        ####### End

        # Log
        ret += r
        if done:
          rets.append(ret)
          ret = 0
          obs = env.reset()

        if (steps+1) % checkpoint == 0:
          avgrets.append(np.mean(rets))
          rets = []
          plt.clf()
          plt.plot(range(checkpoint, (steps+1)+checkpoint, checkpoint), avgrets)
          plt.pause(0.001)
    name = sys.argv[0].split('.')[-2].split('_')[-1]
    data = np.zeros((2, len(avgrets)))
    data[0] = range(checkpoint, num_steps+1, checkpoint)
    data[1] = avgrets
    np.savetxt(name+str(seed)+".txt", data)
    # plt.show()


if __name__ == "__main__":
    main()
