import sys
import argparse
import os
import torch
import gym

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from rl_suite.algo.sac import SACAgent
from rl_suite.algo.replay_buffer import SACReplayBuffer
from rl_suite.envs.visual_reacher import VisualMujocoReacher2D
from rl_suite.plot import smoothed_curve
from sys import platform
if platform == "darwin":    # For MacOS
    import matplotlib as mpl
    mpl.use("TKAgg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NN architecture
mlp_config = {
    'mlp': {
        'hidden_sizes': [64, 64],
        'activation': "relu",
    }
}

def parse_args():
    parser = argparse.ArgumentParser()
    # Task
    parser.add_argument('--seed', default=0, type=int, help="Seed for random number generator")
    parser.add_argument('--tol', default=0.018, type=float, help="Target size in [0.09, 0.018, 0.036, 0.072]")
    parser.add_argument('--image_period', default=1, type=int, help="Update image obs only every 'image_period' steps")
    parser.add_argument('--max_timesteps', default=100000, type=int, help="# timesteps for the run")
    parser.add_argument('--timeout', default=500, type=int, help="Timeout for the env")
    # Algorithm
    parser.add_argument('--replay_buffer_capacity', default=10000, type=int)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--update_every', default=1, type=int)
    parser.add_argument('--update_epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--gamma', default=1, type=float, help="Discount factor")
    ## Actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    ## Critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.001, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    ## Entropy
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    ## Encoder
    parser.add_argument('--encoder_tau', default=0.001, type=float)
    parser.add_argument('--l2_reg', default=0, type=float, help="L2 regularization coefficient")
    parser.add_argument('--bootstrap_terminal', default=0, type=int, help="Bootstrap on terminal state")
    # RAD
    parser.add_argument('--rad_offset', default=0.01, type=float)
    parser.add_argument('--freeze_cnn', default=0, type=int)
    # Misc
    parser.add_argument('--work_dir', default='/home/vasan/src/rl_suite/rl_suite/results', type=str)
    parser.add_argument('--checkpoint', default=5000, type=int, help="Save plots and rets every checkpoint")
    args = parser.parse_args()
    return args

def save_returns(rets, ep_lens, fname):
    data = np.zeros((2, len(rets)))
    data[0] = ep_lens
    data[1] = rets
    np.savetxt(fname, data)

def run(args, env):    
    seed = args.seed

    # Task setup block starts
    # Do not change    
    env.seed(seed)
    # Task setup block end

    # Learner setup block
    ####### Start
    torch.manual_seed(seed)
    np.random.seed(seed)
    # args.observation_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.net_params = mlp_config

    buffer = SACReplayBuffer(args.obs_dim, args.action_dim, args.replay_buffer_capacity, args.batch_size)
    learner = SACAgent(cfg=args, buffer=buffer, device=device)
    ####### End

    # Experiment block starts
    fname = os.path.join(args.work_dir, "sac_reacher_tol-{}_bs-{}_{}.txt".format(args.tol, args.batch_size, seed))
    plt_fname = os.path.join(args.work_dir, "sac_reacher_tol-{}_bs-{}_{}.png".format(args.tol, args.batch_size, seed))
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
        if t < args.init_steps:
            action = env.action_space.sample()
        else:
            action = learner.sample_action(obs)
        ####### End

        # Observe
        next_obs, r, done, infos = env.step(action)

        # Learn
        ####### Start
        learner.push_and_update(obs, action, r, done)
        if t % 100 == 0:
            print("Step: {}, Obs: {}, Action: {}, Reward: {:.2f}, Done: {}".format(
                t, obs[:2], action, r, done))
        obs = next_obs
        ####### End

        # Log
        ret += r
        step += 1
        if done or step == args.timeout:    # Bootstrap on timeout
            i_episode += 1
            rets.append(ret)
            ep_lens.append(step)
            print("Episode {} ended after {} steps with return {}".format(i_episode, step, ret))
            ret = 0
            step = 0
            obs = env.reset()

        if (t+1) % args.checkpoint == 0:
            plot_rets, plot_x = smoothed_curve(
                np.array(rets), np.array(ep_lens), x_tick=args.checkpoint, window_len=args.checkpoint)
            if len(plot_rets):
                plt.clf()
                plt.plot(plot_x, plot_rets)
                plt.pause(0.001)
                plt.savefig(plt_fname)
            save_returns(rets, ep_lens, fname)

    save_returns(rets, ep_lens, fname)
    # plt.show()

def main():
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    args = parse_args()
    # env = VisualMujocoReacher2D(tol=args.tol)
    env = gym.make('Reacher-v2')
    run(args, env)

if __name__ == "__main__":
    main()
