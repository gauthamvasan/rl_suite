import argparse
import os
import torch

import numpy as np

from rl_suite.algo.sac import SACAgent
from rl_suite.algo.replay_buffer import SACReplayBuffer
from rl_suite.experiment import Experiment


class SACExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self.args = self.parse_args()
        self.env = self.make_env()
        base_fname = os.path.join(self.args.work_dir, "{}_sac_{}_{}-{}".format(
            self.run_id, self.env.name, self.args.description, self.args.seed))
        self.fname = base_fname + ".txt"
        self.plt_fname = base_fname + ".png"

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Task
        parser.add_argument('--env', required=True, type=str, help="e.g., 'ball_in_cup', 'sparse_reacher', 'Hopper-v2' ")
        parser.add_argument('--seed', default=0, type=int, help="Seed for random number generator")
        parser.add_argument('--tol', default=0.018, type=float, help="Target size in [0.09, 0.018, 0.036, 0.072]")
        parser.add_argument('--image_period', default=1, type=int, help="Update image obs only every 'image_period' steps")
        parser.add_argument('--max_timesteps', default=100000, type=int, help="# timesteps for the run")
        parser.add_argument('--timeout', default=500, type=int, help="Timeout for the env")
        # Algorithm
        parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
        parser.add_argument('--init_steps', default=1000, type=int)
        parser.add_argument('--update_every', default=1, type=int)
        parser.add_argument('--update_epochs', default=1, type=int)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--gamma', default=1, type=float, help="Discount factor")
        parser.add_argument('--bootstrap_terminal', default=0, type=int, help="Bootstrap on terminal state")
        ## Actor
        parser.add_argument('--actor_lr', default=1e-3, type=float)
        parser.add_argument('--actor_update_freq', default=2, type=int)
        ## Critic
        parser.add_argument('--critic_lr', default=1e-3, type=float)
        parser.add_argument('--critic_tau', default=0.001, type=float)
        parser.add_argument('--critic_target_update_freq', default=2, type=int)
        ## Entropy
        parser.add_argument('--init_temperature', default=0.1, type=float)
        parser.add_argument('--alpha_lr', default=1e-4, type=float)
        ## Encoder
        parser.add_argument('--encoder_tau', default=0.001, type=float)
        parser.add_argument('--l2_reg', default=0, type=float, help="L2 regularization coefficient")        
        # MLP params
        parser.add_argument('--actor_hidden_sizes', default="512 512", type=str)
        parser.add_argument('--critic_hidden_sizes', default="512 512", type=str)
        parser.add_argument('--nn_activation', default="relu", type=str)
        # RAD
        parser.add_argument('--rad_offset', default=0.01, type=float)
        parser.add_argument('--freeze_cnn', default=0, type=int)
        # Misc
        parser.add_argument('--work_dir', default='/home/vasan/src/rl_suite/rl_suite/results', type=str)
        parser.add_argument('--checkpoint', default=5000, type=int, help="Save plots and rets every checkpoint")
        parser.add_argument('--device', default="cuda", type=str)
        parser.add_argument('--description', required=True, type=str)
        args = parser.parse_args()

        args.actor_nn_params = {
            'mlp': {
                'hidden_sizes': list(map(int, args.actor_hidden_sizes.split())),
                'activation': args.nn_activation,
            }
        }
        args.critic_nn_params = {
            'mlp': {
                'hidden_sizes': list(map(int, args.critic_hidden_sizes.split())),
                'activation': args.nn_activation,
            }
        }
        if args.device == 'cpu':
            args.device = torch.device("cpu")
        else:
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return args


    def run(self):
        # Reproducibility
        self.set_seed()
                
        # args.observation_shape = env.observation_space.shape
        self.args.action_shape = self.env.action_space.shape
        self.args.obs_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.shape[0]

        buffer = SACReplayBuffer(self.args.obs_dim, self.args.action_dim, self.args.replay_buffer_capacity, self.args.batch_size)
        learner = SACAgent(cfg=self.args, buffer=buffer, device=self.args.device)

        # Experiment block starts
        ret = 0
        step = 0
        rets = []
        ep_lens = []
        obs = self.env.reset()
        i_episode = 0
        for t in range(self.args.max_timesteps):
            # Select an action
            ####### Start
            # Replace the following statement with your own code for
            # selecting an action
            # a = np.random.randint(a_dim)
            if t < self.args.init_steps:
                # TODO: Fix bug with lack of reproducibility in using env.action_space.sample()
                # action = self.env.action_space.sample()       
                action = np.random.uniform(
                    low=self.env.action_space.low, high=self.env.action_space.high, size=self.args.action_dim)         
            else:
                print("Obs:", obs.shape)
                action = learner.sample_action(obs)
                print(action, self.args.action_dim)
            ####### End

            # Observe
            next_obs, r, done, infos = self.env.step(action)

            # Learn
            ####### Start
            learner.push_and_update(obs, action, r, done)
            # if t % 100 == 0:
                # print("Step: {}, Obs: {}, Action: {}, Reward: {:.2f}, Done: {}".format(
                    # t, obs[:2], action, r, done))
            obs = next_obs
            ####### End

            # Log
            ret += r
            step += 1
            if done or step == self.args.timeout:    # Bootstrap on timeout
                i_episode += 1
                rets.append(ret)
                ep_lens.append(step)
                print("Episode {} ended after {} steps with return {}. Total steps: {}".format(
                    i_episode, step, ret, t))
                ret = 0
                step = 0
                obs = self.env.reset()

            if (t+1) % self.args.checkpoint == 0:
                self.learning_curve(rets, ep_lens)
                self.save_returns(rets, ep_lens, self.fname)

        self.save_returns(rets, ep_lens, self.fname)
        learner.save(model_dir=self.args.work_dir, step=self.args.max_timesteps)
        # plt.show()

def main():
    runner = SACExperiment()
    runner.run()


if __name__ == "__main__":
    main()
