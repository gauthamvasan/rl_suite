import argparse
import os
import torch

import numpy as np

from rl_suite.algo.sac import ResetSACAgent
from rl_suite.algo.replay_buffer import SACReplayBuffer
from rl_suite.experiment import Experiment


class SACExperiment(Experiment):
    def __init__(self):
        super(SACExperiment, self).__init__(self.parse_args())
        self.env = self.make_env()
        base_fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.args.work_dir, "{}_sac_{}_{}-{}".format(
            self.run_id, self.env.name, self.args.description, self.args.seed))
        self.fname = base_fname + ".txt"
        self.plt_fname = base_fname + ".png"

        print('-'*50)
        print("{}-{}".format(self.run_id, base_fname))
        print('-'*50)               

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Task
        parser.add_argument('--env', default="dot_reacher", type=str, help="e.g., 'ball_in_cup', 'sparse_reacher', 'Hopper-v2' ")
        parser.add_argument('--seed', default=0, type=int, help="Seed for random number generator")       
        parser.add_argument('--N', default=150000, type=int, help="# timesteps for the run")
        parser.add_argument('--timeout', default=10000, type=int, help="Timeout for the env")        
        ## DotReacher
        parser.add_argument('--pos_tol', default=0.25, type=float, help="Position tolerance in [0.05, ..., 0.25]")
        parser.add_argument('--vel_tol', default=0.1, type=float, help="Velocity tolerance in [0.05, ..., 0.1]")
        parser.add_argument('--dt', default=0.2, type=float, help="Simulation action cycle time")
        parser.add_argument('--clamp_action', default=1, type=int, help="Clamp action space")
        # Algorithm
        parser.add_argument('--replay_buffer_capacity', default=150000, type=int)
        parser.add_argument('--init_steps', default=5000, type=int)
        parser.add_argument('--update_every', default=50, type=int)
        parser.add_argument('--update_epochs', default=50, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--gamma', default=0.995, type=float, help="Discount factor")
        parser.add_argument('--bootstrap_terminal', default=0, type=int, help="Bootstrap on terminal state")
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
        # MLP params
        parser.add_argument('--actor_hidden_sizes', default="512 512", type=str)
        parser.add_argument('--critic_hidden_sizes', default="512 512", type=str)
        parser.add_argument('--nn_activation', default="relu", type=str)
        # Misc
        parser.add_argument('--work_dir', default='./results/sac_dot_reacher', type=str)
        parser.add_argument('--checkpoint', default=10000, type=int, help="Save plots and rets every checkpoint")
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

        buffer = SACReplayBuffer(self.args.obs_dim, self.args.action_dim+1, self.args.replay_buffer_capacity, self.args.batch_size)
        learner = ResetSACAgent(cfg=self.args, buffer=buffer, device=self.args.device)

        # Experiment block starts
        ret = 0
        step = 0
        rets = []
        ep_lens = []
        obs = self.env.reset()
        i_episode = 0
        for t in range(self.args.N):
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
                x_action = torch.as_tensor(action.astype(np.float32)).view((1, -1))
                reset_action = np.random.uniform(-1, 1)
                action = np.concatenate((action, np.array([reset_action])))
            else:
                _, action, _, _ = learner.actor(obs)
                action = action.detach().cpu()
                x_action = action[:, :self.args.action_dim]
                reset_action = action[:, -1]
            ####### End

            # Observe
            next_obs, r, done, infos = self.env.step(x_action)

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
            if done:    # Bootstrap on timeout
                i_episode += 1
                rets.append(ret)
                ep_lens.append(step)
                print("Episode {} ended after {} steps with return {:.2f}. Total steps: {}".format(
                    i_episode, step, ret, t))
                ret = 0
                step = 0
                obs = self.env.reset()

            if (t+1) % self.args.checkpoint == 0:
                self.learning_curve(rets, ep_lens, save_fig=self.plt_fname)
                self.save_returns(rets, ep_lens, self.fname)

        self.save_returns(rets, ep_lens, self.fname)
        learner.save(model_dir=self.args.work_dir, step=self.args.N)
        # plt.show()

def main():
    runner = SACExperiment()
    runner.run()


if __name__ == "__main__":
    main()
