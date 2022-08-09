import argparse
import os
import torch

import numpy as np

from rl_suite.algo.sac import ResetSACAgent
from rl_suite.algo.sac_rad import ResetSACRADAgent
from rl_suite.algo.replay_buffer import SACReplayBuffer, SACRADBuffer
from rl_suite.experiment import Experiment


class SACExperiment(Experiment):
    def __init__(self):
        super(SACExperiment, self).__init__(self.parse_args())
        self.env = self.make_env()
        base_fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.args.work_dir, "{}_sac_{}_{}-{}".format(
            self.run_id, self.env.name, self.args.description, self.args.seed))
        self.fname = base_fname + ".txt"
        self.plt_fname = base_fname + ".png"
        self.base_fname = base_fname

        print('-'*50)
        print("{}-{}".format(self.run_id, base_fname))
        print('-'*50)               

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Task
        parser.add_argument('--env', default="sparse_reacher", type=str, help="e.g., 'ball_in_cup', 'sparse_reacher', 'Hopper-v2' ")
        parser.add_argument('--seed', default=0, type=int, help="Seed for random number generator")       
        parser.add_argument('--N', default=150000, type=int, help="# timesteps for the run")
        parser.add_argument('--timeout', default=500, type=int, help="Timeout for the env")
        ## Sparse reacher
        parser.add_argument('--tol', default=0.036, type=float, help="Target size in [0.09, 0.018, 0.036, 0.072]")
        # Reset threshold
        parser.add_argument('--reset_thresh', default=0.9, type=float, help="Action threshold between [-1, 1]")
        # Algorithm
        parser.add_argument('--algo', default="sac", type=str, help="Choices: ['sac', 'sac_rad']")
        parser.add_argument('--replay_buffer_capacity', default=150000, type=int)
        parser.add_argument('--init_steps', default=1000, type=int)
        parser.add_argument('--update_every', default=50, type=int)
        parser.add_argument('--update_epochs', default=50, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--gamma', default=0.995, type=float, help="Discount factor")
        parser.add_argument('--bootstrap_terminal', default=0, type=int, help="Bootstrap on terminal state")
        # Ball In A Cup
        parser.add_argument('--penalty', default=0.1, type=float, help="Reward penalty")
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
        # RAD
        parser.add_argument('--rad_offset', default=0.01, type=float)
        parser.add_argument('--freeze_cnn', default=0, type=int)        
        # MLP params
        parser.add_argument('--actor_hidden_sizes', default="512 512", type=str)
        parser.add_argument('--critic_hidden_sizes', default="512 512", type=str)
        parser.add_argument('--nn_activation', default="relu", type=str)
        # Misc
        parser.add_argument('--work_dir', default='./results', type=str)
        parser.add_argument('--checkpoint', default=5000, type=int, help="Save plots and rets every checkpoint")
        parser.add_argument('--device', default="cuda", type=str)
        parser.add_argument('--description', required=True, type=str)
        args = parser.parse_args()

        assert args.algo in ["sac", "sac_rad"]        
        if args.algo == "sac":
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
        else:
            # TODO: Fix this hardcoding by providing choice of network architectures
            args.net_params = {
                # Spatial softmax encoder net params
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
                    [1024, 1024],
                    [1024, -1]
                ],
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

        # One additonal action_dim for reset action
        if self.args.algo == "sac":
            buffer = SACReplayBuffer(self.args.obs_dim, self.args.action_dim+1, self.args.replay_buffer_capacity, self.args.batch_size)
            learner = ResetSACAgent(cfg=self.args, buffer=buffer, device=self.args.device)
        else:
            self.args.image_shape = self.env.image_space.shape
            self.args.proprioception_shape = self.env.proprioception_space.shape
            self.args.action_shape = (self.env.action_space.shape[0]+1,)
            buffer = SACRADBuffer(self.env.image_space.shape, self.env.proprioception_space.shape, 
                (self.args.action_dim+1,), self.args.replay_buffer_capacity, self.args.batch_size)
            learner = ResetSACRADAgent(cfg=self.args, buffer=buffer, device=self.args.device)

        # Experiment block starts
        ret = 0
        step = 0
        t = 0
        rets = []
        ep_lens = []
        n_resets = []
        obs = self.env.reset()
        i_episode = 0
        n_reset = 0
        while t < self.args.N:
            if self.args.algo == "sac_rad":
                img = obs.images
                prop = obs.proprioception
            
            # Select an action
            if t < self.args.init_steps:
                x_action = np.random.uniform(low=-1, high=1, size=self.args.action_dim)
                reset_action = np.random.uniform(-1, 1)
                action = np.concatenate((x_action, np.array([reset_action])))
            else:
                if self.args.algo == "sac":
                    action = learner.sample_action(obs)                
                else:
                    action = learner.sample_action(img, prop)
                x_action = action[:self.args.action_dim]
                reset_action = action[-1]

            # Reset action
            if reset_action > self.args.reset_thresh: 
                n_reset += 1
                t += 10
                step += 10               
                next_obs = self.env.reset()
                r = -1
                done = False
                infos = "Agent chose to reset itself"
            else:
                next_obs, r, done, infos = self.env.step(x_action)
                
            # Learn
            ####### Start
            if self.args.algo == "sac":
                learner.push_and_update(obs, action, r, done)
            else:
                learner.push_and_update(img, prop, action, r, done)
                
            # if t % 100 == 0:
                # print("Step: {}, Obs: {}, Action: {}, Reward: {:.2f}, Done: {}".format(
                    # t, obs, action, r, done))
            obs = next_obs
            ####### End

            # Log
            ret += r
            step += 1
            t += 1
            if done:    # Bootstrap on timeout
                i_episode += 1
                rets.append(ret)
                ep_lens.append(step)
                n_resets.append(n_reset)
                print("Episode {} ended after {} steps with return {:.2f}. # resets: {}. Total steps: {}".format(
                    i_episode, step, ret, n_reset, t))
                
                ret = 0
                step = 0
                n_reset = 0
                obs = self.env.reset()

            if (t+1) % self.args.checkpoint == 0:
                self.learning_curve(rets, ep_lens, save_fig=self.plt_fname)
                self.save_returns(rets, ep_lens, self.fname)
                np.savetxt(self.base_fname + "num_resets.txt", np.array(n_resets))

        self.save_returns(rets, ep_lens, self.fname)
        np.savetxt(self.base_fname + "num_resets.txt", np.array(n_resets))
        learner.save(model_dir=self.args.work_dir, step=self.args.N)
        # plt.show()

def main():
    runner = SACExperiment()
    runner.run()


if __name__ == "__main__":
    main()
