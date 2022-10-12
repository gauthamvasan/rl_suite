import argparse
import torch
import numpy as np
import cv2

from rl_suite.algo.sac import SACAgent
from rl_suite.algo.sac_rad import SACRADAgent
from rl_suite.algo.replay_buffer import SACReplayBuffer, SACRADBuffer
from rl_suite.experiment import Experiment
from rl_suite.running_stats import RunningStats
from datetime import datetime

class SACExperiment(Experiment):
    def __init__(self):
        super(SACExperiment, self).__init__(self.parse_args())
        self.env = self.make_env()
        
        # Reproducibility
        self.set_seed()
                
        # args.observation_shape = env.observation_space.shape
        self.args.action_shape = self.env.action_space.shape
        self.args.obs_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.shape[0]

        if self.args.algo == "sac":
            buffer = SACReplayBuffer(self.args.obs_dim, self.args.action_dim, self.args.replay_buffer_capacity, self.args.batch_size)
            self.learner = SACAgent(cfg=self.args, buffer=buffer, device=self.args.device)
        else:
            self.args.image_shape = self.env.image_space.shape
            print("image shape:", self.args.image_shape)
            self.args.proprioception_shape = self.env.proprioception_space.shape
            self.args.action_shape = self.env.action_space.shape
            buffer = SACRADBuffer(self.env.image_space.shape, self.env.proprioception_space.shape, 
                self.args.action_shape, self.args.replay_buffer_capacity, self.args.batch_size)
            self.learner = SACRADAgent(cfg=self.args, buffer=buffer, device=self.args.device)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Task
        parser.add_argument('--env', required=True, type=str, help="e.g., 'ball_in_cup', 'dm_reacher_easy', 'dm_reacher_hard', 'Hopper-v2' ")        
        parser.add_argument('--seed', required=True, type=int, help="Seed for random number generator")       
        parser.add_argument('--N', required=True, type=int, help="# timesteps for the run")
        parser.add_argument('--timeout', required=True, type=int, help="Timeout for the env")
        # Minimum-time tasks
        parser.add_argument('--penalty', default=-1, type=float, help="Reward penalty for min-time specification")
        ## Mujoco sparse reacher
        parser.add_argument('--tol', default=0.036, type=float, help="Target size in [0.09, 0.018, 0.036, 0.072]")
        ## DotReacher
        parser.add_argument('--pos_tol', default=0.1, type=float, help="Position tolerance in [0.05, ..., 0.25]")
        parser.add_argument('--vel_tol', default=0.05, type=float, help="Velocity tolerance in [0.05, ..., 0.1]")
        parser.add_argument('--dt', default=0.2, type=float, help="Simulation action cycle time")
        parser.add_argument('--clamp_action', default=1, type=int, help="Clamp action space")        
        # Algorithm
        parser.add_argument('--algo', required=True, type=str, help="Choices: ['sac', 'sac_rad']")
        parser.add_argument('--replay_buffer_capacity', required=True, type=int)
        parser.add_argument('--init_steps', required=True, type=int)
        parser.add_argument('--update_every', default=2, type=int)
        parser.add_argument('--update_epochs', default=1, type=int)
        parser.add_argument('--batch_size', default=256, type=int)
        parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
        parser.add_argument('--bootstrap_terminal', default=0, type=int, help="Bootstrap on terminal state")
        parser.add_argument('--normalize', default=0, type=int, help="Normalize observation")
        ## Actor
        parser.add_argument('--actor_lr', default=3e-4, type=float)
        parser.add_argument('--actor_update_freq', default=1, type=int)
        ## Critic
        parser.add_argument('--critic_lr', default=3e-4, type=float)
        parser.add_argument('--critic_tau', default=0.005, type=float)
        parser.add_argument('--critic_target_update_freq', default=1, type=int)
        ## Entropy
        parser.add_argument('--init_temperature', default=0.1, type=float)
        parser.add_argument('--alpha_lr', default=3e-4, type=float)
        ## Encoder
        parser.add_argument('--encoder_tau', default=0.005, type=float)
        parser.add_argument('--l2_reg', default=1e-4, type=float, help="L2 regularization coefficient")        
        # MLP params
        parser.add_argument('--actor_hidden_sizes', default="512 512", type=str)
        parser.add_argument('--critic_hidden_sizes', default="512 512", type=str)
        parser.add_argument('--nn_activation', default="relu", type=str)
        # RAD
        parser.add_argument('--rad_offset', default=0.01, type=float)
        parser.add_argument('--freeze_cnn', default=0, type=int)
        # Misc
        parser.add_argument('--output_dir', default='./outputs', type=str, help="Save outputs to this dir")
        parser.add_argument('--experiment_dir', required=True, type=str, help="Save experiment outputs, relative to output_dir")
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
                    [-1, 512],
                    [512, 512],
                    [512, -1]
                ],
            }            

        if args.device == 'cpu':
            args.device = torch.device("cpu")
        else:
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return args


    def run(self):
        # Normalize wrapper
        rms = RunningStats()

        # Experiment block starts
        ret = 0
        epi_steps = 0
        rets = []
        ep_lens = []
        obs = self.env.reset()
        # img_to_show = np.transpose(obs.images, [1, 2, 0])
        # img_to_show = img_to_show[:,:,-3:]
        # cv2.imshow("", img_to_show)
        # cv2.waitKey(0)
        i_episode = 0
        start_time = datetime.now()
        for t in range(self.args.N):                      
            if self.args.algo == "sac_rad":
                img = obs.images
                prop = obs.proprioception
            else:
                if self.args.normalize:
                    rms.push(obs)
                    obs = rms.zscore(obs)

            # Select an action
            if self.args.algo == "sac":
                action = self.learner.sample_action(obs)                
            else:
                action = self.learner.sample_action(img, prop)
                
            # if t < self.args.init_steps:
            #     # TODO: Fix bug with lack of reproducibility in using env.action_space.sample()
            #     # action = self.env.action_space.sample()       
            #     action = self.env.action_space.sample()
            # else:
            #     if self.args.algo == "sac":
            #         action = self.learner.sample_action(obs)                
            #     else:
            #         action = self.learner.sample_action(img, prop)
            
            # img_to_show = np.transpose(obs.images, [1, 2, 0])
            # img_to_show = img_to_show[:,:,-3:]
            # cv2.imshow("", img_to_show)
            # cv2.waitKey(50)
            # Observe
            next_obs, r, done, infos = self.env.step(action)

            # Learn
            if self.args.algo == "sac":
                self.learner.push_and_update(obs, action, r, done)
            else:
                self.learner.push_and_update(img, prop, action, r, done)
                    
            # if t % 100 == 0:
                # print("Step: {}, Obs: {}, Action: {}, Reward: {:.2f}, Done: {}".format(
                    # t, obs[:2], action, r, done))
            obs = next_obs

            # Log
            ret += r
            epi_steps += 1
            if done or epi_steps == self.args.timeout:    # Bootstrap on timeout
                i_episode += 1
                rets.append(ret)
                ep_lens.append(epi_steps)
                print("Episode {} ended after {} steps with return {:.2f}. Total steps: {}".format(
                    i_episode, epi_steps, ret, t))
                ret = 0
                epi_steps = 0
                
                if 'dm_reacher' in self.args.env:
                    obs = self.env.reset(randomize_target=done)
                else:
                    obs = self.env.reset()

            if (t+1) % self.args.checkpoint == 0:
                if rets:
                    self.show_learning_curve(rets, ep_lens, save_fig=True)
                    self.save_returns(rets, ep_lens)

        duration = datetime.now() - start_time
        self.save_returns(rets, ep_lens)
        self.save_model(t)
        print(f"Finished in {duration}")

def main():
    runner = SACExperiment()
    runner.run()

if __name__ == "__main__":
    main()
