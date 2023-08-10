import argparse
import torch
import cv2

import numpy as np

from datetime import datetime
from tqdm import tqdm
from rl_suite.algo.sac import SACAgent
from rl_suite.algo.sac_rad import SACRADAgent
from rl_suite.algo.replay_buffer import SACReplayBuffer, SACRADBuffer
from rl_suite.experiment import Experiment
from rl_suite.running_stats import RunningStats


class SACExperiment(Experiment):
    def __init__(self):
        # Create env, run_id, results dir, etc.
        super(SACExperiment, self).__init__(self.parse_args())
        
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
        parser.add_argument('--use_image', action='store_true')
        ## Minimum-time tasks
        parser.add_argument('--reward', default=-1, type=float, help="Reward penalty for min-time specification")
        parser.add_argument('--reset_penalty_steps', default=20, type=float, help="Reset penalty steps for min-time specification")
        ## Mujoco sparse reacher
        parser.add_argument('--tol', default=0.009, type=float, help="Target size in [0.09, 0.018, 0.036, 0.072]")
        ## DotReacher
        parser.add_argument('--pos_tol', default=0.1, type=float, help="Position tolerance in [0.05, ..., 0.25]")
        parser.add_argument('--vel_tol', default=0.05, type=float, help="Velocity tolerance in [0.05, ..., 0.1]")
        parser.add_argument('--dt', default=0.2, type=float, help="Simulation action cycle time")
        parser.add_argument('--clamp_action', default=1, type=int, help="Clamp action space")    
        ## Point Maze
        parser.add_argument('--maze_type', default="small", type=str, help= "Maze type in ['small', 'medium', 'large']")
        parser.add_argument('--reward_type', default="sparse", type=str, help= "Reward type in ['sparse', 'dense']")
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
        parser.add_argument('--betas', default="0.9 0.999", type=str)
        ## Actor
        parser.add_argument('--actor_lr', default=3e-4, type=float)
        parser.add_argument('--update_actor_every', default=1, type=int)
        ## Critic
        parser.add_argument('--critic_lr', default=3e-4, type=float)
        parser.add_argument('--critic_tau', default=0.005, type=float)
        parser.add_argument('--update_critic_target_every', default=1, type=int)
        ## Entropy
        parser.add_argument('--init_temperature', default=0.1, type=float)
        parser.add_argument('--alpha_lr', default=3e-4, type=float)
        ## Encoder
        parser.add_argument('--encoder_tau', default=0.005, type=float)
        parser.add_argument('--l2_reg', default=1e-4, type=float, help="L2 regularization coefficient")        
        # MLP params
        parser.add_argument('--actor_hidden_sizes', default="512,512", type=str)
        parser.add_argument('--critic_hidden_sizes', default="512,512", type=str)
        parser.add_argument('--nn_activation', default="relu", type=str)
        # RAD
        parser.add_argument('--rad_offset', default=0.01, type=float)
        parser.add_argument('--freeze_cnn', default=0, type=int)
        # Misc
        parser.add_argument('--init_policy_test', action='store_true', help="Initiate hits vs timeout test")
        parser.add_argument('--results_dir', required=True, type=str, help="Save results to this dir")
        parser.add_argument('--xlimit', default=None, type=str)
        parser.add_argument('--ylimit', default=None, type=str)
        parser.add_argument('--checkpoint', default=5000, type=int, help="Save plots and rets every checkpoint")
        parser.add_argument('--model_checkpoint', default=0, type=int, help="Save plots and rets every checkpoint")
        parser.add_argument('--device', default="cuda", type=str)
        args = parser.parse_args()

        assert args.algo in ["sac", "sac_rad"]
        assert args.reward < 0 and args.reset_penalty_steps >= 0

        if args.xlimit is not None:
            args.xlimit = tuple(args.xlimit)

        if args.ylimit is not None:
            args.ylimit = tuple(args.ylimit)
        
        args.betas = list(map(float, args.betas.split()))

        if args.algo == "sac":
            args.actor_nn_params = {
                'mlp': {
                    'hidden_sizes': list(map(int, args.actor_hidden_sizes.split(","))),
                    'activation': args.nn_activation,
                }
            }
            args.critic_nn_params = {
                'mlp': {
                    'hidden_sizes': list(map(int, args.critic_hidden_sizes.split(","))),
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
            args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        return args

    def run(self):
        if self.args.init_policy_test:
            print("Initial policy test")
            self._run_init_policy_test()
        else:
            print("{}-{} run with {} steps starts now ...".format(self.args.env, self.args.algo, self.args.N))
            self._run_experiment()
            
    def _run_experiment(self):
        # Normalize wrapper
        rms = RunningStats()

        # Experiment block starts
        experiment_done = False
        total_steps = 0
        sub_epi = 0
        returns = []
        epi_lens = []
        start_time = datetime.now()
        print(f'Experiment starts at: {start_time}')
        while not experiment_done: 
            obs = self.env.reset() # start a new episode
            ret = 0
            epi_steps = 0
            sub_steps = 0
            epi_done = 0
            while not experiment_done and not epi_done:
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
                
                # Observe
                next_obs, r, epi_done, _ = self.env.step(action)
                
                # Learn
                if self.args.algo == "sac":
                    self.learner.push_and_update(obs, action, r, epi_done)
                else:
                    self.learner.push_and_update(img, prop, action, r, epi_done)
                
                # if total_steps % 50 == 0: 
                #     print("Step: {}, Next Obs: {}, reward: {}, done: {}".format(total_steps, next_obs, r, epi_done))

                obs = next_obs

                # Log
                total_steps += 1
                ret += r
                epi_steps += 1
                sub_steps += 1

                # Save model
                if self.args.model_checkpoint:
                    if total_steps % self.args.model_checkpoint == 0:
                        self.save_model(unique_str=f"{self.run_id}_model_{total_steps//1000}K")
                
                if not epi_done and sub_steps >= self.args.timeout: # set timeout here
                    sub_steps = 0
                    sub_epi += 1

                    # Prevent discontinuity in saving models
                    x = total_steps//10000
                    y = (total_steps + self.args.reset_penalty_steps)//10000
                    # Save model
                    if self.args.model_checkpoint and x != y:
                        self.save_model(unique_str=f"{self.run_id}_model_{y}0K")

                    ret += self.args.reset_penalty_steps * self.args.reward
                    epi_steps += self.args.reset_penalty_steps
                    total_steps += self.args.reset_penalty_steps
                    print(f'Sub episode {sub_epi} done. Total steps: {total_steps}')
                    if 'dm_reacher' or 'point_maze' in self.args.env:
                        obs = self.env.reset(randomize_target=epi_done)
                    else:
                        obs = self.env.reset()

                experiment_done = total_steps >= self.args.N

            if epi_done: # episode done, save result
                returns.append(ret)
                epi_lens.append(epi_steps)
                self.save_returns(returns, epi_lens)
                self.learning_curve(returns, epi_lens, save_fig=True)
                print(f"Episode {len(returns)} ended after {epi_steps} steps with return {ret:.2f}. Total steps: {total_steps}")

        duration = datetime.now() - start_time
        self.save_model(unique_str=f"{self.run_id}_model")
        self.save_returns(returns, epi_lens)
        self.learning_curve(returns, epi_lens, save_fig=True)

        print(f"Finished in {duration}")

    def _run_init_policy_test(self):
        """ N.B: Use only for minimum-time tasks """
        timeouts = [1, 2, 5, 10, 25, 50, 100, 500, 1000, 5000, 10000]
        total_steps = 20000
        steps_record = open(f"{self.args.env}_steps_record.txt", 'w')
        hits_record = open(f"{self.args.env}_random_stat.txt", 'w')

        for timeout in tqdm(timeouts):
            for seed in range(30):
                self.args.seed = seed
                self.set_seed()

                steps_record.write(f"timeout={timeout}, seed={seed}: ")
                # Experiment
                hits = 0
                steps = 0
                epi_steps = 0
                obs = self.env.reset()
                while steps < total_steps:
                    action = self.learner.sample_action(obs)
                    # action = np.random.normal(size=self.env.action_space.shape)
                    
                    # Receive reward and next state            
                    _, _, done, _ = self.env.step(action)
                    
                    # print("Step: {}, Next Obs: {}, reward: {}, done: {}".format(steps, next_obs, reward, done))

                    # Log
                    steps += 1
                    epi_steps += 1

                    # Termination
                    if done or epi_steps == timeout:
                        if 'dm_reacher' or 'point_maze' in self.args.env:
                            self.env.reset(randomize_target=done)
                        else:
                            self.env.reset()
                            
                        epi_steps = 0

                        if done:
                            hits += 1
                        else:
                            steps += 20
                            
                        steps_record.write(str(steps)+', ')

                steps_record.write('\n')
                hits_record.write(f"timeout={timeout}, seed={seed}: {hits}\n")
            
        steps_record.close()
        hits_record.close()

def main():
    runner = SACExperiment()
    runner.run()

if __name__ == "__main__":
    main()
