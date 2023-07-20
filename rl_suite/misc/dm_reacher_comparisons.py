import os
import torch
import numpy as np

from datetime import datetime
from rl_suite.envs.dm_control_wrapper import ReacherWrapper
from rl_suite.envs import Observation
from rl_suite.sac_experiment import SACExperiment
from rl_suite.algo.sac import SACAgent
from rl_suite.algo.sac_rad import SACRADAgent
from rl_suite.algo.replay_buffer import SACReplayBuffer, SACRADBuffer


class EuclideanReacher(ReacherWrapper):
    def __init__(self, seed, penalty=-1, mode="easy", use_image=False, img_history=3):
        super().__init__(seed, penalty=penalty, mode=mode, use_image=use_image, img_history=img_history)
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

        x = self.env.step(action)

        """ 
        Source: https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/mujoco/reacher.py#L28C23-L28C42
        reward = reward_dist + reward_ctrl
        """
        reward = -self.env._physics.finger_to_target_dist() + -np.square(action).sum()
        done = x.reward
        info = {}

        if self._use_image:
            next_obs = Observation()
            next_obs.proprioception = self.make_obs(x)
            new_img = self._get_new_img()
            self._image_buffer.append(new_img)
            next_obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            next_obs = self.make_obs(x)
            
        return next_obs, reward, done, info


class DMReacher(ReacherWrapper):
    def __init__(self, seed, penalty=-1, mode="easy", use_image=False, img_history=3):
        super().__init__(seed, penalty, mode, use_image, img_history)
        self.timeout = 1000

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

        x = self.env.step(action)

        self.steps +=  1
        reward = x.reward
        done = self.steps == self.timeout
        info = {}

        if self._use_image:
            next_obs = Observation()
            next_obs.proprioception = self.make_obs(x)
            new_img = self._get_new_img()
            self._image_buffer.append(new_img)
            next_obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            next_obs = self.make_obs(x)
            
        return next_obs, reward, done, info
    

class DMReacherComparison(SACExperiment):
    def __init__(self):
        args = self.parse_args()
        
        self.run_id = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        self.run_id += f"_seed-{args.seed}"
        self.args = args
        self._expt_dir = os.path.join(args.results_dir, args.env)
        self.make_dir(self._expt_dir)
        
        # Make env
        
        
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
