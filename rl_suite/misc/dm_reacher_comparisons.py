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


class VelTolReacher(ReacherWrapper):
    def __init__(self, seed, penalty=-1, mode="easy", use_image=False, img_history=3, vel_tol=None):
        super().__init__(seed, penalty, mode, use_image, img_history)
        if vel_tol is None:
            if mode in ["easy", "hard"]:
                self.vel_min = np.array([-0.43, -1.19])
                self.vel_max = np.array([0.52, 1.17])
            # elif mode == "hard":  ### For a stricter constraint.
                # self.vel_min = np.array([-0.29, -0.9])
                # self.vel_max = np.array([0.29, 0.89])
        else:
            if isinstance(vel_tol, list):
                vel_tol = np.array(vel_tol)
            assert (vel_tol > 0).all()
            self.vel_min = -vel_tol
            self.vel_max = vel_tol

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

        x = self.env.step(action)

        reward = self.reward
        info = {}

        if self._use_image:
            next_obs = Observation()
            next_obs.proprioception = self.make_obs(x)
            vel = next_obs.proprioception[2:4]
            new_img = self._get_new_img()
            self._image_buffer.append(new_img)
            next_obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            next_obs = self.make_obs(x)
            vel = next_obs[2:4]
        
        within_vel = ((vel[0] > self.vel_min[0]) and (vel[0] < self.vel_max[0])) and ((vel[1] > self.vel_min[1]) and (vel[1] < self.vel_max[1]))
        done = x.reward and within_vel

        return next_obs, reward, done, info


class AdditiveRewardReacher(ReacherWrapper):
    def __init__(self, seed, mode="easy", use_image=False, img_history=3):
        super().__init__(seed=seed, mode=mode, use_image=use_image, img_history=img_history)
        self.timeout = 1000
        self.steps = 0

    def reset(self):
        self.steps = 0
        return super().reset()

    def compute_reward(self, x, action):
        """ 
        Source: https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/mujoco/reacher.py#L28C23-L28C42
        reward = reward_dist + reward_ctrl
        """
        if x.reward:
            return 1

        distance = self.env._physics.finger_to_target_dist()
        # reward = -distance + np.exp(-100 * (distance**2))
        return -distance

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

        x = self.env.step(action)
        self.steps +=  1

        reward = self.compute_reward(x, action)
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


class BrockmanTassaReacher(AdditiveRewardReacher):
    def __init__(self, seed, mode="easy", use_image=False, img_history=3):
        super().__init__(seed=seed, mode=mode, use_image=use_image, img_history=img_history)
    
    def compute_reward(self, x, action):
        if x.reward:
            return 1
        reward = -self.env._physics.finger_to_target_dist() + -np.square(action).sum()
        return reward


class FixedTimeLimitReacher(ReacherWrapper):
    def __init__(self, seed, mode="easy", use_image=False, img_history=3):
        super().__init__(seed=seed, mode=mode, use_image=use_image, img_history=img_history)
        self.timeout = 1000
        self.steps = 0

    def reset(self):
        self.steps = 0
        return super().reset()

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
        self.args = self.parse_args()
        
        self.run_id = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        self.run_id += f"_seed-{self.args.seed}"
        self._expt_dir = os.path.join(self.args.results_dir, self.args.env)
        self.make_dir(self._expt_dir)
        
        # Make env
        if self.args.env == "vt_reacher_easy":
            self.env = VelTolReacher(seed=self.args.seed, penalty=self.args.reward, mode="easy", use_image=self.args.use_image)
        elif self.args.env == "vt_reacher_hard":
            self.env = VelTolReacher(seed=self.args.seed, penalty=self.args.reward, mode="hard", use_image=self.args.use_image)
        elif self.args.env == "ar_reacher_easy":
            self.env = AdditiveRewardReacher(seed=self.args.seed, mode="easy", use_image=self.args.use_image)
        elif self.args.env == "ar_reacher_hard":
            self.env = AdditiveRewardReacher(seed=self.args.seed, mode="hard", use_image=self.args.use_image)
        elif self.args.env == "ftl_reacher_easy":
            self.env = FixedTimeLimitReacher(seed=self.args.seed, mode="easy", use_image=self.args.use_image)
        elif self.args.env == "ftl_reacher_hard":
            self.env = FixedTimeLimitReacher(seed=self.args.seed, mode="hard", use_image=self.args.use_image)
        elif self.args.env == "bt_reacher_easy":
            self.env = BrockmanTassaReacher(seed=self.args.seed, mode="easy", use_image=self.args.use_image)
        elif self.args.env == "bt_reacher_hard":
            self.env = BrockmanTassaReacher(seed=self.args.seed, mode="hard", use_image=self.args.use_image)
        
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


def main():
    runner = DMReacherComparison()
    runner.run()

if __name__ == "__main__":
    main()
    