import gym
import torch
import numpy as np

from dm_control import suite
from dm_control.rl.control import flatten_observation
from rl_suite.envs import Observation
from gym.spaces.box import Box

ENV_MAP = {
    "acrobot":                  {'domain': "acrobot", "task": "swingup"},
    "acrobot_sparse":           {'domain': "acrobot", "task": "swingup_sparse"},
    "ball_in_cup":              {"domain": "ball_in_cup", "task": "catch"},
    "cartpole_balance":         {"domain": "cartpole", "task": "balance"},
    "cartpole_balance_sparse":  {"domain": "cartpole", "task": "balance_sparse"},
    "cartpole_swingup":         {"domain": "cartpole", "task": "swingup"},
    "cartpole_swingup_sparse":  {"domain": "cartpole", "task": "swingup_sparse"},
    "cheetah":                  {"domain": "cheetah", "task": "run"},
    "finger_spin":              {"domain": "finger", "task": "spin"},
    "finger_turn_easy":         {"domain": "finger", "task": "turn_easy"},
    "finger_turn_hard":         {"domain": "finger", "task": "turn_hard"},
    "fish_upright":             {"domain": "fish", "task": "upright"},
    "fish_swim":                {"domain": "fish", "task": "swim"},
    "hopper_stand":             {"domain": "hopper", "task": "stand"},
    "hopper_hop":               {"domain": "hopper", "task": "hop"},
    "humanoid_stand":           {"domain": "humanoid", "task": "stand"},
    "humanoid_walk":            {"domain": "humanoid", "task": "walk"},
    "humanoid_run":             {"domain": "humanoid", "task": "run"},
    "manipulator_bring_ball":   {"domain": "manipulator", "task": "bring_ball"},
    "pendulum_swingup":         {"domain": "pendulum", "task": "swingup"},
    "point_mass_easy":          {"domain": "point_mass", "task": "easy"},
    "reacher_easy":             {"domain": "reacher", "task": "easy"},
    "reacher_hard":             {"domain": "reacher", "task": "hard"},
    "swimmer6":                 {"domain": "swimmer", "task": "swimmer6"},
    "swimmer15":                {"domain": "swimmer", "task": "swimmer15"},
    "walker_stand":             {"domain": "walker", "task": "stand"},
    "walker_walk":              {"domain": "walker", "task": "walk"},
    "walker_run":               {"domain": "walker", "task": "run"},
}


class DMControl():
    def __init__(self, env_name, seed):
        self.env = suite.load(domain_name=ENV_MAP[env_name]['domain'], task_name=ENV_MAP[env_name]['task'], 
                              task_kwargs={'random': seed})
        
        # Observation space
        self._obs_dim = 0
        for key, val in self.env.observation_spec().items():
            if val.shape:
                self._obs_dim += val.shape[0]
            else:
                self._obs_dim += 1
        
        # Action space
        self._action_dim = self.env.action_spec().shape[0]
    
    def make_obs(self, x):
        obs = []
        for _, val in x.items():
            obs.append(val.ravel())
        return np.concatenate(obs)

    def reset(self, **kwargs):
        time_step = self.env.reset()
        obs = self.make_obs(time_step.observation)
        return obs, {}
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

        x = self.env.step(action)

        reward = x.reward
        terminated = x.last()
        truncated = False
        info = {}
        next_obs = self.make_obs(x.observation)

        return next_obs, reward, terminated, truncated, info

    @property
    def observation_space(self):
        return Box(shape=(self._obs_dim,), high=10, low=-10)

    @property
    def image_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')

        image_shape = (3 * self._image_buffer.maxlen, 100, 120)
        return Box(low=0, high=255, shape=image_shape)

    @property
    def proprioception_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')
        
        return self.observation_space

    @property
    def action_space(self):
        return Box(shape=(self._action_dim,), high=1, low=-1)

    def render(self):
        self.env.render()
    

if __name__ == "__main__":
    env = DMControl(env_name="finger_turn_easy", seed=42)    
    EP = 50 
    rets = []
    ep_lens = []
    for i in range(EP):
        obs = env.reset()
        terminated = False
        steps = 0
        ret = 0
        while not terminated:
            action = np.random.uniform(-1, 1, size=env._action_dim)
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs
            # print(next_obs, reward, terminated)
            steps += 1 
            ret += reward
        rets.append(ret)
        ep_lens.append(steps)
        print("Episode: {} ended in {} steps with return: {}".format(i+1, steps, ret))

    # Random policy stats
    rets = np.array(rets)
    ep_lens = np.array(ep_lens)
    print("Mean: {:.2f}".format(np.mean(rets)))
    print("Standard Error: {:.2f}".format(np.std(rets) / np.sqrt(len(rets) - 1)))
    print("Median: {:.2f}".format(np.median(rets)))
    print("Max length:", max(ep_lens))
    print("Min length:", min(ep_lens))
