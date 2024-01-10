import gym
import torch
import numpy as np

from dm_control import suite
from dm_control.rl.control import flatten_observation
from rl_suite.envs import Observation
from rl_suite.envs.dm_control_wrapper import DMControlBaseEnv

class FingerSpin(DMControlBaseEnv):
    def __init__(self, seed):
        self.env = suite.load(domain_name="finger", task_name="spin", 
                              task_kwargs={'random': seed})
        
        # Observation space
        self._obs_dim = 0
        for key, val in self.env.observation_spec().items():
            self._obs_dim += val.shape[0]
        
        # Action space
        self._action_dim = self.env.action_spec().shape[0]
    
    def make_obs(self, x):
        obs = []
        for _, val in x.items():
            obs.append(val.ravel())
        return np.concatenate(obs)


    def reset(self):
        time_step = self.env.reset()
        obs = self.make_obs(time_step.observation)
        return obs
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

        x = self.env.step(action)

        reward = x.reward
        done = x.last()
        info = {}
        next_obs = self.make_obs(x.observation)

        return next_obs, reward, done, info


class FingerTurn(FingerSpin):
    def __init__(self, task, seed):
        self.env = suite.load(domain_name="finger", task_name=task, 
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


if __name__ == "__main__":
    # env = FingerSpin(seed=42)
    env = FingerTurn(task="turn_easy", seed=42)
    
    EP = 50 
    rets = []
    ep_lens = []
    for i in range(EP):
        obs = env.reset()
        done = False
        steps = 0
        ret = 0
        while not done:
            action = np.random.uniform(-1, 1, size=env._action_dim)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            # print(next_obs, reward, done)
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
