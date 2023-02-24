"""
Simple wrapper for DeepMind control suite tasks 
- https://github.com/deepmind/dm_control/tree/main/dm_control
"""

from asyncio import tasks
from socket import timeout
import torch
import numpy as np

from dm_control import suite
from collections import deque
from rl_suite.envs.dm_control_wrapper import DMControlBaseEnv, Observation


class DMReacher(DMControlBaseEnv):
    def __init__(self, seed, mode="easy", timeout=1000, use_image=False, img_history=3):
        assert mode in ["easy", "hard"]
        self.timeout = timeout

        self.env = suite.load(domain_name="reacher", task_name=mode, task_kwargs={'random': seed})

        self._obs_dim = 4 if use_image else 6
        self._action_dim = 2

        self._use_image = use_image
        if use_image:
            self._image_buffer = deque([], maxlen=img_history)
            print(f'Visual dm reacher {mode}')
        else:
            print(f'Non visual dm reacher {mode}')
        
        self.steps = 0

    def make_obs(self, x):
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        obs[:2] = x.observation['position'].astype(np.float32)
        obs[2:4] = x.observation['velocity'].astype(np.float32)

        if not self._use_image: # this should be inferred from image
            obs[4:6] = x.observation['to_target'].astype(np.float32)
        
        return obs

    def _get_new_img(self):
        img = self.env.physics.render()
        img = img[85:155, 110:210, :]
        img = np.transpose(img, [2, 0, 1])  # c, h, w
        return img

    def reset(self):
        self.steps = 0
        if self._use_image:
            obs = Observation()
            obs.proprioception = self.make_obs(self.env.reset())
            new_img = self._get_new_img()
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_img)

            obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            obs = self.make_obs(self.env.reset())

        return obs

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

        x = self.env.step(action)

        self.steps +=  1
        reward = x.reward
        done = self.step == self.timeout
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


def interaction():
    seed = 42
    timeout = 100
    total_steps = 20000

    env = DMReacher(seed=seed, timeout=timeout)
    env.reset()

    steps = 0
    epi_steps = 0
    hits = 0
    n_episodes = 0
    ret = 0
    while steps < total_steps:
        action = np.random.normal(size=env.action_space.shape)

        # Receive reward and next state            
        next_obs, r, done, info = env.step(action)
        
        # if epi_steps % 10 == 0:
            # print(f"Step: {epi_steps}, Next Obs: {next_obs}, reward: {r}, done: {done}")

        # Log
        steps += 1
        epi_steps += 1
        ret += r

        # Termination
        if done or epi_steps == timeout:
            n_episodes += 1
            print(f"Episode {n_episodes} finished with steps: {epi_steps} and return: {ret}. {done}")
            env.reset()
            epi_steps = 0
            ret = 0
            if done:
                hits += 1
            else:
                steps += 20

if __name__ == "__main__":
    interaction()
