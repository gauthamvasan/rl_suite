import numpy as np
import cv2
import gym
from gym.spaces import Box
from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)
from collections import deque
from rl_suite.envs.env_utils import Observation

class MJReacherWrapper(gym.Wrapper):
    def __init__(self, tol, penalty=-1, use_image=False, img_history=3):
        super().__init__(gym.make('Reacher-v2').unwrapped)
        self._tol = tol
        self.reward = penalty
        self._use_image = use_image
        if use_image:
            self._image_buffer = deque([], maxlen=img_history)
            print(f'Visual mujoco reacher tol={tol}')
        else:
            print(f'Non visual mujoco reacher tol={tol}')

        # remember to reset 
        self._reset = False
        
    def step(self, a):
        assert self._reset

        x, _, done, info = self.env.step(a)
        dist_to_target = np.linalg.norm(x[-3:])

        reward = self.reward
        if dist_to_target <= self._tol:
            done = 1
        else:
            done = 0

        if self._use_image:
            next_obs = Observation()
            next_obs.proprioception = self._make_obs(x)
            new_img = self._get_new_img()
            self._image_buffer.append(new_img)
            next_obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            next_obs = self._make_obs(x)

        if done:
            self._reset = False
        info['dist_to_target'] = dist_to_target
        return next_obs, reward, done, info

    def reset(self):
        if self._use_image:
            obs = Observation()
            obs.proprioception = self._make_ob(self.env.reset())
            new_img = self._get_new_img()
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_img)

            obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            obs = self._make_obs(self.env.reset())
            
        self._reset = True
        return obs

    def _get_new_img(self):
        img = self.env.render(mode='rgb_array')
        img = img[150:400, 50:450, :]
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = np.transpose(img, [2, 0, 1]) # c, h, w

        return img
    
    def _make_obs(self, ob):
        if not self._use_image:
            return ob

        return np.array(list(ob[0:4])+list(ob[6:8]))

    @property
    def image_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')

        image_shape = (3 * self._image_buffer.maxlen, 125, 200)
        return Box(low=0, high=255, shape=image_shape)

    @property
    def proprioception_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')
        
        low = list(self.env.observation_space.low[0:4]) + list(self.env.observation_space.low[6:8])
        high = list(self.env.observation_space.high[0:4]) + list(self.env.observation_space.high[6:8])
        
        return Box(np.array(low), np.array(high))
   
    def close(self):
        super().close()
        
        del self

if __name__ == '__main__':
    import torch
    print(torch.__version__)
    env = ReacherWrapper(0.009, (9, 125, 200), image_period = 3)
    img, ob = env.reset()

    img = np.transpose(img, [1, 2, 0])
    cv2.imshow('', img[:,:,6:9])
    cv2.waitKey(0)
    episode_step = 0
    while True:
        a = env.action_space.sample()
        img, ob, reward, done, info = env.step(a)

        episode_step += 1

        img = np.transpose(img, [1, 2, 0])
        cv2.imshow('', img[:,:,6:9])
        cv2.waitKey(30)

        if done or episode_step == 50:
            
            episode_step = 0
            img, ob = env.reset()
            img = np.transpose(img, [1, 2, 0])
            cv2.imshow('', img[:,:,6:9])
            cv2.waitKey(0)