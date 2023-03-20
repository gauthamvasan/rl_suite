import cv2
import gym
import torch

import numpy as np
from gym.spaces import Box
from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)
from collections import deque
from rl_suite.envs import Observation
from tqdm import tqdm
from math import pi


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
    
    def _reset_model_agent_only(self):
        qpos = (
            self.env.np_random.uniform(low=-pi, high=pi, size=self.env.model.nq)
        )
        
        qpos[-2:] = self.env.goal
        qvel = self.env.init_qvel + self.env.np_random.uniform(
            low=-0.005, high=0.005, size=self.env.model.nv
        )
        qvel[-2:] = 0
        self.env.set_state(qpos, qvel)
        return self.env._get_obs()

    def _reset_agent_only(self):
        self.env.sim.reset()
        ob = self._reset_model_agent_only()
        return ob
        
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

    def reset(self, randomize_target=True):
        if self._use_image:
            obs = Observation()
            if randomize_target:
                obs.proprioception = self._make_obs(self.env.reset())
            else:
                obs.proprioception = self._make_obs(self._reset_agent_only())

            new_img = self._get_new_img()
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_img)

            obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            if randomize_target:
                obs = self._make_obs(self.env.reset())
            else:
                obs = self._make_obs(self._reset_agent_only())
            
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

def ranndom_policy_hits_vs_timeout():
    total_steps = 20000
    
    tol = 0.009
    env = MJReacherWrapper(tol=tol)
    steps_record = open(f"mj reacher tol={tol}_steps_record.txt", 'w')
    hits_record = open(f"mj reacher tol={tol}_random_stat.txt", 'w')

    for timeout in tqdm([1, 2, 5, 10, 25, 50, 100, 500, 1000, 5000]):
        for seed in range(30):
            torch.manual_seed(seed)
            np.random.seed(seed)

            steps_record.write(f"timeout={timeout}, seed={seed}: ")
            # Experiment
            hits = 0
            steps = 0
            epi_steps = 0
            env.reset()
            while steps < total_steps:
                action = np.random.normal(size=env.action_space.shape)

                # Receive reward and next state            
                _, _, done, _ = env.step(action)
                
                # print("Step: {}, Next Obs: {}, reward: {}, done: {}".format(steps, next_obs, reward, done))

                # Log
                steps += 1
                epi_steps += 1

                # Termination
                if done or epi_steps == timeout:
                    env.reset()
                        
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

if __name__ == '__main__':
    ranndom_policy_hits_vs_timeout()

    # env = MJReacherWrapper(tol=0.036, use_image=True)
    # obs = env.reset()
    # img = obs.images

    # print(img.shape)
    # img_to_show = np.transpose(img, [1, 2, 0])
    # img_to_show = img_to_show[:,:,-3:]
    # cv2.imshow('', img_to_show)
    # cv2.waitKey(0)

    # for t in range(1000):
    #     randomize_target = t % 100 == 0
            
    #     next_obs = env.reset(randomize_target=randomize_target)
    #     next_img = next_obs.images
    #     img_to_show = np.transpose(next_img, [1, 2, 0])
    #     img_to_show = img_to_show[:,:,-3:]
    #     cv2.imshow('', img_to_show)
    #     cv2.waitKey(0 if randomize_target else 50)
        
