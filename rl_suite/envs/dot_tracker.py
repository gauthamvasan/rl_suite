import sys
import os
import random
import time
import pygame

import numpy as np

from rl_suite.envs import Observation
from rl_suite.arp import ARProcess
from gym.core import Env
from gym.spaces.box import Box
from collections import deque

#instructions to windows to center the game window in the center of
#the screen, which it might ignore
os.environ["SDL_VIDEO_CENTERED"] = "1"


class DotTracker(Env):
    def __init__(self, dt=0.2, timeout=10000, pos_tol=0.05, penalty=-1, 
                 seed=42, use_image=False, img_history=3) -> None:
        super().__init__()
        self.dt = dt
        self.pos_tol = pos_tol
        self.timeout = timeout
        self.penalty = penalty
        self.use_image = use_image
        
        # Screen dimensions
        self.width = self.height = 1000
        self.screen_size = [self.width, self.height]
        self.bg_color = pygame.Color("darkslategrey")        

        # Circle 
        self.radius = int(0.04 * self.width)
        self.circle_color = pygame.Color("navajowhite")

        # Target 
        self.target_width = int(0.08 * self.width)
        self.target_color = pygame.Color("lightcoral")
        self.arp_dt = 0.4 

        self.steps = 0
        self.seed(seed)

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Dot Tracker")

        if use_image:
            self._image_buffer = deque([], maxlen=img_history)            

    def seed(self, seed):
        np.random.seed(seed)
    
    def reset(self):
        self.steps = 0
        self.pos = np.random.uniform(-1, 1, 2)
        self.vel = np.zeros(2)

        self.ar = ARProcess(3, 0.8, 2)
        self.prev_ar_pos = self.ar.step()[0]
        self.target_pos = np.random.uniform(-1, 1, 2)
        
        if self.use_image:
            obs = Observation()
            img = pygame.surfarray.array3d(self.screen)
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(img)
            obs.images = np.concatenate(self._image_buffer, axis=0)
            obs.proprioception = np.concatenate((self.pos, self.vel))
        else:
            obs = np.concatenate((self.pos, self.vel, self.target_pos))
        return obs

    def step(self, action):
        """

        Args:
            action: 2-D Tensor (vals between [-1, 1])

        Returns:

        """
        self.steps += 1

        action = np.clip(action, a_min=-1, a_max=1)

        # Acceleration control for smoothness
        self.pos = self.pos + self.vel * self.dt + 0.5 * action * self.dt ** 2
        self.vel[self.pos < -1] = -0.1 * self.vel[self.pos < -1]
        self.vel[self.pos > 1] = -0.1 * self.vel[self.pos > 1]
        self.pos = np.clip(self.pos, -1, 1)

        # Target pos
        current_ar_pos = self.ar.step()[0]
        self.target_pos += self.arp_dt * (current_ar_pos - self.prev_ar_pos)
        self.target_pos = np.clip(self.target_pos, -1, 1)
        self.prev_ar_pos = current_ar_pos

        self.vel += action * self.dt

        ###
        self.screen.fill(self.bg_color)
    
        # Keep circle within boundary visually
        pos = np.clip(self.pos, -0.92, 0.92)
        pixel_pos = self.pos_to_pixel(pos)
        
        # Keep target within boundary visually
        target_pos = self.pos_to_pixel(self.target_pos)
        target_pos = np.clip(target_pos, 0, self.width - 0.08*self.width)
        
        pygame.draw.rect(self.screen, self.target_color, target_pos.tolist() + [self.target_width, self.target_width])
        pygame.draw.circle(self.screen, self.circle_color, pixel_pos, self.radius)
        ###

        # Observation
        if self.use_image:
            next_obs = Observation()
            next_obs.proprioception = np.concatenate((self.pos, self.vel))            
            self._image_buffer.append(pygame.surfarray.array3d(self.screen))
            next_obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            next_obs = np.concatenate((self.pos, self.vel, self.target_pos))

        # Reward
        reward = self.penalty

        # Done
        done = np.allclose(self.pos, self.target_pos, atol=self.pos_tol)

        # Metadata
        info = {}

        return next_obs, reward, done, info
    
    def pos_to_pixel(self, pos):
        pixel_pos = (((pos + 1) * self.width)/2)
        return pixel_pos

    def render(self):
        pygame.display.update()

    @property
    def action_space(self):
        # TODO: Enforce this constraint
        return Box(low=-1, high=1, shape=(2,))

    @property
    def observation_space(self):
        # TODO: Verify that min/max velocity are always within these bounds
        return Box(low=-10, high=10, shape=(4,))

    def close(self) -> None:
        pygame.quit()
        return super().close()


if __name__ == "__main__":       
    n_episodes = 100
    timeout = 20000
    seed = 42
    env = DotTracker(dt=0.2, timeout=timeout, pos_tol=0.1, use_image=False) 

    for i_episode in range(n_episodes):
        obs = env.reset() 
        done = False
        steps = 0
        ret = 0
        ep_len = 0
        while not done and steps < timeout:
            env.render()
            time.sleep(0.05)
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            # print(f"Step: {steps}, Obs: {obs[:2]}, reward: {reward}, done: {done}")
            obs = next_obs
            steps += 1
            ret += reward
            ep_len += 1
        
        print(f"Episode {i_episode+1} took {ep_len} steps and ended with return {ret}. Total steps: {steps}")

    env.close()
