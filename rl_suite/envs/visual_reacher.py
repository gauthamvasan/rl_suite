import cv2
import gym

import numpy as np

from collections import deque
from gym.spaces import Box
from rl_suite.envs.env_utils import Observation


class VisualMujocoReacher2D(gym.Wrapper):
    def __init__(self, tol, penalty, img_history=3, image_period=1, control_mode='accel'):
        """

        Args:
            tol (float): Smaller the value, smaller the target size (e.g., 0.009, 0.018, 0.036, 0.072, etc.)
            img_history (int): Number of images used in obs
            image_period (int): Update image obs only every 'image_period' steps
        """
        super().__init__(gym.make('Reacher-v2').unwrapped)
        self._tol = tol
        self._image_period = image_period
        low = list(self.env.observation_space.low[0:4]) + list(self.env.observation_space.low[6:8])
        high = list(self.env.observation_space.high[0:4]) + list(self.env.observation_space.high[6:8])
        self.proprioception_space = Box(np.array(low), np.array(high))

        self._image_buffer = deque([], maxlen=img_history)
        self._use_image = True
        self.image_shape = (3 * img_history, 125, 200)

        self.image_space = Box(low=0, high=255, shape=self.image_shape)

        # remember to reset
        self._latest_image = None
        self._reset = False
        self._step = 0
        self.penalty = penalty

    def reset(self):
        prop = self.env.reset()
        prop = self._get_ob(prop)

        if self._use_image:
            new_img = self._get_new_img()
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_img)

            self._latest_image = np.concatenate(self._image_buffer, axis=0)

        self._reset = True
        self._step = 0

        obs = Observation()
        obs.images = self._latest_image
        obs.proprioception = prop
        obs.metadata = [self._step]
        return obs

    def step(self, a):
        assert self._reset

        prop, _, done, _, info = self.env.step(a)
        prop = self._get_ob(prop)
        self._step += 1

        dist_to_target = -info["reward_dist"]

        reward = -self.penalty
        if dist_to_target <= self._tol:
            info['reached'] = True
            done = True

        if self._use_image and (self._step % self._image_period) == 0:
            new_img = self._get_new_img()
            self._image_buffer.append(new_img)
            self._latest_image = np.concatenate(self._image_buffer, axis=0)

        if done:
            self._reset = False

        obs = Observation()
        obs.images = self._latest_image
        obs.proprioception = prop
        obs.metadata = [self._step]

        return obs, reward, done, info

    def _get_new_img(self):
        img = self.env.render(mode='rgb_array')
        img = img[150:400, 50:450, :]
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = np.transpose(img, [2, 0, 1])  # c, h, w
        return img

    def _get_ob(self, ob):
        return np.array(list(ob[0:4]) + list(ob[6:8]))

    def close(self):
        super().close()
        del self


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt

    print(torch.__version__)
    env = VisualMujocoReacher2D(0.072, img_history=3, image_period=3, penalty=1)
    obs = env.reset()
    img = np.transpose(obs.images, [1, 2, 0])
    # create two subplots
    plt.ion()
    ax1 = plt.subplot(1, 1, 1)
    im1 = ax1.imshow(img[:, :, 6:9])

    waitKey = 1
    while True:
        im1.set_data(img[:, :, 6:9])
        plt.pause(0.05)
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        print(obs.proprioception)
        img = np.transpose(obs.images, [1, 2, 0])
        if done:
            env.reset()
    plt.show()
