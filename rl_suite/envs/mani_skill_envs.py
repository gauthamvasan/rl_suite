""" Adaptation of some envs from ManiSkill2: https://github.com/haosulab/ManiSkill2 """

import gym
import cv2
import time
import mani_skill2.envs # This registers all envs

import numpy as np
from rl_suite.envs import Observation
from rl_suite.envs.dm_control_wrapper import DMControlBaseEnv
from gym.spaces import Box
from collections import deque


class PickCube(DMControlBaseEnv):   
    def __init__(self, seed, obs_mode="rgbd", control_mode="pd_joint_vel", use_image=False, img_history=3) -> None:
        """
        control_mode: ["pd_joint_delta_pos", "pd_joint_vel"]
        """
        super().__init__()

        self.env = gym.make("PickCube-v0", obs_mode=obs_mode, control_mode=control_mode)
        self.env.seed(seed)

        """
        N.B: There is something weird with the way the simulator is set up. 
        While there the action vector is only 8-dimensional, there pos and vel vector in 
        the observation is 9-dimensional. This is because, the last two values represent 
        the positions of each gripper finger rather than one value to denote the distance
        between the gripper fingers. 
        """
        self._obs_dim = 18 if use_image else 21
        self._action_dim = self.env.action_space.shape[0]    # 8 dimensional action
        self._use_image = use_image

        if use_image:
            self._image_buffer = deque([], maxlen=img_history)
            print(f'Visual PickCube-v0 {control_mode}')
        else:
            print(f'Non-visual PickCube-v0 {control_mode}')

    def _get_image(self, obs_dict):
        # Horizontal stack. Returns a concatenated image of size (128, 256, 3). Base camera (left), hand camera (right)
        img = np.concatenate((obs_dict["image"]["base_camera"]["rgb"], obs_dict["image"]["hand_camera"]["rgb"]), axis=1)
        img = np.transpose(img, [2, 0, 1]) # (3, 128, 256)
        return img

    def make_obs(self, obs_dict):
        """
            obs_dict (dict): Dictionary of observation with the following structure:
                    obs_dict = {
                                'agent': {'qpos', 'qvel', 'base_pose'} 
                                'extra': {'tcp_pose', 'goal_pos'}
                                'camera_param': {'extrinsic_cv', 'cam2world_gl', 'intrinsic_cv'}
                                'image': {'base_camera', 'hand_camera'}
                    }
        """
        
        prop = np.zeros(self._obs_dim, dtype=np.float32)
        prop[:9] = obs_dict['agent']['qpos'].astype(np.float32)
        prop[9:18] = obs_dict['agent']['qvel'].astype(np.float32)
        if not self._use_image:
           prop[-3:] = obs_dict['extra']['goal_pos'].astype(np.float32)
           obs = prop
        else:
            obs = Observation()
            obs.proprioception = prop

            new_img = self._get_image(obs_dict)
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_img)

            obs.images = np.concatenate(self._image_buffer, axis=0)

        return obs
    
    def reset(self):
        return self.make_obs(self.env.reset())
    
    def step(self, action):
        next_obs_dict, reward, done, info = self.env.step(action)
        next_obs = self.make_obs(next_obs_dict)

        return next_obs, reward, done, info
    
    @property
    def image_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')

        image_shape = (3 * self._image_buffer.maxlen, 70, 100)
        return Box(low=0, high=255, shape=image_shape)


def visualize_pick_cube():
    # env = gym.make("PickCube-v0", obs_mode="rgbd", control_mode="pd_joint_delta_pos")
    env = gym.make("PickCube-v0", obs_mode="rgbd", control_mode="pd_joint_vel")
    # print("Observation space", env.observation_space)
    # print("Action space", env.action_space)

    env.seed(0)  # specify a seed for randomness
    obs = env.reset()
    print(obs.keys())
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        steps += 1
        # print(f"Step: {steps}, obs: {obs['agent']['qpos'][:2]}, reward: {reward}, done: {done}")
        # env.render()  # a display is required to render
        
        # Horizontal stack
        img = np.concatenate((obs["image"]["base_camera"]["rgb"], obs["image"]["hand_camera"]["rgb"]), axis=1)
        print(obs['agent']['qpos'], len(action))
        cv2.imshow("", img)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        time.sleep(0.05)
        obs = next_obs

    env.close()

def main():
    env = PickCube(seed=42, use_image=True)

    obs = env.reset()
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        # print(obs.images.shape, reward, done, steps)
        steps += 1
        
        cv2.imshow("", np.transpose(obs.images[6:9, :, :], [1, 2, 0]))
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        time.sleep(0.05)
        obs = next_obs

    print(f"Episode ended in {steps}")

if __name__ == "__main__":
    main()
