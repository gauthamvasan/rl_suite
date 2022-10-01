import cv2
import torch
import gym

import numpy as np
import matplotlib.pyplot as plt

from tkinter import N
from rl_suite.envs.env_utils import Observation
from dm_control import suite
from gym.spaces import Box
from collections import deque
from statistics import mean

class BallInCupWrapperFixReset:
    def __init__(self, seed, timeout, penalty=-1, use_image=False, img_history=3):
        """ Outputs state transition data as torch arrays """
        self.env = suite.load(domain_name="ball_in_cup", task_name="catch", task_kwargs={'random': seed})
        self._timeout = timeout
        self.reward = penalty
        self._obs_dim = 8 if not use_image else 4
        self._action_dim = 2

        self._use_image = use_image
        
        if use_image:
            self._image_buffer = deque([], maxlen=img_history)
            print("Visual ball in cup")
        else:
            print('Non visual ball in cup')

    def make_obs(self, x):
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        if not self._use_image:
            obs[:4] = x.observation['position'].astype(np.float32)
            obs[4:8] = x.observation['velocity'].astype(np.float32)
        else:
            obs[:4] = x.observation['velocity'].astype(np.float32)
        return obs

    def _get_new_img(self):
        img = self.env.physics.render()
        img = img[20:120, 100:220, :]
        img = np.transpose(img, [2, 0, 1])  # c, h, w
        return img

    def reset(self):
        self.steps = 0
        x = self.env.reset()
        while x.observation['position'][-1] > 0.33:
            x = self.env.reset()

        if self._use_image:
            obs = Observation()
            obs.proprioception = self.make_obs(x)
            new_img = self._get_new_img()
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_img)

            obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            obs = self.make_obs(x)

        return obs

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()
        self.steps += 1

        x = self.env.step(action)

        reward = self.reward
        done = x.reward # or self.steps == self._timeout
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

class BallInCupWrapper:
    def __init__(self, seed, timeout, penalty=-1, use_image=False, img_history=3):
        """ Outputs state transition data as torch arrays """
        self.env = suite.load(domain_name="ball_in_cup", task_name="catch", task_kwargs={'random': seed})
        self._timeout = timeout
        self.reward = penalty
        self._obs_dim = 8 if not use_image else 4
        self._action_dim = 2

        self._use_image = use_image
        
        if use_image:
            self._image_buffer = deque([], maxlen=img_history)
            print("Visual ball in cup")
        else:
            print('Non visual ball in cup')

    def make_obs(self, x):
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        if not self._use_image:
            obs[:4] = x.observation['position'].astype(np.float32)
            obs[4:8] = x.observation['velocity'].astype(np.float32)
        else:
            obs[:4] = x.observation['velocity'].astype(np.float32)
        return obs

    def _get_new_img(self):
        img = self.env.physics.render()
        img = img[20:120, 100:220, :]
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
        self.steps += 1

        x = self.env.step(action)

        reward = self.reward
        done = x.reward # or self.steps == self._timeout
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


class ReacherWrapper(gym.Wrapper):
    def __init__(self, seed, timeout, penalty=-1, mode="easy", use_image=False, img_history=3):
        """ Outputs state transition data as torch arrays """
        assert mode in ["easy", "hard"]
        self.env = suite.load(domain_name="reacher", task_name=mode, task_kwargs={'random': seed})
        self._timeout = timeout

        self._obs_dim = 4 if use_image else 6
        self._action_dim = 2
        
        self.reward = penalty
        self._use_image = use_image
        
        if use_image:
            self._image_buffer = deque([], maxlen=img_history)
            print(f'Visual dm reacher {mode}')
        else:
            print(f'Non visual dm reacher {mode}')

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
        self.steps += 1

        x = self.env.step(action)

        reward = self.reward
        done = x.reward # or self.steps == self._timeout
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

    @property
    def observation_space(self):
        return Box(shape=(self._obs_dim,), high=10, low=-10)

    @property
    def image_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')

        image_shape = (3 * self._image_buffer.maxlen, 70, 100)
        return Box(low=0, high=255, shape=image_shape)

    @property
    def proprioception_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')
        
        return self.observation_space
        
    @property
    def action_space(self):
        return Box(shape=(self._action_dim,), high=1, low=-1)


class ManipulatorWrapper:
    def __init__(self, task_name, seed, timeout=10000):
        """

        Args:
            task_name (str): ["bring_ball", "bring_peg", "insert_ball", "insert_peg"]
            seed (int): Seed for random number generator
            timeout (int): Max episode length
        """
        """ Outputs state transition data as torch arrays """
        if task_name != "bring_ball":
            raise NotImplemented(task_name)

        self.env = suite.load(domain_name="manipulator", task_name=task_name, task_kwargs={'random': seed})
        self._timeout = timeout

    def make_obs(self, x):
        obs = torch.zeros((1, 8), dtype=torch.float32)
        obs[:, :4] = torch.as_tensor(x.observation['position'].astype(np.float32))
        obs[:, 4:] = torch.as_tensor(x.observation['velocity'].astype(np.float32))
        return obs

    def reset(self):
        self.steps = 0
        return self.make_obs(self.env.reset())

    def step(self, action):
        self.steps += 1

        x = self.env.step(action)
        next_obs = self.make_obs(x)
        reward = -0.01
        done = x.reward or self.steps == self._timeout
        info = {}

        return next_obs, reward, done, info

    @property
    def observation_space(self):
        return Box(shape=(8,), high=10, low=-10)

    @property
    def action_space(self):
        return Box(shape=(2,), high=1, low=-1)


def visualize_behavior(domain_name, task_name, seed=1):
    # Load one task
    # env = suite.load(domain_name="dog", task_name="fetch")
    env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})

    # N.B: See suite.ALL_TASKS for a list of tasks
    # Iterate over a task set:
    # for domain_name, task_name in suite.BENCHMARKING: # suite.ALL_TASKS
    #   env = suite.load(domain_name, task_name)

    # Step through an episode and print out reward, discount and observation.
    action_spec = env.action_spec()
    time_step = env.reset()

    # create two subplots
    img = env.physics.render()
    ax1 = plt.subplot(1, 1, 1)
    im1 = ax1.imshow(img)
    while not time_step.last():
        action = np.random.uniform(action_spec.minimum,
                                   action_spec.maximum,
                                   size=action_spec.shape)
        time_step = env.step(action)
        img = env.physics.render()
        im1.set_data(img)
        plt.pause(0.02)
        # print(time_step.reward, time_step.discount, time_step.observation)

def ranndom_policy_done_2_done_length():
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    total_dones = 50
    for timeout in [1, 2, 5, 10, 25, 50, 100, 500, 1000]:
        # Env
        task = 'ball in cup 20'
        env = BallInCupWrapper(seed, timeout=timeout, penalty=-1)
        # env = ReacherWrapper(seed=seed, mode="easy", timeout=timeout)
        if not hasattr(env, "_action_dim"):
            env._action_dim = env.action_spec().shape[0]

        # Experiment
        done_2_done_lens = []
        steps = 0
        while len(done_2_done_lens) < total_dones:
            env.reset()
            epi_steps = 0
            done = 0
            done_2_done_steps = 0
            resets = 0
            while not done: 
                A = env.action_space.sample()

                # Receive reward and next state            
                _, _, done, _ = env.step(A)
                
                # print("Step: {}, Next Obs: {}, reward: {}, done: {}".format(steps, next_obs, reward, done))

                # Log
                steps += 1
                epi_steps += 1
                done_2_done_steps += 1

                # Termination
                if epi_steps == timeout:
                    resets += 1
                    env.reset()
                    epi_steps = 0

            done_2_done_lens.append(done_2_done_steps + resets*20)
            print('-' * 50)
            print("Episode: {}, done_2_done steps: {}, resets: {}, Total Steps: {}".format(len(done_2_done_lens), done_2_done_steps, resets, steps))
            print('-' * 50)
    
        with open(task + '_timeout='+str(timeout)+'_random_stat.txt', 'w') as out_file:
            for length in done_2_done_lens:
                out_file.write(str(length)+'\n')
            
            out_file.write(f"\nMean: {mean(done_2_done_lens)}")
        
def random_policy_stats():
    # Problem
    seed = 0
    timeout = 2
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Env
    task = 'ball in cup'
    env = BallInCupWrapper(seed, timeout=timeout, penalty=-1, use_image=True)
    for _ in range(10000):
        obs = env.reset()
        img_to_show = np.transpose(obs.images, [1,2,0])[:,:,-3:]
        cv2.imshow("", img_to_show)
        cv2.waitKey(0)
        
    # env = ReacherWrapper(seed=seed, mode="hard", timeout=timeout)
    # env = suite.load(domain_name="quadruped", task_name="fetch", task_kwargs={'random': seed})
    # env = suite.load(domain_name="reacher", task_name="easy", task_kwargs={'random': seed})
    if not hasattr(env, "_action_dim"):
        env._action_dim = env.action_spec().shape[0]

    # Experiment
    total_dones = 50
    rets = []
    ep_lens = []
    steps = 0
    dones = 0
    while dones < total_dones:
        obs = env.reset()
        ret = 0
        epi_steps = 0
        while True:
            A = env.action_space.sample()
            
            # Receive reward and next state            
            next_obs, reward, done, _ = env.step(A)
            
            # Log
            ret += reward
            steps += 1
            epi_steps += 1

            # Termination
            if done or epi_steps == timeout:
                rets.append(ret)
                ep_lens.append(epi_steps)
                print('-' * 50)
                print("Episode: {}: # steps = {}, return = {}. Total Steps: {}".format(dones, epi_steps, ret, steps))
                print('-' * 50)

                if done:
                    dones += 1

                break

            obs = next_obs

    # Random policy stats
    rets = np.array(rets)
    ep_lens = np.array(ep_lens)

    prev_steps = 0
    new_ep_lens = []
    for steps in ep_lens:
        if steps == timeout:
            prev_steps += steps
        else:
            new_ep_lens.append(steps + prev_steps)
            prev_steps = 0
    
    with open(task + '_timeout='+str(timeout)+'_random_stat.txt', 'w') as out_file:
        for ep_len in new_ep_lens:
            out_file.write(str(ep_len)+'\n')

        out_file.write("\nMean: {:.2f}".format(np.mean(new_ep_lens)))
        
def interaction(domain_name, task_name, seed=1):
    # Load one task:
    # env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})
    env = ReacherWrapper(tol=0.009, timeout=5000, seed=0)

    # Step through an episode and print out reward, discount and observation.
    action_spec = env.action_spec()
    EP = 50
    rets = []
    ep_lens = []
    for i in range(EP):
        time_step = env.reset()
        steps = 0
        ret = 0
        while not time_step.last():
            action = np.random.uniform(action_spec.minimum,
                                       action_spec.maximum,
                                       size=action_spec.shape)
            time_step = env.step(action)
            print(steps, time_step.reward, time_step.discount)
            steps += 1
            ret += time_step.reward
        rets.append(ret)
        ep_lens.append(steps)
        print('-' * 100)
        print("Episode: {} ended in {} steps with return: {}".format(i+1, steps, ret))
        print('-' * 100)

    # Random policy stats
    rets = np.array(rets)
    ep_lens = np.array(ep_lens)
    print("Mean: {:.2f}".format(np.mean(ep_lens)))
    print("Standard Error: {:.2f}".format(np.std(ep_lens) / np.sqrt(len(ep_lens) - 1)))
    print("Median: {:.2f}".format(np.median(ep_lens)))
    inds = np.where(ep_lens == env._timeout)
    print("Success Rate (%): {:.2f}".format((1 - len(inds[0]) / len(ep_lens)) * 100.))
    print("Max length:", max(ep_lens))
    print("Min length:", min(ep_lens))

if __name__ == '__main__':
    # for domain_name, task_name in suite.ALL_TASKS: # suite.BENCHMARKING
    #     print(domain_name, texit()ask_name)
    #     env = suite.load(domain_name, task_name)

    # m = ManipulatorWrapper(task_name="insert_ball", seed=3)

    # interaction("manipulator", "bring_ball")
    # interaction("acrobot", "swingup_sparse")

    # visualize_behavior("manipulator", "bring_ball")
    # visualize_behavior("ball_in_cup", "catch")
    # visualize_behavior("reacher", "hard")
    # visualize_behavior("acrobot", "swingup_sparse")

    # r = ReacherWrapper(seed=1)
    # random_policy_stats()
    ranndom_policy_done_2_done_length()
    # env = BallInCupWrapper(1, 1000, use_image=True)
    # obs = env.reset()
    # img = obs.images

    # print(img.shape)
    # img_to_show = np.transpose(img, [1, 2, 0])
    # img_to_show = img_to_show[:,:,-3:]
    # cv2.imshow('', img_to_show)
    # cv2.waitKey(0)

    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     next_obs, _, _, _ = env.step(action)
    #     next_img = next_obs.images
    #     img_to_show = np.transpose(next_img, [1, 2, 0])
    #     img_to_show = img_to_show[:,:,-3:]
    #     cv2.imshow('', img_to_show)
    #     cv2.waitKey(50)
