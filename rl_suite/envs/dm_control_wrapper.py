import torch
import gym

import numpy as np
import matplotlib.pyplot as plt

from dm_control import suite
from gym.spaces import Box


class BallInCupWrapper:
    def __init__(self, seed, timeout, penalty=0.01):
        """ Outputs state transition data as torch arrays """
        self.env = suite.load(domain_name="ball_in_cup", task_name="catch", task_kwargs={'random': seed})
        self._timeout = timeout
        self._obs_dim = 8
        self._action_dim = 2
        
        assert penalty > 0
        self.reward = -penalty

    def make_obs(self, x):
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        obs[:4] = x.observation['position'].astype(np.float32)
        obs[4:8] = x.observation['velocity'].astype(np.float32)
        return obs

    def reset(self):
        self.steps = 0
        return self.make_obs(self.env.reset())

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()
        self.steps += 1

        x = self.env.step(action)
        next_obs = self.make_obs(x)
        reward = self.reward
        done = x.reward # or self.steps == self._timeout
        info = {}

        return next_obs, reward, done, info

    @property
    def observation_space(self):
        return Box(shape=(self._obs_dim,), high=10, low=-10)

    @property
    def action_space(self):
        return Box(shape=(self._action_dim,), high=1, low=-1)


class ReacherWrapper(gym.Wrapper):
    def __init__(self, tol, seed, penalty, timeout=500):
        """

        Args:
            tol (float): [0.009, 0.018, 0.036, 0.072]
            seed:
        """
        super().__init__(gym.make('Reacher-v2').unwrapped)
        self.env.seed(seed)
        self.penalty = penalty
        self._tol = tol        
        self._timeout = timeout
        self._action_dim = 2

    def reset(self):
        self.steps = 0
        obs = self.env.reset()
        return obs.astype(np.float32)

    def step(self, action):
        if isinstance(action, torch.Tensor):            
            action = action.cpu().numpy().flatten()
        self.steps += 1
        next_obs, _, done, info = self.env.step(action)
        next_obs = next_obs.astype(np.float32)

        dist_to_target = -info["reward_dist"]

        reward = -self.penalty
    
        if dist_to_target <= self._tol:
            info['reached'] = True
            done = True

        done = done # or self.steps == self._timeout

        return next_obs, reward, done, info


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

def random_policy_stats():
    # Problem
    seed = 10
    timeout = 1000
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Env
    # env = BallInCupWrapper(seed, timeout=5000)
    # env = ReacherWrapper(seed=seed, tol=0.009, timeout=timeout)
    env = suite.load(domain_name="cartpole", task_name="swingup_sparse", task_kwargs={'random': seed})
    if not hasattr(env, "_action_dim"):
        env._action_dim = env.action_spec().shape[0]

    # Experiment
    EP = 50
    rets = []
    ep_lens = []
    steps = 0
    for ep in range(EP):
        obs = env.reset()
        ret = 0
        epi_steps = 0
        while True:
            # Take action
            A = torch.rand((1, env._action_dim))
            # A = A * (env.action_space.high - env.action_space.low) + env.action_space.low
            # print(A)

            # Receive reward and next state
            next_obs, reward, done, _ = env.step(A)
            # print("Step: {}, Next Obs: {}, reward: {}, done: {}".format(steps, next_obs[:4], reward, done))

            # Log
            ret += reward
            steps += 1
            epi_steps += 1

            # Termination
            if done or epi_steps == timeout:
                rets.append(ret)
                ep_lens.append(epi_steps)
                print('-' * 50)
                print("Episode: {}: # steps = {}, return = {}. Total Steps: {}".format(ep, epi_steps, ret, steps))
                print('-' * 50)
                break

            obs = next_obs

    # Random policy stats
    rets = np.array(rets)
    ep_lens = np.array(ep_lens)
    print("Mean: {:.2f}".format(np.mean(ep_lens)))
    print("Standard Error: {:.2f}".format(np.std(ep_lens) / np.sqrt(len(ep_lens) - 1)))
    print("Median: {:.2f}".format(np.median(ep_lens)))
    inds = np.where(ep_lens == timeout)
    print("Success Rate (%): {:.2f}".format((1 - len(inds[0]) / len(ep_lens)) * 100.))
    print("Max length:", max(ep_lens))
    print("Min length:", min(ep_lens))

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
        # print(domain_name, task_name)
        # env = suite.load(domain_name, task_name)

    # m = ManipulatorWrapper(task_name="insert_ball", seed=3)

    # interaction("manipulator", "bring_ball")
    # interaction("acrobot", "swingup_sparse")

    # visualize_behavior("manipulator", "bring_ball")
    # visualize_behavior("ball_in_cup", "catch")
    # visualize_behavior("reacher", "hard")
    # visualize_behavior("acrobot", "swingup_sparse")

    # r = ReacherWrapper(seed=1)
    random_policy_stats()
