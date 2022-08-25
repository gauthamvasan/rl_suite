import torch
import gym

import numpy as np
import matplotlib.pyplot as plt

from dm_control import suite
from gym.spaces import Box


class BallInCupWrapper:
    def __init__(self, seed, timeout, penalty=0.1):
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
    def __init__(self, seed, timeout, penalty=1, mode="easy"):
        """ Outputs state transition data as torch arrays """
        assert mode in ["easy", "hard"]
        self.env = suite.load(domain_name="reacher", task_name=mode, task_kwargs={'random': seed})
        self._timeout = timeout
        self._obs_dim = 6
        self._action_dim = 2
        
        assert penalty > 0
        self.reward = -penalty

    def make_obs(self, x):
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        obs[:2] = x.observation['position'].astype(np.float32)
        obs[2:4] = x.observation['velocity'].astype(np.float32)
        obs[4:6] = x.observation['to_target'].astype(np.float32)
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
    seed = 1
    timeout = 1000
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Env
    # env = BallInCupWrapper(seed, timeout=timeout, penalty=1)
    env = ReacherWrapper(seed=seed, mode="hard", timeout=timeout)
    # env = suite.load(domain_name="quadruped", task_name="fetch", task_kwargs={'random': seed})
    # env = suite.load(domain_name="reacher", task_name="easy", task_kwargs={'random': seed})
    if not hasattr(env, "_action_dim"):
        env._action_dim = env.action_spec().shape[0]

    # Experiment
    EP = 40
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
            
            # print("Step: {}, Next Obs: {}, reward: {}, done: {}".format(steps, next_obs, reward, done))

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

    prev_steps = 0
    new_ep_lens = []
    for steps in ep_lens:
        if steps == timeout:
            prev_steps += steps
        else:
            new_ep_lens.append(steps + prev_steps)
            prev_steps = 0
    
    print(new_ep_lens)
    print(np.mean(new_ep_lens), np.median(new_ep_lens), np.std(new_ep_lens) / np.sqrt(len(new_ep_lens) - 1))

    print("Mean: {:.2f}".format(np.mean(ep_lens)))
    print("Standard Error: {:.2f}".format(np.std(ep_lens) / np.sqrt(len(ep_lens) - 1)))
    print("Median: {:.2f}".format(np.median(ep_lens)))
    inds = np.where(ep_lens == timeout)
    print("Success Rate (%): {:.2f}".format((1 - len(inds[0]) / len(ep_lens)) * 100.))
    print("Max length:", max(ep_lens))
    print("Min length:", min(ep_lens))
    print(np.mean(ep_lens[np.where(ep_lens != env._timeout)]))

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
    random_policy_stats()
