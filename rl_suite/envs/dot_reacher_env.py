import cv2
import torch
import time

import numpy as np

from gym.core import Env
from gym.spaces.box import Box
from beautifultable import BeautifulTable
from rl_suite.envs import Observation


class DotReacherEnv(Env):
    def __init__(self, pos_tol=0.25, vel_tol=0.1, dt=0.2, timeout=20000, clamp_action=False, penalty=-0.1):
        """ Continuous Action Space; Acceleration Control

        Args:
            pos_tol (float): Position tolerance - how close should the agent be to the the target.
            vel_tol (float): Velocity tolerance - how fast should the agent be moving when it's close to the target
            dt (float): Action cycle time
            timeout (int): Maximum episode length
            clamp_action (bool): Should actions be clamped or unbounded
        """

        self._pos_tol = pos_tol
        self._vel_tol = vel_tol
        self._timeout = timeout
        self.reward = penalty
        self._clamp_action = clamp_action
        self.dt = dt

        super(DotReacherEnv, self).__init__()

        self._action_low = torch.tensor([[-1., -1.]])
        self._action_high = torch.tensor([[1., 1.]])
        self._pos_low = torch.tensor([[-1., -1.]])
        self._pos_high = torch.tensor([[1., 1.]])
        self.steps = 0

    @property
    def action_space(self):
        # TODO: Enforce this constraint
        return Box(low=-1, high=1, shape=(2,))

    @property
    def observation_space(self):
        # TODO: Verify that min/max velocity are always within these bounds
        return Box(low=-10, high=10, shape=(4,))

    def reset(self):
        self.steps = 0
        self.pos = torch.rand((1, 2)) * (self._pos_high - self._pos_low) + self._pos_low
        self.vel = torch.zeros((1, 2))
        obs = torch.cat((self.pos, self.vel), 1)
        return obs

    def step(self, action):
        """

        Args:
            action: 2-D Tensor (vals between [-1, 1])

        Returns:

        """
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action.astype(np.float32)).view((1, -1))

        self.steps += 1

        # Clamp the action
        if self._clamp_action:
            action = torch.clamp(action, min=self._action_low, max=self._action_high)

        # Acceleration control for smoothness
        self.pos = self.pos + self.vel * self.dt + 0.5 * action * self.dt ** 2
        self.vel[self.pos < self._pos_low] = -0.1 * self.vel[self.pos < self._pos_low]
        self.vel[self.pos > self._pos_high] = -0.1 * self.vel[self.pos > self._pos_high]
        self.pos = torch.clamp(self.pos, self._pos_low, self._pos_high)
        self.vel += action * self.dt

        # Observation
        next_obs = torch.cat((self.pos, self.vel), 1)

        # Reward
        reward = self.reward

        # Done
        done = torch.allclose(self.pos, torch.zeros(2), atol=self._pos_tol) and \
               torch.allclose(self.vel, torch.zeros(2), atol=self._vel_tol)
        # done = done or self.steps == self._timeout

        # Metadata
        info = {}

        return next_obs, reward, done, info


class VisualDotReacherEnv(DotReacherEnv):
    def __init__(self, pos_tol=0.1, vel_tol=0.05, dt=1, timeout=20000, clamp_action=False, 
            img_dim=(120, 160, 3), penalty=-0.1):
        super(VisualDotReacherEnv, self).__init__(pos_tol, vel_tol, dt, timeout, clamp_action, penalty)
        self.img_dim = np.array(img_dim)
        self.target_radius = np.round(self._pos_tol * img_dim[0] / 2.).astype(np.int)
        self.dot_radius = np.round(0.02 * img_dim[0]).astype(np.int)

        self.proprioception_dim = 2 # Velocity
        self.pixel_target = (self.img_dim[1] // 2, self.img_dim[0] // 2)
        self.pt1 = (self.pixel_target[0] - self.target_radius, self.pixel_target[1] - self.target_radius)
        self.pt2 = (self.pixel_target[0] + self.target_radius, self.pixel_target[1] + self.target_radius)

    def pos_to_pixel(self, pos):
        """ Convert position in [-1, 1] to pixel space specified by img_dim

        Args:
            pos: 2D vector

        Returns:

        """
        pixel_loc = (pos + 1) * np.array([self.img_dim[1], self.img_dim[0]]) * 0.5
        pixel_loc = [round(x.item()) for x in pixel_loc.flatten()]
        return tuple(pixel_loc)

    def get_image(self, pos):
        img = np.ones(self.img_dim, dtype=np.uint8) * 255
        # Red target, Blue agent
        image = cv2.rectangle(img=img, pt1=self.pt1, pt2=self.pt2, color=(25, 25, 200), thickness=-1)
        pixel_pos = self.pos_to_pixel(pos)
        image = cv2.circle(img=image, center=pixel_pos, radius=self.dot_radius, color=(255, 83, 73), thickness=-1)
        image = np.transpose(image, [2, 0, 1]).astype(np.float32)  # c, h, w

        return torch.as_tensor(image)

    def get_obs(self):
        obs = Observation()
        obs.images = self.get_image(self.pos)
        obs.proprioception = self.vel.view(-1)
        return obs

    def reset(self):
        super(VisualDotReacherEnv, self).reset()
        return self.get_obs()

    def step(self, action):
        _, reward, done, info = super(VisualDotReacherEnv, self).step(action)
        obs = self.get_obs()
        return obs, reward, done, info

    @property
    def image_space(self):
        image_shape = (self.img_dim[2], self.img_dim[0], self.img_dim[1])
        return Box(low=0, high=255, shape=image_shape)

    @property
    def proprioception_space(self):
        return Box(low=-10, high=10, shape=(2,))

def random_pi_dot_reacher():
    # Problem
    torch.manual_seed(3)

    # Env
    env = DotReacherEnv(pos_tol=0.1, vel_tol=0.05, dt=1, clamp_action=False, timeout=20000)

    # Experiment
    EP = 50
    rets = []
    ep_lens = []
    # Slogs = []
    steps = 0
    for ep in range(EP):
        # Slogs.append([])
        obs = env.reset()
        # Slogs[-1].append(obs)
        ret = 0
        epi_steps = 0
        while True:
            # Take action
            A = torch.rand((1, 2))
            A = A * (env._action_high - env._action_low) + env._action_low
            # print(A)

            # Receive reward and next state
            next_obs, R, done, _ = env.step(A)
            # print("Step: {}, Obs: {}, Next Obs: {}".format(steps, obs, next_obs))

            # Log
            # Slogs[-1].append(next_obs)
            ret += R
            steps += 1
            epi_steps += 1

            # Termination
            if done:
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
    print("Mean:", np.mean(ep_lens))
    print("Standard Error:", np.std(ep_lens)/np.sqrt(len(ep_lens)-1))
    print("Median:", np.median(ep_lens))
    inds = np.where(ep_lens == env._timeout)
    print("Success Rate (%):", (1 - len(inds[0])/len(ep_lens))*100.)
    print("Max length:", max(ep_lens))
    print("Min length:", min(ep_lens))

    # Plotting
    # plt.plot(-100 * torch.tensor(rets))
    # plt.figure()
    #
    # colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:red", "tab:brown"]
    # for i in range(-min(30, EP), 0):
    #     color = colors[i % len(colors)]
    #     Slog = torch.cat(Slogs[i])
    #     for i in range(Slog.shape[0] - 1):
    #         plt.plot(Slog[i:i + 2, 0], Slog[i:i + 2, 1], alpha=(i + 1) / Slog.shape[0], color=color, marker='.')
    # plt.xlim([env._pos_low[0, 0], env._pos_high[0, 0]])
    # plt.ylim([env._pos_low[0, 1], env._pos_high[0, 1]])
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid()
    # plt.show()


def viz_dot_reacher():
    import matplotlib.pyplot as plt

    # Problem
    torch.manual_seed(3)

    # Env
    env = VisualDotReacherEnv(pos_tol=0.1, vel_tol=0.05, timeout=20000, dt=0.2, clamp_action=True)

    # Experiment
    EP = 5
    rets = []
    steps = 0
    # create two subplots
    ax1 = plt.subplot(1, 1, 1)

    # create two image plots
    plt.ion()
    for ep in range(EP):
        obs = env.reset()
        im1 = ax1.imshow(np.ones(env.img_dim, dtype=np.uint8)*255)
        ret = 0
        epi_steps = 0
        while True:
            img = obs.images
            proprioception = obs.proprioception
            im1.set_data(np.transpose(img.numpy(), [1, 2, 0]).astype(np.uint8))            
            # Take action
            A = torch.rand((1, 2))
            A = A * (env._action_high - env._action_low) + env._action_low
            # print(A)

            # Receive reward and next state
            plt.pause(0.02)
            next_obs, R, done, _ = env.step(A)            
            print("Step: {}, Obs: {}, Next Obs: {}".format(steps, obs.proprioception, next_obs.proprioception))

            # Log
            # Slogs[-1].append(next_obs)
            ret += R
            steps += 1
            epi_steps += 1

            # Termination
            if done:
                rets.append(ret)
                print('-' * 50)
                print("Episode: {}: # steps = {}, return = {}. Total Steps: {}".format(ep, epi_steps, ret, steps))
                print('-' * 50)
                time.sleep(2)
                break

            obs = next_obs

    # Plotting
    plt.plot(-100 * torch.tensor(rets))
    plt.show()

def p_value_table():
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    pos_tols = [0.25, 0.15, 0.1, 0.1, 0.05]
    vel_tols = [0.25, 0.1, 0.075, 0.05, 0.05]
    timeouts = [500, 1000, 5000, 20000]
    n_timesteps = 50000

    table = BeautifulTable(maxwidth=140, precision=6)
    table.set_style(BeautifulTable.STYLE_MARKDOWN)
    table.columns.header = ['pos_tol', 'vel_tol', 'timeout', 'Mean', 'Std. Error', 'Median', 'p-val', 'Success Rate', 'Max', 'Min']

    for timeout in timeouts:
        for pos_tol, vel_tol in zip(pos_tols, vel_tols):
            env = DotReacherEnv(pos_tol, vel_tol, dt=0.2, clamp_action=True, timeout=timeout)
            rets = []
            ep_lens = []
            ep = 0            
            step = 0
            ret = 0
            epi_steps = 0         
            obs = env.reset()
            while step <= n_timesteps:                
                # Take action
                A = torch.rand((1, 2))
                A = A * (env._action_high - env._action_low) + env._action_low
                # print(A)

                # Receive reward and next state
                next_obs, R, done, _ = env.step(A)
                # print("Step: {}, Obs: {}, Next Obs: {}".format(steps, obs, next_obs))

                ret += R                    
                epi_steps += 1
                obs = next_obs
                step += 1

                # Termination
                if done or epi_steps == timeout:
                    ep += 1
                    rets.append(ret)
                    ep_lens.append(epi_steps)
                    print('-' * 50)
                    print("Episode: {}: # steps = {}, return = {}. Total Steps: {}".format(ep, epi_steps, ret, step))
                    print('-' * 50)
                    obs = env.reset()
                    ret = 0
                    epi_steps = 0

            # Random policy stats
            rets = np.array(rets)
            ep_lens = np.array(ep_lens)
            inds = np.where(ep_lens == timeout)
            row = [pos_tol, vel_tol, timeout, np.mean(ep_lens), np.std(ep_lens)/np.sqrt(len(ep_lens)-1), np.median(ep_lens), 
                    n_timesteps/float(ep-len(inds)), (1 - len(inds[0])/len(ep_lens))*100., max(ep_lens), min(ep_lens)]
            table.rows.append(row)
            print(row)
    
    print(table)


if __name__ == '__main__':
    # random_pi_dot_reacher()
    viz_dot_reacher()
    # p_value_table()
    