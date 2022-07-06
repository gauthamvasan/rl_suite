import gym


class MountainCarContinuous:
    """ Minimum-time version with longer timelimits"""
    def __init__(self) -> None:
        self.env = gym.make("MountainCarContinuous-v0")

    def reset(self):
        self.steps = 0
        return self.env.reset()
    
    def step(self, action):
        next_obs, _, _, info = self.env.step(action)
        self.steps += 1
        reward = -0.1
        pos, vel = next_obs
        done = bool(pos >= self.env.goal_position and vel >= self.env.goal_velocity)
        return next_obs, reward, done, info
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space


if __name__ == "__main__":
    env = MountainCarContinuous()
    n_episodes = 10
    timeout = 10000

    for i_ep in range(n_episodes):
        done = False
        ret = 0
        step = 0
        obs = env.reset()
        while (not done and step < timeout):
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            ret += reward
            step += 1
            obs = next_obs
        print("Episode {} ended in {} steps with return {}".format(i_ep+1, step, ret))
