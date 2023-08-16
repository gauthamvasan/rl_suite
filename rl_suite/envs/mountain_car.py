import gymnasium as gym
import numpy as np


class MountainCarContinuous:
    def __init__(self, seed, reward=-1) -> None:
        assert reward < 0, "Per-timestep reward must be negative"
        self.env = gym.make('MountainCarContinuous-v0')
        self.env.reset(seed=seed)
        self.reward = reward
    
    def reset(self):
        return self.env.reset()[0]
    
    def step(self, action):
        next_obs, r, done, _, info = self.env.step(action)
        reward = self.reward
        return next_obs, reward, done, info

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


if __name__ == "__main__":   
    seed = 42
    n_episodes = 100
    timeout = 10000
    use_image = True
    render_mode = None  # "human", "rgb_array"
    np.random.seed(seed)
    env = MountainCarContinuous(seed=seed, reward=-1)

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
            env.render()
            
        print("Episode {} ended in {} steps with return {:.2f}. Done: {}".format(i_ep+1, step, ret, done))
    
    env.close()