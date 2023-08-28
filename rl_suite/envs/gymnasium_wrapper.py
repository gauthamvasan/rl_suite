import gymnasium as gym
import numpy as np


class GymnasiumWrapper:
    def __init__(self, env, seed, time_limit) -> None:
        self.env = gym.make(env)
        self.time_limit = time_limit
        observation, info = self.env.reset()

        # Set seed
        self.env.reset(seed=seed) # This is just to set the seed
        self.env.action_space.seed(seed)

    def reset(self):
        self.steps = 0
        observation, info = self.env.reset()
        return observation

    def step(self, action):
        self.steps += 1
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        terminated = terminated or self.steps == self.time_limit
        return next_observation, reward, terminated, info
    
    def render(self):
        return self.env.render()

    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space

    def close(self):
        self.env.close()


if __name__ == "__main__":
    env = "Ant-v2"
    seed = 42
    n_episodes = 100
    timeout = 1000
    np.random.seed(seed)
    env = GymnasiumWrapper(env, seed)

    obs = env.reset()
    for i_ep in range(n_episodes):
        done = False
        ret = 0
        step = 0
        while (not done and step < timeout):
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            ret += reward
            step += 1
            obs = next_obs

            # if step % 100 == 0:
            #     print(f"Obs: {obs[:4]}, action: {action}, reward: {reward}")
            env.render()

        obs = env.reset()
        print("Episode {} ended in {} steps with return {:.2f}. Done: {}.".format(i_ep+1, step, ret, done))
    
    env.close()
    