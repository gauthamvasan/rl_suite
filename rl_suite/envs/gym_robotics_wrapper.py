"""
D4RL references:
- https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/maze/point_maze.py
- https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/maze/maze_v4.py
"""
import time
import gymnasium as gym
import numpy as np


OPEN_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 'c', 'c', 'c', 'c', 'c', 1],
    [1, 'c', 'c', 'c', 'c', 'c', 1],
    [1, 'c', 'c', 'c', 'c', 'c', 1],
    [1, 1, 1, 1, 1, 1, 1]
]

MEDIUM_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'c', 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 'c', 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 'c', 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 'c', 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]

LARGE_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'c', 0, 0, 0, 1, 'c', 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 'c', 0, 1, 0, 0, 'c', 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 'c', 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 'c', 0, 'c', 1, 0, 'c', 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]


class PointMaze():
    all_maps = {"small": OPEN_DIVERSE_GR, "medium": MEDIUM_MAZE_DIVERSE_GR, "large": LARGE_MAZE_DIVERSE_GR}
    """
    PointMaze_UMazeDense=v3 uses np.exp(-np.linalg.norm(a-b)) as reward 
    """
    def __init__(self, seed, map_type="small", reward_type="sparse", use_image=False) -> None:
        assert reward_type in ["sparse", "dense"]
        assert map_type in ["small", "medium", "large"]
        
        self.reward_type = reward_type
        self.use_image = use_image
        
        self.env = gym.make("PointMaze_UMaze-v3", maze_map=self.all_maps[map_type])
        self.set_seeds(seed)

        self._obs_dim = 4 if use_image else 6
        self._action_dim = 2
    
    def set_seeds(self, seed):
        self.seed = seed        
        self.env.reset(seed=seed) # This is just to set the seed
        self.env.action_space.seed(seed)
        
    def make_obs(self, x):
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        if not self.use_image:
            obs[:4] = x['observation'].astype(np.float32)
            obs[4:] = x['desired_goal'].astype(np.float32)
        else:
            obs[:4] = x['observation'].astype(np.float32)
            obs[4:] = x['desired_goal'].astype(np.float32)
        return obs
    
    def reset(self):
        x = self.env.reset()
        return self.make_obs(x[0])

    def step(self, action):
        next_x, r, _, _, info = self.env.step(action)
        next_obs = self.make_obs(next_x)
        done = r

        if self.reward_type == "sparse":
            reward = -1
        else:
            reward = -np.linalg.norm(next_x['achieved_goal'] - next_x['desired_goal'])

        return next_obs, reward, done, info

    @property
    def action_space(self):
        return self.env.action_space
    
    def close(self):
        self.env.close()

def main():    
    seed = 42
    np.random.seed(seed)

    env = PointMaze(seed=42, reward_type="sparse", map_type="small")
    # env = gym.make('PointMaze_UMaze-v3', maze_map=LARGE_MAZE_DIVERSE_G, render_mode = "human")
    

    n_episodes = 10
    timeout = 1000

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
            # print(f"Obs: {obs}, action: {action}, reward: {reward}")
            # env.render()

        print("Episode {} ended in {} steps with return {:.2f}. Done: {}".format(i_ep+1, step, ret, done))
    
    env.close()

if __name__ == "__main__":
    main()
    