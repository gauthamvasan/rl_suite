import gymnasium as gym


OPEN_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 'C', 'C', 'C', 'C', 'C', 1],
    [1, 'C', 'C', 'C', 'C', 'C', 1],
    [1, 'C', 'C', 'C', 'C', 'C', 1],
    [1, 1, 1, 1, 1, 1, 1]
]

MEDIUM_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'C', 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 'C', 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 'C', 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 'C', 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]

LARGE_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'C', 0, 0, 0, 1, 'C', 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 'C', 0, 1, 0, 0, 'C', 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 'C', 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 'C', 0, 'C', 1, 0, 'C', 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

def main():


    # env = gym.make('PointMaze_UMaze-v3', maze_map=LARGE_MAZE_DIVERSE_G, render_mode = "human")
    env = gym.make('PointMaze_UMaze-v3', maze_map=LARGE_MAZE_DIVERSE_GR)

    n_episodes = 10
    timeout = 10000

    for i_ep in range(n_episodes):
        done = False
        ret = 0
        step = 0
        obs = env.reset()
        while (not done and step < timeout):
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            ret += reward
            step += 1
            obs = next_obs
            # env.render()
        print("Episode {} ended in {} steps with return {}".format(i_ep+1, step, ret))

if __name__ == "__main__":
    main()
    