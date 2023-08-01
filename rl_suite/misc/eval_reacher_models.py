import torch
import pickle
import glob
import argparse
import numpy as np

from rl_suite.algo.mlp_policies import SquashedGaussianMLPActor
from rl_suite.misc.dm_reacher_comparisons import FixedTimeLimitReacher

EP = 50
TIMEOUT = 5000
obs_dim = 6
action_dim = 2
device = torch.device('cuda')
actor_nn_params = {
    'mlp': {
        'hidden_sizes': [512, 512],
        'activation': 'relu',
    }
}
actor = SquashedGaussianMLPActor(obs_dim, action_dim, actor_nn_params, device)
    
def interaction(model_path, mode):
    print('-' * 50)
    print(model_path)
    print('-' * 50)

    model_dict = torch.load(model_path)
    actor.load_state_dict(model_dict['actor'])

    env = FixedTimeLimitReacher(seed=42, mode=mode, use_image=False)
    env.timeout = TIMEOUT
    rets = []
    ep_lens = []
    steps_to_goal = []
    for ep in range(EP):
        obs = env.reset()
        ret = 0
        reached_goal = False
        step = 0
        while True:
            # Take action
            x = torch.tensor(obs.astype(np.float32)).to(device).unsqueeze(0)
            with torch.no_grad():
                mu, action, _, log_std = actor(x)
            action = action.cpu().data.numpy()

            # Receive reward and next state
            next_obs, R, done, _ = env.step(action)
            if (R == 1) and (not reached_goal):
                reached_goal = True
                steps_to_goal.append(step)

            ret += R
            step += 1

            # Termination
            if done:
                rets.append(ret)
                ep_lens.append(step)
                print(f"Episode {ep+1} ended in {step} steps with return {ret}")
                break
            obs = next_obs

    with open(model_path[:-1]+"kl", "wb") as handle:
        pickle.dump({'ret': np.mean(rets), 'steps_to_goal': np.mean(steps_to_goal)}, handle)

    return np.mean(rets), np.mean(steps_to_goal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True, type=int, help="Seed for random number generator")       
    parser.add_argument('--env', required=True, type=str)
    parser.add_argument('--mode', required=True, type=str)
    args = parser.parse_args()
    
    basepath = "/home/vasan/src/rl_suite/rl_suite/misc/results"
    model_path = glob.glob(f"{basepath}/{args.env}/*-{args.seed}_model.pt")
    ret, steps_to_goal = interaction(model_path[0], args.mode)
