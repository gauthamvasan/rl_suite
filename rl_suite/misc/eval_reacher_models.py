import torch
import pickle
import glob
import argparse
import numpy as np

from rl_suite.algo.mlp_policies import SquashedGaussianMLPActor
from rl_suite.misc.dm_reacher_comparisons import FixedTimeLimitReacher, VelTolReacher, AdditiveRewardReacher

N = 501000
EP = 500
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


def eval_reacher_across_tasks():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True, type=int, help="Seed for random number generator")
    parser.add_argument('--env', required=True, type=str, help="Model env")
    parser.add_argument('--eval_env', required=True, type=str, help="Evaluation env")
    args = parser.parse_args()

    seed = args.seed
    basepath = f"/home/vasan/scratch/tro_paper/rupam_eval/{args.env}"

    if args.eval_env == "vt_reacher_easy":
        env = VelTolReacher(seed=seed, mode="easy", use_image=False)
    elif args.eval_env == "vt_reacher_hard":
        env = VelTolReacher(seed=seed, mode="hard", use_image=False)
    elif args.eval_env == "ar_reacher_easy":
        env = AdditiveRewardReacher(seed=seed, mode="easy", use_image=False)
    elif args.eval_env == "ar_reacher_hard":
        env = AdditiveRewardReacher(seed=seed, mode="hard", use_image=False)
    elif args.eval_env == "ftl_reacher_easy":
        env = FixedTimeLimitReacher(seed=seed, mode="easy", use_image=False)
    elif args.eval_env == "ftl_reacher_hard":
        env = FixedTimeLimitReacher(seed=seed, mode="hard", use_image=False)

    ret = 0
    step = 0
    rets = []
    ep_lens = []
    obs = env.reset()
    done = False
    ep = 0
    for t in range(N):
        if t % 10000 == 0 and t > 0:
            model_path = glob.glob(f"{basepath}/*-{args.seed}_model_{t//1000}K.pt")
            if not model_path:
                model_path = glob.glob(f"{basepath}/*-{args.seed}_model_{t/1000}K.pt")
            
            assert len(model_path) == 1, print(f"{len(model_path)} files found.")

            model_dict = torch.load(model_path[0])
            actor.load_state_dict(model_dict['actor'])
            print(f"Model load from {model_path} successful")

        # Take action
        x = torch.tensor(obs.astype(np.float32)).to(device).unsqueeze(0)
        with torch.no_grad():
            mu, action, _, log_std = actor(x)
        action = action.cpu().data.numpy()

        # Receive reward and next state
        next_obs, R, done, _ = env.step(action)

        ret += R
        step += 1

        obs = next_obs

        # Termination
        if done:
            rets.append(ret)
            ep_lens.append(step)
            ep += 1
            print(f"Episode {ep} ended in {step} steps with return {ret}")
            obs = env.reset()
            done = False
            step = 0
            ret = 0
        

    data = np.zeros((2, len(rets)))
    data[0] = np.array(ep_lens)
    data[1] = np.array(rets)
    np.savetxt(f"{basepath}/{args.env}_model_on_{args.eval_env}_eval_seed-{seed}.txt", data)


def eval_reacher_models_on_5K_episodes():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True, type=int, help="Seed for random number generator")       
    parser.add_argument('--env', required=True, type=str)
    parser.add_argument('--mode', required=True, type=str)
    args = parser.parse_args()
    
    # basepath = "/home/vasan/src/rl_suite/rl_suite/misc/results"
    basepath = "/home/vasan/scratch/tro_paper/rupam_eval"
    model_path = glob.glob(f"{basepath}/{args.env}/*-{args.seed}_model.pt")
    try:
        ret, steps_to_goal = interaction(model_path[0], args.mode)
    except IndexError as e:
        print(model_path)
        print(e)

if __name__ == "__main__":
    eval_reacher_models_on_5K_episodes()
    # eval_reacher_across_tasks()
