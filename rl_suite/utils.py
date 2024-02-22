import json
import numpy as np
import matplotlib.pyplot as plt

from rl_suite.envs.min_time_dm_control import BallInCupWrapper, ReacherWrapper
from rl_suite.envs.point_maze import PointMaze
from rl_suite.envs.dot_seeker import DotSeeker
from rl_suite.envs.dot_reacher_env import DotReacherEnv, VisualDotReacherEnv
from rl_suite.envs.gymnasium_wrapper import GymnasiumWrapper
from rl_suite.envs.dm_control_wrapper import ENV_MAP, DMControl
from rl_suite.plot import smoothed_curve


def make_env(args):
    env_str, reward, seed, timeout = args.env, args.reward, args.seed, args.timeout
    use_image = getattr(args, 'use_image', False)

    if env_str == "ball_in_cup":        
        env = BallInCupWrapper(seed=seed, timeout=timeout, reward=reward, use_image=use_image)
    elif env_str == "dm_reacher_easy":        
        env = ReacherWrapper(seed=seed, timeout=timeout, reward=reward, mode="easy", use_image=use_image)
    elif env_str == "dm_reacher_hard":        
        env = ReacherWrapper(seed=seed, timeout=timeout, reward=reward, mode="hard", use_image=use_image)
    elif "dot_reacher" in env_str:
        if env_str == "dot_reacher_easy":
            pos_tol = 0.25; vel_tol = 0.1
        elif env_str == "dot_reacher_hard":
            pos_tol = 0.1; vel_tol = 0.05
            
        if use_image:
            env = VisualDotReacherEnv(pos_tol=pos_tol, vel_tol=vel_tol, penalty=reward, timeout=timeout)
        else:
            env = DotReacherEnv(pos_tol=pos_tol, vel_tol=vel_tol, penalty=reward, timeout=timeout)                
    elif env_str == "dot_seeker":
        env = DotSeeker(pos_tol=pos_tol, penalty=reward,
                            timeout=timeout, use_image=use_image)
    elif env_str == "point_maze":        
        env = PointMaze(seed=seed, map_type=args.maze_type, reward_type=args.reward_type, use_image=use_image, timeout=timeout, reward=reward)
        env_str += f"_{args.maze_type}_{args.reward_type}"  # Make env name clear for saving results
    elif env_str in ENV_MAP:
        env = DMControl(env_name=env_str, seed=seed)
    else:
        from rl_suite.envs.gymnasium_wrapper import GymnasiumWrapper
        env = GymnasiumWrapper(env=env_str, seed=seed, time_limit=timeout)
    env.name = env_str
    return env


def learning_curve(rets, ep_lens, save_path, x_tick=10000, window_len=10000):
    if len(rets) > 0:
        plot_rets, plot_x = smoothed_curve(np.array(rets), np.array(ep_lens), x_tick=x_tick, window_len=window_len)
        if len(plot_rets):
            plt.clf()
            plt.plot(plot_x, plot_rets)
            plt.pause(0.001)
            plt.savefig(save_path, dpi=200)


def save_returns(rets, ep_lens, save_path):
    """ Save learning curve data as a numpy text file 

    Args:
        rets (list/array): A list or array of episodic returns
        ep_lens (list/array):  A list or array of episodic length
        savepath (str): Save path
    """
    data = np.zeros((2, len(rets)))
    data[0] = ep_lens
    data[1] = rets
    np.savetxt(save_path, data)


class NpEncoder(json.JSONEncoder):
    """ 
    JSON does not like Numpy elements. Convert to native python datatypes for json dump.  
    Ref: https://bobbyhadz.com/blog/python-typeerror-object-of-type-int64-is-not-json-serializable
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_args(args, save_path):
    """ Save hyper-parameters as a json file """
    hyperparas_dict = vars(args)
    hyperparas_dict["device"] = str(hyperparas_dict["device"])
    json.dump(hyperparas_dict, open(save_path, 'w'), indent=4, cls=NpEncoder)
