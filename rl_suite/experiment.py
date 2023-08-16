"""
General experiment template script
- N.B: Use `export MUJOCO_GL=osmesa` for compute canada (https://github.com/deepmind/dm_control/issues/48)
"""

import argparse
import gym
import torch
import random
import warnings
import os
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
# from rl_suite.mysql_db import MySQLDBManager
from rl_suite.envs.dot_reacher_env import DotReacherEnv, VisualDotReacherEnv
from rl_suite.envs.mountain_car import MountainCarContinuous
from rl_suite.envs.dot_seeker import DotSeeker, DotBoxReacher
from rl_suite.plot.plot import smoothed_curve
from sys import platform

# For MacOS
if platform == "darwin":    
    import matplotlib as mpl
    mpl.use("Qt5Agg")


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


class Experiment:
    def __init__(self, args):
        self.args = args
        self.run_id = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        self.run_id += f"_seed-{args.seed}"

        # Make env
        self.env = self.make_env()
        
        self._expt_dir = os.path.join(args.results_dir, args.env)
        self.make_dir(self._expt_dir)
        self.save_args()

    def save_args(self):
        """ Save hyper-parameters as a json file """
        hyperparas_dict = vars(self.args)
        hyperparas_dict["device"] = str(hyperparas_dict["device"])
        json.dump(hyperparas_dict, open(os.path.join(self._expt_dir, "{}_args.json".format(self.run_id)), 'w'), indent=4, cls=NpEncoder)

    def make_env(self):
        if self.args.env == "ball_in_cup":
            from rl_suite.envs.dm_control_wrapper import BallInCupWrapper
            env = BallInCupWrapper(seed=self.args.seed, penalty=self.args.reward, use_image=self.args.use_image)
        elif self.args.env == "dm_reacher_easy":
            from rl_suite.envs.dm_control_wrapper import ReacherWrapper
            env = ReacherWrapper(seed=self.args.seed, penalty=self.args.reward, mode="easy", use_image=self.args.use_image)
        elif self.args.env == "dm_reacher_hard":
            from rl_suite.envs.dm_control_wrapper import ReacherWrapper
            env = ReacherWrapper(seed=self.args.seed, penalty=self.args.reward, mode="hard", use_image=self.args.use_image)
        elif self.args.env == "dm_reacher_torture":
            from rl_suite.envs.dm_control_wrapper import ReacherWrapper
            env = ReacherWrapper(seed=self.args.seed, penalty=self.args.reward, mode="torture", use_image=self.args.use_image)
        elif "dot_reacher" in self.args.env:
            if self.args.env == "dot_reacher_easy":
                self.args.pos_tol = 0.25
                self.args.vel_tol = 0.1
            elif self.args.env == "dot_reacher_hard":
                self.args.pos_tol = 0.1
                self.args.vel_tol = 0.05
                
            if self.args.use_image:
                env = VisualDotReacherEnv(pos_tol=self.args.pos_tol, vel_tol=self.args.vel_tol, penalty=self.args.reward,
                    dt=self.args.dt, timeout=self.args.timeout, clamp_action=self.args.clamp_action)
            else:
                env = DotReacherEnv(pos_tol=self.args.pos_tol, vel_tol=self.args.vel_tol, penalty=self.args.reward,
                    dt=self.args.dt, timeout=self.args.timeout, clamp_action=self.args.clamp_action)                
        elif self.args.env == "mountain_car_continuous":
            env = MountainCarContinuous(seed=self.args.seed, penalty=self.args.reward)
        elif self.args.env == "mj_reacher":
            from rl_suite.envs.visual_reacher import MJReacherWrapper            
            env = MJReacherWrapper(tol=self.args.tol, penalty=self.args.reward, use_image=self.args.use_image)            
        elif self.args.env == "pick_cube":
            from rl_suite.envs.mani_skill_envs import PickCube
            env = PickCube(seed=self.args.seed, use_image=self.args.use_image)
        elif self.args.env == "dot_seeker":
            env = DotSeeker(pos_tol=self.args.pos_tol, penalty=self.args.reward, dt=self.args.dt, 
                             timeout=self.args.timeout, use_image=self.args.use_image)
        elif self.args.env == "dot_box_reacher":
            env = DotBoxReacher(pos_tol=self.args.pos_tol, vel_tol=self.args.vel_tol, penalty=self.args.reward, 
                                dt=self.args.dt, timeout=self.args.timeout, use_image=self.args.use_image)
        elif self.args.env == "point_maze":
            from rl_suite.envs.point_maze import PointMaze
            env = PointMaze(seed=self.args.seed, map_type=self.args.maze_type, reward_type=self.args.reward_type, 
                            use_image=self.args.use_image, penalty=self.args.reward,)
            self.args.env += f"_{self.args.maze_type}_{self.args.reward_type}"  # Make env name clear for saving results
        else:
            env = gym.make(self.args.env)
            env.seed(self.args.seed)
        env.name = self.args.env
        return env

    def save_returns(self, rets, ep_lens):
        """ Save learning curve data as a numpy text file 

        Args:
            rets (list/array): A list or array of episodic returns
            ep_lens (list/array):  A list or array of episodic length
            savepath (str): Save path
        """
        data = np.zeros((2, len(rets)))
        data[0] = ep_lens
        data[1] = rets
        np.savetxt(f"{self._expt_dir}/{self.run_id}_returns.txt", data)
    
    def save_model(self, unique_str):
        self.learner.save(self._expt_dir, unique_str)

    def set_seed(self):
        seed = self.args.seed

        np.random.seed(seed)
        random.seed(seed)
        try:
            self.env.seed(seed)
        except AttributeError as e:
            warnings.warn("AttributeError: '{}' object has no attribute 'seed'".format(self.args.env))

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # torch.use_deterministic_algorithms(True)

    def learning_curve(self, rets, ep_lens, save_fig=True):
        if len(rets) > 0:
            plot_rets, plot_x = smoothed_curve(
                    np.array(rets), np.array(ep_lens), x_tick=self.args.checkpoint, window_len=self.args.checkpoint)
            if len(plot_rets):
                plt.clf()
                if self.args.xlimit is not None:
                    plt.xlim(self.args.xlimit)
            
                if self.args.ylimit is not None:
                    plt.ylim(self.args.ylimit)
                    
                plt.plot(plot_x, plot_rets)
                plt.pause(0.001)
                if save_fig:
                    plt.savefig(f"{self._expt_dir}/{self.run_id}_learning_curve.png")
    
    def run(self):
        """ This needs to be algorithm specific """
        raise NotImplemented

    @staticmethod
    def make_dir(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as e:
            print(e)
        return dir_path


class MySQLExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)
        creds = pickle.load(open("/home/vasan/src/creds.pkl", "rb"))
        self.db = MySQLDBManager(user=creds["user"],
                 host=creds["host"], 
                 password=creds["password"],
                 database=args.db,
                 table=args.table,)
    
    def save_returns(self, rets, ep_lens, savepath):
        self.db.update(episodic_returns=rets, episodic_lengths=ep_lens, model=None, metadata={})
    
    def save_args(self, fname):
        self.db.save(cfg=vars(self.args), run_id=self.run_id, episodic_returns=[], episodic_lengths=[], metadata={})
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help="Seed for random number generator")
    parser.add_argument('--checkpoint', default=5000, type=int, help="Save plots and rets every checkpoint")
    args = parser.parse_args()
        
    expt = Experiment(args)
    a = np.loadtxt("/home/vasan/src/rl_suite/rl_suite/results/sparse_reacher/20220613-233856_sac_sparse_reacher_test-7.txt")
    expt.learning_curve(a[1], a[0], "./test.png")
