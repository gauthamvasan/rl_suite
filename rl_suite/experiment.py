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

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
# from rl_suite.mysql_db import MySQLDBManager
from rl_suite.envs.dot_reacher_env import DotReacherEnv, VisualDotReacherEnv
from rl_suite.envs.gym_wrapper import MountainCarContinuous
from rl_suite.envs.visual_reacher import MJReacherWrapper
from rl_suite.envs.dm_control_wrapper import ReacherWrapper, BallInCupWrapper
from rl_suite.plot.plot import smoothed_curve
from sys import platform
from pathlib import Path

# For MacOS
if platform == "darwin":    
    import matplotlib as mpl
    mpl.use("Qt5Agg")


class Experiment:
    def __init__(self, args):
        self.run_id = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        self.args = args

        assert not args.experiment_dir.startswith('/'), 'experiment_dir must use relative path'

        self._return_dir = Path(args.results_dir)/args.experiment_dir/'returns'
        self._model_dir = Path(args.results_dir)/args.experiment_dir/'models'

        self.make_dir(self._return_dir)
        self.make_dir(self._model_dir)

        hyperparas_dict = vars(self.args)
        hyperparas_dict["device"] = str(hyperparas_dict["device"])
        json.dump(hyperparas_dict, open(self._return_dir/f"{self.run_id}_args.json", 'w'), indent=4)

    def make_env(self):
        if self.args.env == "ball_in_cup":
            env = BallInCupWrapper(seed=self.args.seed, penalty=self.args.reward, use_image=self.args.algo=="sac_rad")
        elif self.args.env == "dm_reacher_easy":
            env = ReacherWrapper(seed=self.args.seed, penalty=self.args.reward, mode="easy", use_image=self.args.algo=="sac_rad")
        elif self.args.env == "dm_reacher_hard":
            env = ReacherWrapper(seed=self.args.seed, penalty=self.args.reward, mode="hard", use_image=self.args.algo=="sac_rad")
        elif self.args.env == "dm_reacher_torture":
            env = ReacherWrapper(seed=self.args.seed, penalty=self.args.reward, mode="torture", use_image=self.args.algo=="sac_rad")
        elif self.args.env == "dot_reacher":
            if self.args.algo=="sac":
                env = DotReacherEnv(pos_tol=self.args.pos_tol, vel_tol=self.args.vel_tol, penalty=self.args.reward,
                    dt=self.args.dt, timeout=self.args.timeout, clamp_action=self.args.clamp_action)
            else:
                env = VisualDotReacherEnv(pos_tol=self.args.pos_tol, vel_tol=self.args.vel_tol, penalty=self.args.reward,
                    dt=self.args.dt, timeout=self.args.timeout, clamp_action=self.args.clamp_action)
        elif self.args.env == "mountain_car_continuous":
            env = MountainCarContinuous(timeout=self.args.timeout)
            env.env.seed(self.args.seed)
        elif self.args.env == "mj_reacher":
            env = MJReacherWrapper(tol=self.args.tol, penalty=self.args.reward, use_image=self.args.algo=="sac_rad")            
        else:
            raise NotImplementedError()
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
        np.savetxt(self._return_dir/f'{self.run_id}_returns.txt', data)
    
    def save_model(self, step):
        self.learner.save(self._model_dir, step)

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
                plt.savefig(self._return_dir/f"{self.run_id}_learning_curve.png")
    
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
    