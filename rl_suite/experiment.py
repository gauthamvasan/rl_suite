import argparse
from logging import warning
import gym
import torch
import random
import warnings
import os

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from rl_suite.mysql_db import MySQLDBManager
from rl_suite.envs.dm_control_wrapper import BallInCupWrapper, ReacherWrapper
from rl_suite.envs.dot_reacher_env import DotReacherEnv
from rl_suite.plot import smoothed_curve
from sys import platform

# For MacOS
if platform == "darwin":    
    import matplotlib as mpl
    mpl.use("TKAgg")


class Experiment:
    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Task
        parser.add_argument('--seed', default=0, type=int, help="Seed for random number generator")
        parser.add_argument('--checkpoint', default=5000, type=int, help="Save plots and rets every checkpoint")
        self.args = parser.parse_args()
        return self.args

    def make_env(self):
        if self.args.env == "ball_in_cup":
            env = BallInCupWrapper(seed=self.args.seed, timeout=self.args.timeout)
        elif self.args.env == "sparse_reacher":
            env = ReacherWrapper(seed=self.args.seed, tol=self.args.tol, timeout=self.args.timeout)
        elif self.args.env == "dot_reacher":
            env = DotReacherEnv(pos_tol=self.args.pos_tol, vel_tol=self.args.vel_tol, 
                    dt=self.args.dt, timeout=self.args.timeout, clamp_action=self.args.clamp_action)
        else:
            env = gym.make(self.args.env)
            env.seed(self.args.seed)
        env.name = self.args.env
        return env

    def save_returns(self, rets, ep_lens, savepath):
        """ Save learning curve data as a numpy text file 

        Args:
            rets (list/array): A list or array of episodic returns
            ep_lens (list/array):  A list or array of episodic length
            savepath (str): Save path
        """
        data = np.zeros((2, len(rets)))
        data[0] = ep_lens
        data[1] = rets
        np.savetxt(savepath, data)
    
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
        
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(e)
            warnings.warn("Unable to use deterministic algorithms with pytorch & cuda!!")

    def learning_curve(self, rets, ep_lens, save_fig=""):
        plot_rets, plot_x = smoothed_curve(
                np.array(rets), np.array(ep_lens), x_tick=self.args.checkpoint, window_len=self.args.checkpoint)
        if len(plot_rets):
            plt.clf()
            plt.plot(plot_x, plot_rets)
            plt.pause(0.001)
            if save_fig:
                plt.savefig(save_fig)
    
    def run(self):
        """ This needs to be algorithm specific """
        raise NotImplemented

    @staticmethod
    def make_dir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            pass
        return dir_path

if __name__ == "__main__":
    a = np.loadtxt("/home/vasan/src/rl_suite/rl_suite/results/sparse_reacher/20220613-233856_sac_sparse_reacher_test-7.txt")
    expt = Experiment()
    expt.learning_curve(a[1], a[0], "./test.png")
