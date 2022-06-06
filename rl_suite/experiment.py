import argparse
import gym
import numpy as np

from datetime import datetime
from rl_suite.mysql_db import MySQLDBManager
from rl_suite.envs.dm_control_wrapper import BallInCupWrapper, ReacherWrapper


def make_env(name, *args, **kwargs):
    if name == "ball_in_cup":
        env = BallInCupWrapper(seed, timeout=timeout)
    elif name == "sparse_reacher":
        env = ReacherWrapper(seed=seed, tol=tol, timeout=timeout)
    else:
        env = gym.make(name)
    env.name = name


class Experiment:
    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")   

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Task
        parser.add_argument('--seed', default=0, type=int, help="Seed for random number generator")
        args = parser.parse_args()
        return args

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
    

    
