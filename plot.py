from math import inf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import re
from statistics import mean

if __name__ == "__main__":
    envs = ["ball_in_cup", "dm_reacher_easy", "dm_reacher_hard"]
    envs = ["mj_reacher"]
    timeouts = [10, 25, 50, 100, 500, 1000]
    timeouts = [5000]
    plot_interval = 2000
    tasks = ["non_visual", "visual"]
    tasks = ["non_visual", ]
    steps = []

    for task in tasks:
        for env in envs:
            df = pd.DataFrame(columns=["step", "avg_ret", "seed", "timeout"])
            for timeout in timeouts:
                for seed in range(1):
                    data_folder = Path(__file__).parent/"outputs/returns"/env/task/f"timeout={timeout}"/f"seed={seed}"
                    filename = next(data_folder.glob("*.txt"))

                    with open(filename, 'r') as data_file:
                        steps = [int(float(step)) for step in data_file.readline().split()]
                        returns = [int(float(ret)) for ret in data_file.readline().split()]
            
                    step = 0
                    rets = []
                    end_step = plot_interval
                    acc_ret = 0
                    for (i, epi_s) in enumerate(steps):
                        step += epi_s
                        ret = returns[i]
                        if step > end_step:
                            if len(rets) > 0:
                                df = df.append({'step':end_step, 'avg_ret':mean(rets), 'seed':seed, 'timeout': timeout}, ignore_index=True) 

                                rets = []
                            while end_step < step:
                                end_step += plot_interval
                        
                        acc_ret += ret
                        if epi_s < timeout:
                            rets.append(acc_ret)
                            acc_ret = 0
                        else:
                            acc_ret -= 20
                
            plt.ylim(-1000, 0)
            
            sns.lineplot(x="step", y='avg_ret', data=df, hue='timeout', palette='bright')
            title = f'{task} {env} learning curves, penalty 20'
            plt.title(title)
            plt.savefig(title+'.png')
            plt.close()

            # print(f'Final return of {task} {env}: {final_ret}')
