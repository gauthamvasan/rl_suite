import glob
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from rl_suite.plot import setsizes, setaxes, human_format_numbers, smoothed_curve, set_labels


def smoothed_plot(data, x_tick=1000, window_len=1000):
    """
    Args:
        data: Numpy 2-D array. Row 0 contains episode lengths/timesteps and row 1 contains episodic returns
    Returns:
    """
    color = "tab:blue"
    returns = data[1]
    ep_lens = data[0]

    rets, t = smoothed_curve(returns=returns, ep_lens=ep_lens, x_tick=x_tick, window_len=window_len)
    plt.plot(t, rets, color=color, linewidth=2)
    # plt.fill_between(x, rets - std_errs, rets + std_errs, alpha=0.6)
    plt.xlabel('Timesteps', fontweight='bold', fontsize=14)
    h = plt.ylabel("Return", labelpad=25, fontweight='bold', fontsize=14)
    h.set_rotation(0)
    plt.pause(0.001)
    plt.grid()
    plt.tight_layout()
    plt.show()

def plotter(cfg, title, xlim=None, ylim=None, 
            legend_loc=None, x_tick=5000, window_len=5000, save_path="./"):
    """ Plot confidence intervals for multiple runs.

    Args:
        basepath (str): Root folder with results
        color_palette (dict): {env: color}
        title (str): Plot title, fname for pdf/png file
        ylim (list, optional): ylim for plot. Defaults to None.
        legend_loc (str, optional): Options in ["best", "lower right", "upper left", etc]. Defaults to None.
        x_tick (int, optional): Plot point every N steps. Defaults to 5000.
        window_len (int, optional): Bin size for each plot point. Defaults to 5000.
        save_path (str, optional): Path to save figure. Defaults to "./".

    Returns:
        _type_: _description_
    """    
    print("-"*50)
    print(cfg)
    print("-"*50)
    setsizes()
    setaxes()

    color_palette = {}
    for env, vals in cfg.items():
        color_palette[env] = vals['color']

    df = pd.DataFrame(columns=["step", "avg_ret", "seed", "timeout"])
    legend_elements = []
    for env, vals in cfg.items():
        basepath = vals['basepath']
        color = vals['color']
        all_paths = glob.glob(f"{basepath}/{env}/*_returns.txt")
        assert len(all_paths) > 0, print(f"{basepath}/{env}/*_returns.txt")
        print(f"{len(all_paths)} seeds were found.")
        counter = 0
        last_return = []
        for fp in all_paths:
            counter += 1
            data = np.loadtxt(fp)
            try:
                rets, timesteps = smoothed_curve(data[1], data[0], x_tick, window_len)
                last_return.append(rets[-1])
                for r, t in zip(rets, timesteps):
                    df = pd.concat([df, pd.DataFrame.from_records([{'env': env, 'seed':counter, 'step':t, 'avg_ret':r}])])
            except IndexError as e:
                print(f"Run {fp} incomplete. It was not added to the plot")
        legend_elements.append(Line2D([0], [0], color=color, lw=4, label=env),)
        print("Average return in the last {} steps = {:.2f}".format(window_len, np.mean(last_return)))
        
    # Plot
    # sns.lineplot(x="step", y='avg_ret', data=df[df['env']==env], hue='env', palette=color_palette)
    sns.lineplot(x="step", y='avg_ret', data=df, hue='env', legend='brief', palette=color_palette)
    set_labels(title, xlabel="Timesteps", ylabel="Return", labelpad=25)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if legend_loc is not None:
        # plt.legend(handles=legend_elements, loc='lower right')
        plt.legend(handles=legend_elements, loc=legend_loc)
    else:
        plt.legend([], [], frameon=False)

    plt.tight_layout()
    plt.locator_params(axis='x', nbins=8)
    # plt.savefig(os.path.join(save_path, title+'.png'), dpi=200)
    plt.savefig(os.path.join(save_path, title+'.png'))
    plt.show()
    plt.close()


def sac_baseline_plot():
    # Mean CI plot
    basepath = "/home/vasan/scratch/sac_baseline"
    # envs = ["Hopper-v4", "Humanoid-v4", "Ant-v4", "Reacher-v4",
    #         "HalfCheetah-v4", "Swimmer-v4", "Walker2d-v4", "InvertedDoublePendulum-v4"]
    envs = ["Hopper-v2", "Humanoid-v2", "Ant-v2", "Reacher-v2",
        "HalfCheetah-v2", "Swimmer-v2", "Walker2d-v2", "InvertedDoublePendulum-v2"]

    for env in envs:
        cfg = {
            env: {'color': "tab:orange", 'basepath': basepath},
            f'{env}-nb': {'color': "tab:green", 'basepath': f"{basepath}/no_bootstrap"},
        }
        title = env
        plotter(cfg, title, xlim=[0, 1000000], ylim=None, 
            save_path="./results/nb_comparison", legend_loc="best")

def point_maze_dense_plot():
    basepath = "/home/vasan/scratch/tro_paper"
    color_palette = {
        'point_maze_U_dense': "darkmagenta", 
        'point_maze_open_dense': "mediumaquamarine",
        "point_maze_small_dense": "deeppink",
    }
    title = "point_maze_dense"
    legend_loc = "best"
    save_path="./results"
    plotter(basepath, color_palette, title, ylim=None, save_path="./results")

def point_maze_sparse_plot():
    basepath = "/home/vasan/scratch/tro_paper"
    color_palette = {
        'point_maze_U_sparse': "darkorange", 
        'point_maze_open_sparse': "royalblue",
        "point_maze_small_sparse": "teal",
    }
    title = "point_maze_sparse"
    legend_loc = "best"
    save_path="./results"
    plotter(basepath, color_palette, title, ylim=None, save_path="./results")

def point_maze_steps_to_goal_plot():
    x_tick = 10000
    window_len = 20000
    df = pd.DataFrame(columns=["step", "avg_ret", "seed", "timeout"])
    legend_elements = []
    N = 500000

    # color_palette = {
    #     "point_maze_open_sparse": "mediumaquamarine",
    #     "point_maze_open_dense": "darkmagenta",
    #     "point_maze_medium_sparse": "mediumaquamarine",
    #     "point_maze_medium_dense": "royalblue",
    #     "point_maze_U_sparse": "teal",
    #     "point_maze_U_dense": "deeppink",
    #     "point_maze_min_time_sparse": "darkorange",
    #     "point_maze_min_time_dense": "royalblue",
    # }

    color_palette = {
        "point_maze_open_sparse": "darkorange",
        "point_maze_open_dense": "royalblue",
        "point_maze_medium_sparse": "darkorange",
        "point_maze_medium_dense": "royalblue",
        "point_maze_U_sparse": "darkorange",
        "point_maze_U_dense": "royalblue",
        "point_maze_min_time_sparse": "darkorange",
        "point_maze_min_time_dense": "royalblue",
    }

    # basepath = "/home/vasan/scratch/tro_paper/min_time_K500"
    basepath = "/home/vasan/scratch/tro_paper"
    key = "open"

    for env, color in color_palette.items():
        if key not in env:
            continue

        fp = f"{basepath}/{env}/*.txt"
        all_paths = glob.glob(fp)
        assert len(all_paths) > 0
        counter = 0
        for fp in all_paths:
            counter += 1
            data = np.loadtxt(fp)
            try:
                rets, timesteps = smoothed_curve(data[1], data[0], x_tick=x_tick, window_len=window_len)
                steps_to_goal, timesteps = smoothed_curve(data[0], data[0], x_tick=x_tick, window_len=window_len)
                for i, (r, t) in enumerate(zip(rets, timesteps)):
                    df = pd.concat([df, pd.DataFrame.from_records([{'env': env, 'seed':counter, 'step':t, 'avg_ret':r, 'steps_to_goal': steps_to_goal[i]}])])
            except IndexError as e:
                print(f"Run {fp} incomplete. It was not added to the plot")
        legend_elements.append(Line2D([0], [0], color=color, lw=4, label='Minimum-time Task' if 'sparse' in env else 'Guiding Reward Task'),)

    # plt.figure()
    setsizes()
    setaxes()
    plt.xlim([0, N])
    # plt.ylim([0, 1500])
    plt.yscale('log')
    # sns.lineplot(x="step", y='avg_ret', data=df[df['env']==env], hue='env', palette=color_palette)
    print(color_palette)
    # sns.lineplot(x="step", y='avg_ret', data=df[df['env'].str.contains('point_maze_T_dense')], hue='env', palette=color_palette)
    sns.lineplot(x="step", y='steps_to_goal', data=df[df['env'].str.contains(key)], hue='env', palette=color_palette)
    set_labels("", labelpad=25, xlabel="Timesteps", ylabel="Steps\nto\nGoal")

    # plt.legend(handles=legend_elements, loc='lower right')
    plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    plt.locator_params(axis='x', nbins=8)
    plt.savefig(f'point_maze_{key}_steps_to_goal.pdf', dpi=200)
    plt.show()
    plt.close()

if __name__ == '__main__':
    # Smooth plot
    # x_tick = window_len = 5000
    # fp = "results/ppo_visual_reacher_bs-2048_0.txt"
    # data = np.loadtxt(fp)
    # smoothed_plot(data, x_tick, window_len)

    # point_maze_sparse_plot()
    # sac_baseline_plot()
    point_maze_steps_to_goal_plot()
