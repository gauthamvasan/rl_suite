import glob
import os
import scipy.stats

import matplotlib.pyplot as plt
import numpy as np


def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3

    plt.rcParams['xtick.labelsize'] = 14.0
    plt.rcParams['ytick.labelsize'] = 14.0
    plt.rcParams['xtick.direction'] = "out"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['lines.linewidth'] = 3.0
    plt.rcParams['ytick.minor.pad'] = 50.0

def setaxes():
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='out', which='minor', width=2, length=3,
                   labelsize=12, pad=8)
    ax.tick_params(axis='both', direction='out', which='major', width=2, length=8,
                   labelsize=12, pad=8)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    for tick in ax.xaxis.get_major_ticks():
        # tick.label.set_fontsize(getxticklabelsize())
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        # tick.label.set_fontsize(getxticklabelsize())
        tick.label.set_fontsize(14)

def human_format_numbers(num, use_float=True):
    # Make human readable short-forms for large numbers
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if use_float:
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    return '%d%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def set_labels(title, xlabel="Timesteps", ylabel="Return", labelpad=25):
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    h = plt.ylabel(ylabel, fontsize=14, labelpad=labelpad)
    h.set_rotation(0)
    (ticks, texts) = plt.xticks()
    try:
        new_labels = [human_format_numbers(int(k.get_text())) for k in texts]
    except:
        new_labels = [human_format_numbers(int(k)) for k in ticks]
    plt.xticks(ticks=ticks, labels=new_labels)
    plt.grid()
    plt.tight_layout()

def smoothed_curve(returns, ep_lens, x_tick=5000, window_len=5000):
    """
    Args:
        returns: 1-D numpy array with episodic returs
        ep_lens: 1-D numpy array with episodic returs
        x_tick (int): Bin size
        window_len (int): Length of averaging window
    Returns:
        A numpy array
    """
    rets = []
    x = []
    cum_episode_lengths = np.cumsum(ep_lens)

    if cum_episode_lengths[-1] >= x_tick:
        y = cum_episode_lengths[-1] + 1
        steps_show = np.arange(x_tick, y, x_tick)

        for i in range(len(steps_show)):
            rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_len)) *
                                     (cum_episode_lengths < x_tick * (i + 1))]
            if rets_in_window.any():
                rets.append(np.mean(rets_in_window))
                x.append((i+1) * x_tick)

    return np.array(rets), np.array(x)
    

def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data, axis=0), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def get_confidence_intervals(plt_cfg, confidence, return_scale=1):
    all_files = glob.glob(os.path.join(plt_cfg["fp"], "*.txt"))
    plt_files = [x for x in all_files if plt_cfg["key"] in x]

    all_data = []
    n_curves = 0
    for dfp in plt_files:
        data = np.loadtxt(dfp)

        if plt_cfg["timeout"] is None:
            returns = data[1] * return_scale
            ep_lens = data[0]
        else:
            raw_returns = data[1] * return_scale
            raw_ep_lens = data[0]

            prev_steps = 0
            prev_ret = 0
            returns = []
            ep_lens = []
            for steps, ret in zip(raw_ep_lens, raw_returns):
                if steps == plt_cfg["timeout"]:
                    prev_steps += steps
                    prev_ret += ret
                else:
                    ep_lens.append(steps + prev_steps)
                    returns.append(ret + prev_ret)
                    prev_steps = 0
                    prev_ret = 0
            returns = np.array(returns)
            ep_lens = np.array(ep_lens)

        rets, x = smoothed_curve(returns, ep_lens, plt_cfg["x_tick"], plt_cfg["window_len"])
        n_pts = plt_cfg["N"]//plt_cfg["x_tick"]
        # print(sum(ep_lens), n_pts, len(rets))
        if sum(ep_lens) >= plt_cfg["N"] and len(rets) >= n_pts:
            rets = rets[:n_pts]        
            all_data.append(rets)
            n_curves += 1
        
    avg_rets, ci_low, ci_up = mean_confidence_interval(np.vstack(all_data), confidence=confidence)
    ci_low = np.clip(ci_low, a_max=0, a_min=-10000)
    ci_up = np.clip(ci_up, a_max=0, a_min=-10000)

    return avg_rets, ci_low, ci_up, n_curves

def confidence_interval_plot(plt_cfg, confidence=0.95):
    # Make plots pretty :)
    setaxes()
    setsizes()
    
    avg_rets, ci_low, ci_up, n_curves = get_confidence_intervals(plt_cfg, confidence)
    
    # Plot the confidence interval
    x = np.arange(1, len(avg_rets) + 1) * plt_cfg["x_tick"]
    plt.plot(x, avg_rets)    
    plt.fill_between(x, ci_low, ci_up, color='blue', alpha=0.1)

    locs, labels = plt.xticks()
    new_labels = [human_format_numbers(k) for k in locs[1:]]
    plt.xticks(ticks=locs[1:], labels=new_labels)
    
    # Labels & titles
    plt.xlabel('Steps', fontsize=15)
    h = plt.ylabel("Return", labelpad=25, fontsize=15)
    h.set_rotation(0)
    plt.title(plt_cfg["title"], fontsize=17)
    plt.grid()
    plt.tight_layout()    
    
    if plt_cfg["save_path"]:
        plt.savefig(plt_cfg["save_path"], dpi=200)
    else:
        plt.show()
