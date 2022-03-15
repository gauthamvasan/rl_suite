import matplotlib.pyplot as plt
import numpy as np


def raw_plot(data):
    """
    Args:
        data: Numpy 2-D array. Row 0 contains episode lengths/timesteps and row 1 contains episodic returns
    Returns:
    """
    plt.plot(np.cumsum(data[0]), data[1])
    plt.xlabel('Steps')
    h = plt.ylabel("Return", labelpad=25)
    h.set_rotation(0)
    plt.pause(0.001)
    plt.show()

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
    plt.xlabel('Steps', fontweight='bold', fontsize=14)
    h = plt.ylabel("Return", labelpad=25, fontweight='bold', fontsize=14)
    h.set_rotation(0)
    plt.pause(0.001)
    plt.grid()
    plt.tight_layout()
    plt.show()

def avg_run_plot(data, x_tick):
    """
    Args:
        data: 2D array; Each row contains returns, equally spaced bins
    """

    color = "tab:blue"
    avg_rets = np.mean(data, axis=0)

    std_errs = np.std(data, axis=0) / (len(data) - 1)
    x = np.arange(1, len(avg_rets) + 1) * x_tick
    plt.plot(x, avg_rets, color=color, linewidth=2)
    plt.fill_between(x, avg_rets - std_errs, avg_rets + std_errs, alpha=0.6)
    plt.xlabel('Steps')
    h = plt.ylabel("Return", labelpad=25)
    h.set_rotation(0)
    plt.pause(0.001)
    plt.show()



if __name__ == '__main__':
    # data = np.loadtxt("mover0.txt"); plt.close()

    # Plot all data without smoothing
    # raw_plot(data)
    x_tick = window_len = 5000
    fp = "results/ppo_visual_reacher_bs-2048_0.txt"
    data = np.loadtxt(fp)
    smoothed_plot(data, x_tick, window_len)
