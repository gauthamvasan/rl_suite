import seaborn as sns
import matplotlib.pyplot as plt
import re

# Ignore annoying pandas warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from pathlib import Path

# Assign colors for each choice of timeout
color_dict = {
    # simulation colors
    1:      'tab:blue',
    2:      'tab:orange',
    5:      'tab:green',
    10:     'tab:red',
    25:     'tab:purple', 
    50:     'tab:brown', 
    100:    'tab:pink', 
    500:    'tab:gray', 
    1000:   'tab:olive', 
    5000:   'tab:cyan',
    10000:  'orangered',
    20000:  'midnightblue',
    # real robot colors
    '3s':      'tab:blue',
    '6s':      'tab:orange',
    '15s':     'tab:red',
    '30s':     'tab:green',
}

def human_format_numbers(num, use_float=False):
    # Make human readable short-forms for large numbers
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if use_float:
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    return '%d%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def find_files_and_plot():
    """ Yan's original script preserved for posterity """
    for filename in Path(__file__).parent.glob('*random_stat.txt'):
        env = re.match(r".*/(.+?)_.*", str(filename)).group(1)
        df = pd.DataFrame(columns=["timeout", "seed", "hits"])
        with open(filename, 'r') as file:
            for line in file.readlines():
                match = re.match(r"timeout=(\d+), seed=(\d+): (\d+)", line)
                timeout = match.group(1)
                seed = match.group(2)
                hits = match.group(3)
                df = df.append({'timeout': int(timeout), 'seed':int(seed), 'hits': int(hits)}, ignore_index=True)

        plt.title(env)
        plt.xlabel('Timeout')
        plt.ylabel('Hits')
        ax = sns.barplot(data=df, x='timeout', y='hits')
        ax.bar_label(ax.containers[0])
        plt.savefig(f'{env}.png')
        plt.close()

def hits_vs_timeout():
    env = "point_maze_medium"
    # bp = "/home/vasan/src/rl_suite/rl_suite"
    bp = "/Users/gautham/src/rl_suite/rl_suite"   # MacOS
    fp = bp + f"/{env}_random_stat.txt"

    df = pd.DataFrame(columns=["timeout", "seed", "hits"])
    with open(fp, 'r') as file:
        for line in file.readlines():
            match = re.match(r"timeout=(\d+), seed=(\d+): (\d+)", line)
            timeout = match.group(1)
            seed = match.group(2)
            hits = match.group(3)
            df = df.append({'timeout': int(timeout), 'seed':int(seed), 'hits': int(hits)}, ignore_index=True)

    plt.title(env)
    plt.xlabel('Timeout')
    plt.ylabel('Hits')
    ax = sns.barplot(data=df, x='timeout', y='hits', palette=color_dict)
    # ax.bar_label(ax.containers[0])
    labels = ax.get_xticklabels()
    new_labels = [human_format_numbers(int(k._text)) for k in labels] 
    ax.set_xticklabels(new_labels)
    plt.savefig(f'{env}.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    hits_vs_timeout()
