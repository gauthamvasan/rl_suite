import seaborn as sns
import matplotlib.pyplot as plt
import re
import pandas as pd

from pathlib import Path

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
