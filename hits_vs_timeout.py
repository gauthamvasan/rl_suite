import seaborn as sns
import matplotlib.pyplot as plt
import re

from pathlib import Path

for filename in Path(__file__).parent.glob('*random_stat.txt'):
    env = re.match(r".*/(.+?)_.*", str(filename)).group(1)
    timeouts = []
    hits_list = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            match = re.match(r"timeout=(\d+): (\d+)", line)
            timeout = match.group(1)
            hits = match.group(2)
            timeouts.append(timeout)
            hits_list.append(int(hits))
    plt.title(env)
    sns.barplot(x=timeouts, y=hits_list)
    plt.savefig(f'{env}.png')
    plt.close()
