import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

envs = ["ball in cup", "dm reacher easy", "dm reacher hard"]
envs = ["ball in cup 20"]
timeouts = ["1", "2", "5", "10", "25", "50", "100", "500", "1000"]

for env in envs:
    df = pd.DataFrame(columns=["length", "timeout"])

    for timeout in timeouts:
        with open(env+"_timeout="+str(timeout)+"_random_stat.txt") as data_file:
            lines = data_file.readlines()

        lengths = [ int(length) for length in lines[:-2]]
        temp_df = pd.DataFrame(columns=["length", "timeout"])
        temp_df["length"] = lengths
        temp_df["timeout"] = timeout

        df = df.append(temp_df)

    sns.boxplot(x="length", y="timeout", data=df)
    plt.title(env)
    plt.savefig(env+".png")
    plt.close()
