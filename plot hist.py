import seaborn as sns
import matplotlib.pyplot as plt

t1000_file = open('first 20k t1000_returns.txt', 'r')
t5000_file = open('first 20k t5000_returns.txt', 'r')
# t5000_r_point_1_file = open('5000_r_0.1_returns.txt', 'r')

rets = [ int(float(ret)) for ret in (t1000_file.readlines()[1]).split() ]
plt.xlim((-4000, 0))
sns.histplot(rets, bins=len(rets))
plt.title('timeout = 1000')
plt.xlabel('first 20k episode length')
plt.savefig('timeout=1000_hist.png')

rets = [ int(float(ret)) for ret in (t5000_file.readlines()[1]).split() ]
plt.figure()
plt.xlim((-4000, 0))
sns.histplot(rets, bins=len(rets))
plt.title('timeout = 5000')
plt.xlabel('first 20k episode length')
plt.savefig('timeout=5000_hist.png')

# rets = [ int(float(ret)) for ret in (t5000_r_point_1_file.readlines()[1]).split() ]
# plt.figure()
# sns.histplot(rets)
# plt.title('timeout = 5000, reward = -0.1')
# plt.xlabel('episode length')
# plt.savefig('timeout=5000, reward=-0.1_hist.png')