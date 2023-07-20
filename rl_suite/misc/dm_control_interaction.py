"""
Simple example script that shows how to interace with dm_control  
"""

import numpy as np
from dm_control import suite


def interaction(domain_name, task_name, seed=1):
    # Load one task:
    env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})
    
    # Step through an episode and print out reward, discount and observation.
    action_spec = env.action_spec()
    EP = 50
    rets = []
    ep_lens = []
    for i in range(EP):
        time_step = env.reset()
        steps = 0
        ret = 0
        while not time_step.last():
            action = np.random.uniform(action_spec.minimum,
                                       action_spec.maximum,
                                       size=action_spec.shape)
            time_step = env.step(action)
            print(steps, action, time_step.reward, time_step.observation)
            steps += 1
            ret += time_step.reward
        rets.append(ret)
        ep_lens.append(steps)
        print('-' * 100)
        print("Episode: {} ended in {} steps with return: {}".format(i+1, steps, ret))
        print('-' * 100)

    # Random policy stats
    rets = np.array(rets)
    ep_lens = np.array(ep_lens)
    print("Mean: {:.2f}".format(np.mean(ep_lens)))
    print("Standard Error: {:.2f}".format(np.std(ep_lens) / np.sqrt(len(ep_lens) - 1)))
    print("Median: {:.2f}".format(np.median(ep_lens)))
    inds = np.where(ep_lens == env._timeout)
    print("Success Rate (%): {:.2f}".format((1 - len(inds[0]) / len(ep_lens)) * 100.))
    print("Max length:", max(ep_lens))
    print("Min length:", min(ep_lens))


if __name__ == "__main__":
    interaction(domain_name="pendulum", task_name="swingup", seed=1)
