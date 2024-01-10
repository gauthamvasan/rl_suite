"""
Simple example script that shows how to interace with dm_control  

acrobot swingup
acrobot swingup_sparse
ball_in_cup catch
cartpole balance
cartpole balance_sparse
cartpole swingup
cartpole swingup_sparse
cheetah run
finger spin
finger turn_easy
finger turn_hard
fish upright
fish swim
hopper stand
hopper hop
humanoid stand
humanoid walk
humanoid run
manipulator bring_ball
pendulum swingup
point_mass easy
reacher easy
reacher hard
swimmer swimmer6
swimmer swimmer15
walker stand
walker walk
walker run

"""

import argparse
import numpy as np
from dm_control import suite


def interaction(domain_name, task_name, seed=1):
    # Load one task:
    env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})
    print(env.observation_spec())

    # Step through an episode and print out reward, discount and observation.
    action_spec = env.action_spec()
    EP = 50
    rets = []
    ep_lens = []
    for i in range(EP):
        time_step = env.reset()
        # print(time_step.observation.keys())
        steps = 0
        ret = 0
        while not time_step.last():
            action = np.random.uniform(action_spec.minimum,
                                       action_spec.maximum,
                                       size=action_spec.shape)
            time_step = env.step(action)
            # print(steps, action, time_step.reward, time_step.observation)
            steps += 1
            ret += time_step.reward
        rets.append(ret)
        ep_lens.append(steps)
        print("Episode: {} ended in {} steps with return: {}".format(i+1, steps, ret))

    # Random policy stats
    rets = np.array(rets)
    ep_lens = np.array(ep_lens)
    print("Mean: {:.2f}".format(np.mean(rets)))
    print("Standard Error: {:.2f}".format(np.std(rets) / np.sqrt(len(rets) - 1)))
    print("Median: {:.2f}".format(np.median(rets)))
    print("Max length:", max(ep_lens))
    print("Min length:", min(ep_lens))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', '-d', default="acrobot", type=str)
    parser.add_argument('--task', '-t', default="swingup", type=str)
    args = parser.parse_args()

    interaction(domain_name=args.domain, task_name=args.task, seed=42)


