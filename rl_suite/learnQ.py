import imp
import torch
import gym
import argparse
import numpy as np

from threading import Lock
from rl_suite.algo.mlp_policies import SquashedGaussianMLPActor, MLPQFunction
from rl_suite.algo.replay_buffer import SACReplayBuffer

class QLearner:
    def __init__(self, actor, obs_dim, action_dim, nn_params, device) -> None:
        self.actor= actor
        self.device = device

        self.gamma = 0.995
        self.buffer_size = 10000
        self.batch_size = 256
        self.lr = 0.0003
        self.betas = (0.9, 0.999)

        self.q = MLPQFunction(obs_dim, action_dim, nn_params, device)
        self.buffer = SACReplayBuffer(obs_dim, action_dim, capacity=self.buffer_size, batch_size=self.batch_size)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=self.lr, betas=self.betas)

    def push(self, obs, action, reward, done):
        self.buffer.add(obs, action, reward, done)
    
    def update(self):
        obs, actions, rewards, dones, next_obs = self.buffer.sample()
        obs, actions, rewards, dones, next_obs = obs.to(self.device), actions.to(self.device), \
            rewards.to(self.device), dones.to(self.device), next_obs.to(self.device)
        with torch.no_grad():
            _, policy_action, log_p, _ = self.actor(next_obs)
            next_Q = self.q(next_obs, policy_action)
            target_Q = rewards + (dones * self.gamma * next_Q)
        # get current Q estimates
        current_Q = self.q(obs, actions)

        # Ignore terminal transitions to enable infinite bootstrap
        q_loss = torch.mean((current_Q - target_Q) ** 2 * dones)
        # Optimize the critic
        self.q_opt.zero_grad()
        q_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.q_opt.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NN architecture
    net_params = {
        'mlp': {
            'hidden_sizes': [64, 64],
            'activation': "relu",
        }
    }

    # Env
    env = gym.make('Reacher-v2')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = SquashedGaussianMLPActor(obs_dim, action_dim, net_params, device)
    actor.load_state_dict(torch.load("/home/vasan/src/rl_suite/rl_suite/results/actor_100000.pt"))
    qlearner = QLearner(actor, obs_dim, action_dim, net_params, device)

    max_timesteps = 10**5
    n_warmup = 1000
    timeout = 500
    rets = []
    ep_lens = []
    i_episode = 0
    ret = 0
    step = 0
    obs = env.reset()
    for t in range(max_timesteps):
            # Select an action
            ####### Start
            # Replace the following statement with your own code for
            # selecting an action
            # a = np.random.randint(a_dim)        
            with torch.no_grad():
                x = torch.FloatTensor(obs).to(device)
                x = x.unsqueeze(0)
                mu, action, _, _ = actor(x)
            action = action.cpu()
            ####### End

            # Observe
            next_obs, r, done, infos = env.step(action)

            # Learn
            ####### Start
            # TODO: Q learner
            qlearner.push(obs, action, r, done)
            if t > n_warmup:
                qlearner.update()
            if t % 100 == 0:
                print("Step: {}, Obs: {}, Action: {}, Reward: {:.2f}, Done: {}".format(
                    t, obs[:2], action, r, done))
            obs = next_obs
            ####### End

            # Log
            ret += r
            step += 1
            if done or step == timeout:    # Bootstrap on timeout
                i_episode += 1
                rets.append(ret)
                ep_lens.append(step)
                print("Episode {} ended after {} steps with return {}".format(i_episode, step, ret))
                ret = 0
                step = 0
                torch.save(qlearner.q.state_dict(), "./q_net.pt")
                obs = env.reset()
    
    torch.save(qlearner.q.state_dict(), "./q_net.pt")


if __name__ == "__main__":
    main()
