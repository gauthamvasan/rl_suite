import torch
import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from rl_suite.algo.mlp_policies import MLPDiscreteActor, SACDiscreteCritic
from copy import deepcopy


class SAC_Discrete:
    """ SAC with Automatic Entropy Adjustment. """
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = device
        self.gamma = cfg.gamma
        self.critic_tau = cfg.critic_tau
        self.encoder_tau = cfg.encoder_tau
        self.actor_update_freq = cfg.actor_update_freq
        self.critic_target_update_freq = cfg.critic_target_update_freq

        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.alpha_lr = cfg.alpha_lr

        self.actor = MLPDiscreteActor(cfg.obs_dim, cfg.action_dim, cfg.actor_nn_params, device)
        self.critic = SACDiscreteCritic(cfg.obs_dim, cfg.action_dim, cfg.critic_nn_params, device)
        self.critic_target = deepcopy(self.critic) # also copies the encoder instance

        self.log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = 0.98 * (-np.log(1 / self.cfg.action_dim))        

        self.num_updates = 0

        # optimizers
        self.init_optimizers()
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()
        self.critic_target.share_memory()
        self.log_alpha.share_memory_()

    def init_optimizers(self):
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr, betas=(0.9, 0.999), weight_decay=self.cfg.l2_reg,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, betas=(0.9, 0.999), weight_decay=self.cfg.l2_reg,
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=(0.5, 0.999), weight_decay=self.cfg.l2_reg,
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def sample_action(self, x):
        with torch.no_grad():
            if not isinstance(x, torch.FloatTensor):
                x = torch.FloatTensor(x).to(self.device)
                x = x.unsqueeze(0)
            dist, _ = self.actor(x)
            action = dist.sample()
            return action.cpu().data.numpy().flatten()[0]


    def update(self, obs, action, reward, done, next_obs):
        # Move tensors to device
        obs, action, reward, next_obs, done = obs.to(self.device), action.to(self.device), \
            reward.to(self.device), next_obs.to(self.device), done.to(self.device)

        # print(obs.shape, action.shape, reward.shape, next_obs.shape, done.shape)
        # regular update of SAC_RAD, sequentially augment data and train

        # Calculating the Q-Value target
        with torch.no_grad():
            _, next_probs = self.actor(next_obs)
            next_log_probs = torch.log(next_probs)
            next_q1, next_q2 = self.critic_target(next_obs)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(-1)
            target_Q = reward + self.gamma * (1 - done) * next_v

        q1, q2 = self.critic(obs)
        q1 = q1.gather(1, action.long()).view(-1)
        q2 = q2.gather(1, action.long()).view(-1)
        critic_loss = torch.mean((q1 - target_Q) ** 2 + (q2 - target_Q) ** 2)

        # Calculating the Policy target
        _, probs = self.actor(obs)
        log_probs = torch.log(probs)
        with torch.no_grad():
            q1, q2 = self.critic(obs)
            q = torch.min(q1, q2)

        actor_loss = (probs * (self.alpha.detach() * log_probs - q)).sum(-1).mean()        

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        log_probs = (probs * log_probs).sum(-1)
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()      

        self.num_updates += 1
        if self.num_updates % self.cfg.critic_target_update_freq == 0:
                # self.hard_update_target_network()
                self.soft_update_target()

        stats = {
            'train_actor/loss': actor_loss.item(),
            'train_actor/target_entropy': self.target_entropy.item(),            
            'train_alpha/loss': alpha_loss.item(),
            'train_alpha/value': self.alpha.item(),
            'train/batch_reward': reward.mean().item(),
            'train/num_updates': self.num_updates,
        }

        return stats

    @staticmethod
    def soft_update_params(net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def soft_update_target(self):
        self.soft_update_params(
            self.critic.Q1, self.critic_target.Q1, self.critic_tau
        )
        self.soft_update_params(
            self.critic.Q2, self.critic_target.Q2, self.critic_tau
        )


    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )


    def hard_update_target_network(self):
        self.critic_target.Q1.load_state_dict(self.critic.Q1.state_dict())
        self.critic_target.Q1.eval()
        self.critic_target.Q2.load_state_dict(self.critic.Q2.state_dict())
        self.critic_target.Q2.eval()


class SAC_DiscreteAgent(SAC_Discrete):
    def __init__(self, cfg, buffer, device=torch.device('cpu')):
        super().__init__(cfg, device)
        self._replay_buffer = buffer
        self.steps = 0

    def push_and_update(self, obs, action, reward, done):
        self._replay_buffer.add(obs, action, reward, done)
        
        if self.steps > self.cfg.init_steps and (self.steps % self.cfg.update_every == 0):
            for _ in range(self.cfg.update_epochs):
                # tic = time.time()
                stat = self.update(*self._replay_buffer.sample())
                # print(time.time() - tic)
            return stat
        
        self.steps += 1
