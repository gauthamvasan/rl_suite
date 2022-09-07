import torch
import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from rl_suite.algo.mlp_policies import SquashedGaussianMLPActor, SquashedGaussianMLP_ResetActionActor, SACCritic
from copy import deepcopy


class SAC:
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

        self.actor = SquashedGaussianMLPActor(cfg.obs_dim, cfg.action_dim, cfg.actor_nn_params, device)
        self.critic = SACCritic(cfg.obs_dim, cfg.action_dim, cfg.critic_nn_params, device)
        self.critic_target = deepcopy(self.critic) # also copies the encoder instance

        self.log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(cfg.action_shape)

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

    def sample_action(self, x, deterministic=False):
        with torch.no_grad():
            if not isinstance(x, torch.FloatTensor):
                x = torch.FloatTensor(x).to(self.device)
                x = x.unsqueeze(0)
            mu, action, _, _ = self.actor(x)
            if deterministic:
                return mu.cpu().data.numpy().flatten()
            else:
                return action.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            _, policy_action, log_p, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_p
            if self.cfg.bootstrap_terminal:
                # enable infinite bootstrap
                target_Q = reward + (self.cfg.gamma * target_V)
            else:
                target_Q = reward + ((1.0 - done) * self.cfg.gamma * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = torch.mean(
            (current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2
        )
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        critic_stats = {
            'train_critic/loss': critic_loss.item()
        }

        return critic_stats

    def update_actor_and_alpha(self, obs):
        # detach encoder, so we don't update it with the actor loss
        _, action, log_p, log_std = self.actor(obs)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_p - actor_Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_p - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        actor_stats = {
            'train_actor/loss': actor_loss.item(),
            'train_actor/target_entropy': self.target_entropy.item(),
            'train_actor/entropy': entropy.mean().item(),
            'train_alpha/loss': alpha_loss.item(),
            'train_alpha/value': self.alpha.item(),
            'train/entropy': entropy.mean().item(),
        }
        return actor_stats

    def update(self, obs, action, reward, done, next_obs):
        # Move tensors to device
        obs, action, reward, next_obs, done = obs.to(self.device), action.to(self.device), \
            reward.to(self.device), next_obs.to(self.device), done.to(self.device)

        # print(obs.shape, action.shape, reward.shape, next_obs.shape, done.shape)
        # regular update of SAC_RAD, sequentially augment data and train
        stats = self.update_critic(obs, action, reward, next_obs, done)
        if self.num_updates % self.actor_update_freq == 0:
            actor_stats = self.update_actor_and_alpha(obs)
            stats = {**stats, **actor_stats}
        if self.num_updates % self.critic_target_update_freq == 0:
            self.soft_update_target()
        stats['train/batch_reward'] = reward.mean().item()
        stats['train/num_updates'] = self.num_updates
        self.num_updates += 1
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
        # self.soft_update_params(
        #     self.critic.encoder, self.critic_target.encoder,
        #     self.encoder_tau
        # )

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

class SAC_ResetAction:
    """ SAC with Automatic Entropy Adjustment. """
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = device
        self.gamma = cfg.gamma
        self.critic_tau = cfg.critic_tau
        self.actor_update_freq = cfg.actor_update_freq
        self.critic_target_update_freq = cfg.critic_target_update_freq

        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.alpha_lr = cfg.alpha_lr

        self.actor = SquashedGaussianMLP_ResetActionActor(cfg.obs_dim+1, cfg.action_dim+1, cfg.actor_nn_params, device)
        self.critic = SACCritic(cfg.obs_dim, cfg.action_dim, cfg.critic_nn_params, device)
        self.reset_critic = SACCritic(cfg.obs_dim+1, 1, cfg.critic_nn_params, device)

        if cfg.load_step > -1:
            self.actor.load_state_dict(
                torch.load('%s/actor_%s.pt' % (cfg.model_dir, cfg.load_step))
            )
            self.critic.load_state_dict(
                torch.load('%s/critic_%s.pt' % (cfg.model_dir, cfg.load_step))
            )
            self.reset_critic.load_state_dict(
                torch.load('%s/reset_critic_%s.pt' % (cfg.model_dir, cfg.load_step))
            )

        self.critic_target = deepcopy(self.critic) 
        self.reset_critic_target = deepcopy(self.reset_critic)

        self.log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(cfg.action_shape)

        self.num_updates = 0

        # optimizers
        self.init_optimizers()
        self.train()
        self.critic_target.train()
        self.reset_critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.reset_critic.train(training)

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()
        self.critic_target.share_memory()
        self.log_alpha.share_memory_()
        self.reset_critic.share_memory()
        self.reset_critic_target.share_memory()

    def init_optimizers(self):
        self.action_module_optimizer = torch.optim.Adam(
            self.actor.get_action_module_parameters(), lr=self.actor_lr, betas=(0.9, 0.999), weight_decay=self.cfg.l2_reg
        )

        self.reset_action_module_optimizer = torch.optim.Adam(
            self.actor.get_reset_action_module_parameters(), lr=self.actor_lr, betas=(0.9, 0.999), weight_decay=self.cfg.l2_reg
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, betas=(0.9, 0.999), weight_decay=self.cfg.l2_reg,
        )

        self.reset_critic_optimizer = torch.optim.Adam(
            self.reset_critic.parameters(), lr=self.critic_lr, betas=(0.9, 0.999), weight_decay=self.cfg.l2_reg,
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=(0.5, 0.999), weight_decay=self.cfg.l2_reg,
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_critic(self, obs, action, reward, next_obs, done):
        nonreset_sample_indices = action[:,-1] < self.cfg.reset_thresh
        with torch.no_grad():
            _, x_action, log_p, _, _, reset_action, log_reset_action_p, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs[nonreset_sample_indices,:-1], x_action[nonreset_sample_indices])
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_p[nonreset_sample_indices]
            target_Q = reward[nonreset_sample_indices] + ((1.0 - done[nonreset_sample_indices]) * self.cfg.gamma * target_V)

            target_reset_Q1, target_reset_Q2 = self.reset_critic_target(next_obs, reset_action)
            target_reset_V = torch.min(target_reset_Q1, target_reset_Q2) - self.alpha.detach() * log_reset_action_p
            target_reset_Q = reward + ((1.0 - done) * self.cfg.gamma * target_reset_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs[nonreset_sample_indices,:-1], action[nonreset_sample_indices,:-1])
        critic_loss = torch.mean(
            (current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2
        )

        current_reset_Q1, current_reset_Q2 = self.reset_critic(obs, action[:,-1:])
        reset_critic_loss = torch.mean((current_reset_Q1 - target_reset_Q)**2 + (current_reset_Q2 - target_reset_Q)**2)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # optimize the reset_critic
        self.reset_critic_optimizer.zero_grad()
        reset_critic_loss.backward()
        self.reset_critic_optimizer.step()

        critic_stats = {
            'train_critic/loss': critic_loss.item(),
            'train_critic/reset_critic_loss': reset_critic_loss.item(),
        }

        return critic_stats

    def update_actor_and_alpha(self, obs):
        _, x_action, log_p, log_std, _, reset_action, log_reset_action_p, _ = self.actor(obs)
        action_Q1, action_Q2 = self.critic(obs[:,:-1], x_action)
        reset_action_Q1, reset_action_Q2 = self.reset_critic(obs, reset_action)


        action_Q = torch.min(action_Q1, action_Q2)
        reset_action_Q = torch.min(reset_action_Q1, reset_action_Q2)

        action_loss = (self.alpha.detach() * log_p - action_Q).mean()
        reset_action_loss = (self.alpha.detach() * log_reset_action_p - reset_action_Q).mean()


        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

        # optimize the actor
        self.action_module_optimizer.zero_grad()
        action_loss.backward()
        self.action_module_optimizer.step()

        self.reset_action_module_optimizer.zero_grad()
        reset_action_loss.backward()
        self.reset_action_module_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_p - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        actor_stats = {
            'train_actor/action_loss': action_loss.item(),
            'train_actor/reset_action_loss': reset_action_loss.item(),
            'train_actor/target_entropy': self.target_entropy.item(),
            'train_actor/entropy': entropy.mean().item(),
            'train_alpha/loss': alpha_loss.item(),
            'train_alpha/value': self.alpha.item(),
            'train/entropy': entropy.mean().item(),
        }
        return actor_stats

    def update(self, obs, action, reward, done, next_obs):
        # Move tensors to device
        obs, action, reward, next_obs, done = obs.to(self.device), action.to(self.device), \
            reward.to(self.device), next_obs.to(self.device), done.to(self.device)

        # print(obs.shape, action.shape, reward.shape, next_obs.shape, done.shape)
        # regular update of SAC_RAD, sequentially augment data and train
        stats = self.update_critic(obs, action, reward, next_obs, done)
        if self.num_updates % self.actor_update_freq == 0:
            actor_stats = self.update_actor_and_alpha(obs)
            stats = {**stats, **actor_stats}
        if self.num_updates % self.critic_target_update_freq == 0:
            self.soft_update_target()
        stats['train/batch_reward'] = reward.mean().item()
        stats['train/num_updates'] = self.num_updates
        self.num_updates += 1
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
        self.soft_update_params(
            self.reset_critic.Q1, self.reset_critic_target.Q1, self.critic_tau
        )
        self.soft_update_params(
            self.reset_critic.Q2, self.reset_critic_target.Q2, self.critic_tau
        )

    def sample_action(self, x, deterministic=False):
        if self.steps < self.cfg.init_steps:
            return np.random.uniform(-1, 1, size=self.cfg.action_dim+1)
            
        with torch.no_grad():
            if not isinstance(x, torch.FloatTensor):
                x = torch.FloatTensor(x).to(self.device)
                x = x.unsqueeze(0)
            mu, x_action, _, _, reset_mu, reset_action, _, _ = self.actor(x)
            mu_ext = torch.cat([mu, reset_mu], dim=-1)
            action = torch.cat([x_action, reset_action], dim=-1)
            if deterministic:
                return mu_ext.cpu().data.numpy().flatten()
            else:
                return action.cpu().data.numpy().flatten()
                
    def save(self):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (self.cfg.model_dir, self.steps)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (self.cfg.model_dir, self.steps)
        )
        torch.save(
            self.reset_critic.state_dict(), '%s/reset_critic_%s.pt' % (self.cfg.model_dir, self.steps)
        )

class SACAgent(SAC):
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

class SAC_ResetActionAgent(SAC_ResetAction):
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

class ResetSACAgent(SACAgent):
    def __init__(self, cfg, buffer, device=torch.device('cpu')):
        reset_cfg = deepcopy(cfg)
        reset_cfg.action_dim += 1
        super().__init__(reset_cfg, buffer, device)
