import torch
import copy
import time
import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from rl_suite.algo.cnn_policies import SACRADActor, SACRADCritic


class SAC_RAD:
    """ SAC algorithm. """
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = device
        self.gamma = cfg.gamma
        self.critic_tau = cfg.critic_tau
        self.encoder_tau = cfg.encoder_tau
        self.actor_update_freq = cfg.actor_update_freq
        self.critic_target_update_freq = cfg.critic_target_update_freq
        self.rad_offset = cfg.rad_offset

        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.alpha_lr = cfg.alpha_lr

        self.action_dim = cfg.action_shape[0]

        self.actor = SACRADActor(cfg.image_shape, cfg.proprioception_shape, cfg.action_shape[0], cfg.net_params,
                                cfg.rad_offset, cfg.freeze_cnn).to(device)

        self.critic = SACRADCritic(cfg.image_shape, cfg.proprioception_shape, cfg.action_shape[0], cfg.net_params,
                                cfg.rad_offset, cfg.freeze_cnn).to(device)

        self.critic_target = copy.deepcopy(self.critic) # also copies the encoder instance

        if hasattr(self.actor.encoder, 'convs'):
            self.actor.encoder.convs = self.critic.encoder.convs
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
            self.actor.parameters(), lr=self.actor_lr, betas=(0.9, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, betas=(0.9, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=(0.5, 0.999)
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def sample_action(self, img, prop, deterministic=False):
        with torch.no_grad():
            if img is not None:
                img = torch.FloatTensor(img).to(self.device)
                img = img.unsqueeze(0)
            if prop is not None:
                prop = torch.FloatTensor(prop).to(self.device)
                prop = prop.unsqueeze(0)
            mu, action, _, _ = self.actor(img, prop)
            if deterministic:
                return mu.cpu().data.numpy().flatten()
            else:
                return action.cpu().data.numpy().flatten()

    def update_critic(self, img, prop, action, reward, next_img, next_prop, not_done):
        with torch.no_grad():
            _, policy_action, log_p, _ = self.actor(next_img, next_prop)
            target_Q1, target_Q2 = self.critic_target(next_img, next_prop, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_p
            target_Q = reward + (not_done * self.gamma * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(img, prop, action, detach_encoder=False)

        # Ignore terminal transitions to enable infinite bootstrap
        critic_loss = torch.mean(
            (current_Q1 - target_Q) ** 2 * not_done + (current_Q2 - target_Q) ** 2 * not_done
            #(current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2
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

    def update_actor_and_alpha(self, img, prop):
        # detach encoder, so we don't update it with the actor loss
        _, action, log_p, log_std = self.actor(img, prop, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(img, prop, action, detach_encoder=True)

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

    def update(self, img, prop, action, reward, next_img, next_prop, not_done):
        # Move tensors to device
        img, prop, action, reward, next_img, next_prop, not_done = img.to(self.device), prop.to(self.device), \
            action.to(self.device), reward.to(self.device), next_img.to(self.device), \
                next_prop.to(self.device), not_done.to(self.device)

        # regular update of SAC_RAD, sequentially augment data and train
        stats = self.update_critic(img, prop, action, reward, next_img, next_prop, not_done)
        if self.num_updates % self.actor_update_freq == 0:
            actor_stats = self.update_actor_and_alpha(img, prop)
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
            self.critic.encoder, self.critic_target.encoder,
            self.encoder_tau
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


class SACRADAgent(SAC_RAD):
    def __init__(self, cfg, buffer, device=torch.device('cpu')):
        super().__init__(cfg, device)
        self._replay_buffer = buffer
        self.steps = 0

    def push_and_update(self, image, propri, action, reward, done):
        self._replay_buffer.add(image, propri, action, reward, done)
        
        if self.steps > self.cfg.init_steps and (self.steps % self.cfg.update_every == 0):
            for _ in range(self.cfg.update_epochs):
                # tic = time.time()
                stat = self.update(*self._replay_buffer.sample())
                # print(time.time() - tic)
            return stat
        
        self.steps += 1
