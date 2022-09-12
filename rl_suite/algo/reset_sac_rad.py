import torch
import numpy as np

from copy import deepcopy
from rl_suite.algo.cnn_policies import SAC_RAD_ResetActionActor, SACRADCritic


class SAC_RAD_ResetAction:
    """SAC algorithm."""
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = device
        self.gamma = cfg.gamma
        self.critic_tau = cfg.critic_tau
        self.encoder_tau = cfg.encoder_tau
        self.actor_update_freq = cfg.actor_update_freq
        self.critic_target_update_freq = cfg.critic_target_update_freq
        self.rad_offset = cfg.rad_offset
        self.reset_thresh = cfg.reset_thresh

        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.alpha_lr = cfg.alpha_lr

        prop_t_shape = (cfg.proprioception_shape[0]+1,)

        self.actor = SAC_RAD_ResetActionActor(cfg.image_shape, prop_t_shape, cfg.action_shape[0]+1, cfg.net_params, cfg.rad_offset).to(device)

        self.critic = SACRADCritic(cfg.image_shape, cfg.proprioception_shape, cfg.action_shape[0], cfg.net_params, cfg.rad_offset).to(device)

        self.reset_critic = SACRADCritic(cfg.image_shape, prop_t_shape, 1, cfg.net_params, cfg.rad_offset).to(device)

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

        if hasattr(self.actor.encoder, 'convs'):
            self.actor.encoder.convs = self.critic.encoder.convs
            self.reset_critic.encoder.convs = self.critic.encoder.convs

        self.critic_target = deepcopy(self.critic) # also copies the encoder instance
        self.reset_critic_target = deepcopy(self.reset_critic) # also copies the encoder instance

        if hasattr(self.actor.encoder, 'convs'):
            self.reset_critic_target.encoder.convs = self.critic_target.encoder.convs

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

        self.action_reset_shape = (cfg.action_shape[0]+1,)
        
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
            self.actor.get_action_module_parameters(), lr=self.actor_lr, betas=(0.9, 0.999), weight_decay=self.cfg.l2_reg,
        )

        self.reset_action_module_optimizer = torch.optim.Adam(
            self.actor.get_reset_action_module_parameters(), lr=self.actor_lr, betas=(0.9, 0.999), weight_decay=self.cfg.l2_reg,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, betas=(0.9, 0.999), weight_decay=self.cfg.l2_reg,
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=(0.5, 0.999), weight_decay=self.cfg.l2_reg,
        )

        self.reset_critic_optimizer = torch.optim.Adam(
            self.reset_critic.parameters(), lr=self.critic_lr, betas=(0.9, 0.999), weight_decay=self.cfg.l2_reg,
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def sample_action(self, image, state, step, deterministic=False):
        if step < self.cfg.init_steps:
            return np.random.uniform(-1, 1, self.action_reset_shape)

        with torch.no_grad():
            if image is not None:
                image = torch.FloatTensor(image).to(self.device)
                image.unsqueeze_(0)

            if state is not None:
                state = torch.FloatTensor(state).to(self.device)
                state.unsqueeze_(0)

            mu, pi, _, _, reset_mu, reset_action, _, _ = self.actor(
                image, state, random_rad=False, compute_log_pi=False, 
            )

            mu = torch.cat([mu, reset_mu], dim=-1)
            pi = torch.cat([pi, reset_action], dim=-1)
            if deterministic:
                return mu.cpu().data.numpy().flatten()
            else:
                return pi.cpu().data.numpy().flatten()

    def update_critic(self, images, states, actions, rewards, next_images, next_states, dones):
        nonreset_sample_indices = actions[:,-1] < self.reset_thresh
        nonreset_images = images[nonreset_sample_indices]
        nonreset_next_images = next_images[nonreset_sample_indices]
        with torch.no_grad():
            _, x_actions, log_pis, _, _, reset_actions, log_reset_action_probs, _ = self.actor(next_images, next_states)
            
            target_Q1, target_Q2 = self.critic_target(nonreset_next_images, next_states[nonreset_sample_indices,:-1], x_actions[nonreset_sample_indices])

            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pis[nonreset_sample_indices]

            target_Q = rewards[nonreset_sample_indices] + ((1.0 - dones[nonreset_sample_indices]) * self.cfg.gamma * target_V)

            target_reset_Q1, target_reset_Q2 = self.reset_critic_target(next_images, next_states, reset_actions)
            target_reset_V = torch.min(target_reset_Q1, target_reset_Q2) - self.alpha.detach() * log_reset_action_probs
            target_reset_Q = rewards + ((1.0 - dones) * self.cfg.gamma * target_reset_V)

        # get the current Q estimates
        current_Q1, current_Q2 = self.critic(nonreset_images, states[nonreset_sample_indices,:-1], actions[nonreset_sample_indices,:-1], detach_encoder=False)

        critic_loss = torch.mean((current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2)

        current_reset_Q1, current_reset_Q2 = self.reset_critic(images, states, actions[:,-1:], detach_encoder=True)

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
            'train_critic/critic_loss': critic_loss.item(),
            'train_critic/reset_critic_loss': reset_critic_loss.item(),
        }

        return critic_stats

    def update_actor_and_alpha(self, images, states):
        
        # detach encoder, so we don't update it with the actor loss
        _, x_actions, log_pis, log_stds, _, reset_actions, log_reset_action_probs, reset_log_stds = self.actor(images, states ,detach_encoder=True)
        action_Q1, action_Q2 = self.critic(images, states[:,:-1], x_actions, detach_encoder=True)
        reset_action_Q1, reset_action_Q2 = self.reset_critic(images, states, reset_actions, detach_encoder=True)

        action_Q = torch.min(action_Q1, action_Q2)
        reset_action_Q = torch.min(reset_action_Q1, reset_action_Q2)
        action_loss = (self.alpha.detach() * log_pis - action_Q).mean()
        reset_action_loss = (self.alpha.detach() * log_reset_action_probs - reset_action_Q).mean()
        entropy = 0.5 * log_stds.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_stds.sum(dim=-1)

        # optimize the actor
        self.action_module_optimizer.zero_grad()
        action_loss.backward()
        self.action_module_optimizer.step()

        self.reset_action_module_optimizer.zero_grad()
        reset_action_loss.backward()
        self.reset_action_module_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pis - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        actor_stats = {
            'train_actor/log_pis': log_pis.mean().item(),
            'train_actor/action_loss': action_loss.item(),
            'train_actor/reset_action_loss': reset_action_loss.item(),
            'train_actor/log_std': log_stds.mean().item(),
            'train_alpha/loss': alpha_loss.item(),
            'train_alpha/value': self.alpha.item(),
            'train_alpha/entropy': entropy.mean().item()
        }
        return actor_stats

    def update(self, images, states, actions, rewards, dones, next_images, next_states):
        
        images = torch.as_tensor(images, device=self.device).float()
        next_images = torch.as_tensor(next_images, device=self.device).float()
        
        states = torch.as_tensor(states, device=self.device).float()
        next_states = torch.as_tensor(next_states, device=self.device).float()

        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)
        
        stats = self.update_critic(images, states, actions, rewards, next_images, next_states, dones)
        if self.num_updates % self.actor_update_freq == 0:
            actor_stats = self.update_actor_and_alpha(images, states)
            stats = {**stats, **actor_stats}
        if self.num_updates % self.critic_target_update_freq == 0:
            self.soft_update_target()
        stats['train/batch_reward'] = rewards.mean().item()
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
        self.soft_update_params(
            self.critic.encoder, self.critic_target.encoder,
            self.encoder_tau
        )
        # self.soft_update_params(
        #     self.reset_critic.encoder, self.reset_critic_target.encoder,
        #     self.encoder_tau
        # )

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
        
    def close(self):
        del self


class SAC_RAD_ResetActionAgent(SAC_RAD_ResetAction):
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
