from matplotlib.pyplot import axes, axis
import torch

import numpy as np

from rl_suite.algo.sac_reset_action import ResetSACAgent
from rl_suite.algo.cnn_policies import SSEncoderModel
from copy import deepcopy

class ResetSACRADAgent(ResetSACAgent):
    def __init__(self, cfg, buffer, device=torch.device('cpu')):
        self.encoder_tau = cfg.encoder_tau
        self.rad_offset = cfg.rad_offset
        self.encoder = SSEncoderModel(cfg.image_shape, (0,), cfg.encoder_nn_params,
                                        cfg.rad_offset).to(device) # only used for image encoding
        
        if cfg.load_step > -1:
            self.encoder.load_state_dict(
                torch.load('%s/encoder_%s.pt' % (cfg.model_dir, cfg.load_step))
            )
        
        self.encoder_target = deepcopy(self.encoder)
        self.encoder_target.train()
        cfg.obs_dim += self.encoder.latent_dim

        super().__init__(cfg, buffer, device)
    
    def train(self, training=True):
        super().train(training)
        self.encoder.train(training)

    def share_memory(self):
        super().share_memory()
        self.encoder.share_memory()
        self.encoder_target.share_memory()

    def init_optimizers(self):
        super().init_optimizers()
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters())+list(self.encoder.parameters()), lr=self.critic_lr, betas=(0.9, 0.999), weight_decay=self.cfg.l2_reg,
        )

    def update_critic(self, img, obs, action, reward, next_img, next_obs, done):
        nonreset_sample_indices = action[:,-1] < self.cfg.reset_thresh
        with torch.no_grad():
            encoded_next_img = self.encoder(next_img, None)
            next_latent = torch.cat([encoded_next_img, next_obs], dim=-1)
            _, policy_action, log_p, _, _, reset_action, log_reset_action_p, _ = self.actor(next_latent)

            encoded_next_img_target = self.encoder_target(next_img, None)
            next_latent_target = torch.cat([encoded_next_img_target, next_obs], dim=-1)
            target_Q1, target_Q2 = self.critic_target(next_latent_target[nonreset_sample_indices,:-1], policy_action[nonreset_sample_indices])
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_p[nonreset_sample_indices]
            target_Q = reward[nonreset_sample_indices] + ((1.0 - done[nonreset_sample_indices]) * self.cfg.gamma * target_V)

            target_reset_Q1, target_reset_Q2 = self.reset_critic_target(next_latent_target, reset_action)
            target_reset_V = torch.min(target_reset_Q1, target_reset_Q2) - self.alpha.detach() * log_reset_action_p
            target_reset_Q = reward + ((1.0 - done) * self.cfg.gamma * target_reset_V)
        
        # get current Q estimates
        encoded_current_img = self.encoder(img, None) # only update the encoder via critic update
        current_latent = torch.cat([encoded_current_img, obs], dim=-1)
        current_Q1, current_Q2 = self.critic(current_latent[nonreset_sample_indices,:-1], action[nonreset_sample_indices,:-1])
        critic_loss = torch.mean(
            (current_Q1 - target_Q)**2 + (current_Q2 - target_Q)**2
        )

        current_reset_Q1, current_reset_Q2 = self.reset_critic(current_latent.detach(), action[:,-1:])
        reset_critic_loss = torch.mean(
            (current_reset_Q1 - target_reset_Q)**2 + (current_reset_Q2 - target_reset_Q)**2
        )

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

    def update_actor_and_alpha(self, img, obs):
        with torch.no_grad():
            encoded_img = self.encoder(img, None)
            latent_no_grad = torch.cat([encoded_img, obs], dim=-1)

        _, action, log_p, log_std, _, reset_action, log_reset_action_p, _ = self.actor(latent_no_grad)
        action_Q1, action_Q2 = self.critic(latent_no_grad[:,:-1], action)
        reset_action_Q1, reset_action_Q2 = self.reset_critic(latent_no_grad, reset_action)


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

    def update(self, img, obs, action, reward, done, next_img, next_obs):
        # Move tensors to device
        img, obs, action, reward, next_img, next_obs, done = img.to(self.device), obs.to(self.device), action.to(self.device), \
            reward.to(self.device), next_img.to(self.device), next_obs.to(self.device), done.to(self.device)

        # print(obs.shape, action.shape, reward.shape, next_obs.shape, done.shape)
        # regular update of SAC_RAD, sequentially augment data and train
        stats = self.update_critic(img, obs, action, reward, next_img, next_obs, done)
        if self.num_updates % self.actor_update_freq == 0:
            actor_stats = self.update_actor_and_alpha(img, obs)
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
        super().soft_update_target()
        self.soft_update_params(
            self.encoder, self.encoder_target, self.encoder_tau
        )
        
    def save(self):
        super().save()
        torch.save(
            self.encoder.state_dict(), '%s/encoder_%s.pt' % (self.cfg.model_dir, self.steps)
        )

    def sample_action(self, img, x, deterministic=False):
        if self.steps < self.cfg.init_steps:
            return np.random.uniform(-1, 1, size=self.cfg.action_dim+1)
            
        with torch.no_grad():
            if not isinstance(x, torch.FloatTensor):
                x = torch.FloatTensor(x).to(self.device).unsqueeze(0)
            
            if not isinstance(img, torch.FloatTensor):
                img = torch.FloatTensor(img).to(self.device).unsqueeze(0)

            encoded_img = self.encoder(img, None)
            mu, x_action, _, _, reset_mu, reset_action, _, _ = self.actor(torch.cat([encoded_img, x], dim=-1))
            mu_ext = torch.cat([mu, reset_mu], dim=-1)
            action = torch.cat([x_action, reset_action], dim=-1)
            if deterministic:
                return mu_ext.cpu().data.numpy().flatten()
            else:
                return action.cpu().data.numpy().flatten()

    def push_and_update(self, img, obs, action, reward, done):
        self._replay_buffer.add(img, obs, action, reward, done)
        
        if self.steps > self.cfg.init_steps and (self.steps % self.cfg.update_every == 0):
            for _ in range(self.cfg.update_epochs):
                # tic = time.time()
                stat = self.update(*self._replay_buffer.sample())
                # print(time.time() - tic)
            return stat
        
        self.steps += 1