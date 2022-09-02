import torch
import numpy as np

from copy import deepcopy
from rl_suite.algo.mlp_policies import SquashedGaussianMLPActor, LinearSquashedPolicy, SACCritic
from rl_suite.algo.sac import SAC
from rl_suite.algo.sac_rad import SACRADAgent
from rl_suite.algo.cnn_policies import SACRADActor, SACRADCritic


class ResetSACAgent(SAC):
    """ SAC with Automatic Entropy Adjustment. """
    def __init__(self, cfg, buffer, device=torch.device('cpu')):
        self.cfg = cfg
        self._replay_buffer = buffer
        self.device = device
        self.gamma = cfg.gamma
        self.critic_tau = cfg.critic_tau
        self.encoder_tau = cfg.encoder_tau
        self.actor_update_freq = cfg.actor_update_freq
        self.critic_target_update_freq = cfg.critic_target_update_freq

        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.alpha_lr = cfg.alpha_lr
        self.reset_thresh = cfg.reset_thresh

        # Regular policy: Exclude time from obs.
        ## Assumption: Time is the last value in the observation vector
        self.actor = SquashedGaussianMLPActor(cfg.obs_dim-1, cfg.action_dim, cfg.actor_nn_params, device)
        self.critic = SACCritic(cfg.obs_dim-1, cfg.action_dim, cfg.critic_nn_params, device)
        
        # Reset policy     
        self.reset_actor = SquashedGaussianMLPActor(cfg.obs_dim, 1, cfg.actor_nn_params, device)
        self.reset_critic = SACCritic(cfg.obs_dim, 1, cfg.critic_nn_params, device)
        self.reset_penalty = -cfg.reset_length * cfg.reset_time

        # Critic targets       
        self.critic_target = deepcopy(self.critic) # also copies the encoder instance
        self.reset_critic_target = deepcopy(self.reset_critic) # also copies the encoder instance

        # Log alpha
        self.log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.reset_log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(device)
        self.reset_log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(cfg.action_shape)

        self.num_updates = 0
        self.steps = 0

        # optimizers
        self.init_optimizers()
        self.train()
        self.critic_target.train()
    
    def init_optimizers(self):
        super(ResetSACAgent, self).init_optimizers()

        self.reset_actor_optimizer = torch.optim.Adam(
            self.reset_actor.parameters(), lr=self.actor_lr, betas=(0.9, 0.999), weight_decay=self.cfg.l2_reg,
        )

        self.reset_critic_optimizer = torch.optim.Adam(
            self.reset_critic.parameters(), lr=self.critic_lr, betas=(0.9, 0.999), weight_decay=self.cfg.l2_reg,
        )

        self.reset_log_alpha_optimizer = torch.optim.Adam(
            [self.reset_log_alpha], lr=self.alpha_lr, betas=(0.5, 0.999), weight_decay=self.cfg.l2_reg,
        )
    
    def train(self, training=True):
        super(ResetSACAgent, self).train(training=training)
        self.reset_actor.train(training)
        self.reset_critic.train(training)

    def share_memory(self):
        super(ResetSACAgent, self).share_memory()
        self.reset_actor.share_memory()
        self.reset_critic.share_memory()

    def update_reset_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            _, policy_action, log_p, _ = self.reset_actor(next_obs)
            target_Q1, target_Q2 = self.reset_critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.reset_alpha.detach() * log_p
            if self.cfg.bootstrap_terminal:
                # enable infinite bootstrap
                target_Q = reward + (self.cfg.gamma * target_V)
            else:
                target_Q = reward + ((1.0 - done) * self.cfg.gamma * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.reset_critic(obs, action)
        critic_loss = torch.mean(
            (current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2
        )
        # Optimize the critic
        self.reset_critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.reset_critic_optimizer.step()

        critic_stats = {
            'train_critic/loss': critic_loss.item()
        }

        return critic_stats

    def update_reset_actor_and_alpha(self, obs):
        # detach encoder, so we don't update it with the actor loss
        _, action, log_p, log_std = self.reset_actor(obs)
        actor_Q1, actor_Q2 = self.reset_critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.reset_alpha.detach() * log_p - actor_Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

        # optimize the actor
        self.reset_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.reset_actor_optimizer.step()

        self.reset_log_alpha_optimizer.zero_grad()
        alpha_loss = (self.reset_alpha * (-log_p - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.reset_log_alpha_optimizer.step()

        actor_stats = {
            'train_actor/loss': actor_loss.item(),
            'train_actor/target_entropy': self.target_entropy.item(),
            'train_actor/entropy': entropy.mean().item(),
            'train_alpha/loss': alpha_loss.item(),
            'train_alpha/value': self.alpha.item(),
            'train/entropy': entropy.mean().item(),
        }
        return actor_stats
    
    def sample_action(self, x, deterministic=False):
        with torch.no_grad():
            if not isinstance(x, torch.FloatTensor):
                x = torch.FloatTensor(x).to(self.device)
                x = x.unsqueeze(0)
            reg_mu, reg_action, _, _ = self.actor(x[:, :-1])
            reset_mu, reset_action, _, _ = self.reset_actor(x)
            
            if deterministic:
                return (reg_mu.cpu().data.numpy().flatten(), reset_mu.cpu().data.numpy().flatten())
            
            return (reg_action.cpu().data.numpy().flatten(), reset_action.cpu().data.numpy().flatten())

    def push_and_update(self, obs, action, reward, done, reset_action):
        self._replay_buffer.add(obs, action, reward, done, reset_action)
        
        if self.steps > self.cfg.init_steps and (self.steps % self.cfg.update_every == 0):
            for _ in range(self.cfg.update_epochs):
                # tic = time.time()
                obs, action, reward, done, next_obs = self._replay_buffer.sample()                
                stat = self.update(obs[:, :-1], action, reward, done, next_obs[:, :-1])
                reset_stat = self.reset_update(*self._replay_buffer.reset_action_sample())
                # print(time.time() - tic)
            return {'stat': stat, 'reset_stat': reset_stat}
        
        self.steps += 1
    
    def reset_update(self, obs, action, reward, done, next_obs):
        # Move tensors to device
        obs, action, reward, next_obs, done = obs.to(self.device), action.to(self.device), \
            reward.to(self.device), next_obs.to(self.device), done.to(self.device)

        stats = self.update_reset_critic(obs, action, reward, next_obs, done)
        if self.num_updates % self.actor_update_freq == 0:
            actor_stats = self.update_reset_actor_and_alpha(obs)
            stats = {**stats, **actor_stats}
        if self.num_updates % self.critic_target_update_freq == 0:
            self.soft_update_reset_target()
        stats['train/batch_reward'] = reward.mean().item()
        stats['train/num_updates'] = self.num_updates
        self.num_updates += 1
        return stats

    def soft_update_reset_target(self):
        self.soft_update_params(
            self.reset_critic.Q1, self.reset_critic_target.Q1, self.critic_tau
        )
        self.soft_update_params(
            self.reset_critic.Q2, self.reset_critic_target.Q2, self.critic_tau
        )

    @property
    def reset_alpha(self):
        return self.reset_log_alpha.exp()


class ResetSACRADAgent(SACRADAgent):
    def __init__(self, cfg, buffer, device=torch.device('cpu')):
        reset_cfg = deepcopy(cfg)
        reset_cfg.action_dim += 1
        super(ResetSACRADAgent, self).__init__(reset_cfg, buffer, device)
