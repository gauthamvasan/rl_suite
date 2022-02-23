import torch
import time
import numpy as np

from torch import nn
from torch.optim import Adam
from rl_suite.algo.cnn_policies import ActorModel, CriticModel

class PPO_RAD:
    def __init__(self, cfg, buffer, device=torch.device('cpu')):
        self.cfg = cfg
        self.actor = ActorModel(cfg.image_shape, cfg.proprioception_shape, cfg.action_shape[0], cfg.net_params,
                                cfg.rad_offset, cfg.freeze_cnn).to(device)

        self.critic = CriticModel(cfg.image_shape, cfg.proprioception_shape, cfg.net_params,
                                  cfg.rad_offset, cfg.freeze_cnn).to(device)

        if hasattr(self.actor.encoder, 'convs'):
            # TODO: Max this a configurable parameter
            self.actor.encoder.convs = self.critic.encoder.convs
            print("Sharing CNN weights between Actor and Critic")

        self.buffer = buffer
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.bootstrap_terminal = cfg.bootstrap_terminal
        self.device = device
        self.n_updates = 0
        self.lmbda = self.cfg.lmbda

        self.actor_opt = Adam(self.actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.l2_reg)
        self.critic_loss = nn.MSELoss()
        self.critic_opt = Adam(self.critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.l2_reg)

        self.train()

    def sample_action(self, img, prop):
        img = img.to(self.device)
        prop = prop.to(self.device)

        with torch.no_grad():
            mu, action, lprob = self.actor(img, prop, random_rad=False, detach_encoder=False)

        return action.cpu().view(-1), lprob.cpu().view(-1)

    def estimate_returns_advantages(self, rewards, dones, vals):
        """ len(rewards) = len(dones) = len(vals)-1

        Args:
            rewards:
            dones:
            vals:

        Returns:

        """
        advs = torch.as_tensor(np.zeros(len(vals), dtype=np.float32), device=self.device)

        for t in reversed(range(len(rewards)-1)):
            if self.bootstrap_terminal:
                delta = rewards[t] + self.gamma * vals[t+1] - vals[t]
                advs[t] = delta + self.gamma * self.lmbda * advs[t+1]
            else:
                delta = rewards[t] + (1 - dones[t]) * self.gamma * vals[t + 1] - vals[t]
                advs[t] = delta + (1 - dones[t]) * self.gamma * self.lmbda * advs[t + 1]

        rets = advs[:-1] + vals[:-1]
        return rets, advs[:-1]

    def update(self):
        images, propris, actions, rewards, dones, old_lprobs = self.buffer.sample(len(self.buffer))
        images, propris, actions, rewards, old_lprobs, dones = images.to(self.device), propris.to(self.device), \
                                                             actions.to(self.device), rewards.to(self.device), \
                                                             old_lprobs.to(self.device), dones.to(self.device)
        with torch.no_grad():
            vals = self.critic(images=images, proprioceptions=propris, random_rad=True, detach_encoder=False)
        rets, advs = self.estimate_returns_advantages(rewards=rewards, dones=dones, vals=vals)

        # Normalize advantages
        norm_advs = (advs - advs.mean()) / advs.std()

        inds = np.arange(len(rewards)-1)
        for itr in range(self.cfg.n_epochs):
            np.random.shuffle(inds)
            for i_start in range(0, len(self.buffer), self.cfg.opt_batch_size):
                opt_inds = inds[i_start: min(i_start+self.cfg.opt_batch_size, len(inds)-1)]
                # Policy update preparation
                new_lprobs = self.actor.lprob(images[opt_inds], propris[opt_inds], actions[opt_inds],
                                              random_rad=True, detach_encoder=False)
                new_vals = self.critic(images[opt_inds], propris[opt_inds], random_rad=True, detach_encoder=False)
                ratio = torch.exp(new_lprobs - old_lprobs[opt_inds])
                p_loss = ratio * norm_advs[opt_inds]
                clipped_p_loss = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * norm_advs[opt_inds]
                actor_loss = -(torch.min(p_loss, clipped_p_loss)).mean()
                critic_loss = self.critic_loss(new_vals, rets[opt_inds])
                loss = actor_loss + critic_loss

                # Apply gradients
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                self.actor_opt.step()
                self.critic_opt.step()

    def push_and_update(self, images, proprioception, action, reward, log_prob, done):
        self.buffer.push(images, proprioception, action, reward, done, lprob=log_prob)
        if len(self.buffer) >= self.batch_size and done:
            tic = time.time()
            self.update()
            self.buffer.reset()
            self.n_updates += 1
            print("Update {} took {}s".format(self.n_updates, time.time()-tic))

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)