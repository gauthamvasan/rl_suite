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
        if not hasattr(self.cfg, 'gpu_chunk'):
            self.GPU_CHUNK = 256

        self.actor_opt = Adam(self.actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.l2_reg)
        self.critic_loss = nn.MSELoss()
        self.critic_opt = Adam(self.critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.l2_reg)

        self.train()

    @staticmethod
    def get_as_tensor(img, prop):
        if isinstance(img, (np.ndarray, np.generic)):
            img = torch.as_tensor(img.astype(np.float32))[None, :, :, :]
        
        if isinstance(prop, (np.ndarray, np.generic)):
            prop = torch.as_tensor(prop.astype(np.float32))[None, :]
        
        return img, prop


    def sample_action(self, img, prop):
        img, prop = self.get_as_tensor(img, prop)
        img = img.to(self.device)
        prop = prop.to(self.device)

        with torch.no_grad():
            mu, action, lprob = self.actor(img, prop, random_rad=False, detach_encoder=True)

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

        for t in reversed(range(len(rewards))):
            if self.bootstrap_terminal:
                delta = rewards[t] + self.gamma * vals[t+1] - vals[t]
                advs[t] = delta + self.gamma * self.lmbda * advs[t+1]
            else:
                delta = rewards[t] + (1 - dones[t]) * self.gamma * vals[t + 1] - vals[t]
                advs[t] = delta + (1 - dones[t]) * self.gamma * self.lmbda * advs[t + 1]

        rets = advs[:-1] + vals[:-1]
        return rets, advs[:-1]

    def update(self, next_imgs, next_propris):
        images, propris, actions, rewards, dones, old_lprobs = self.buffer.sample(len(self.buffer))
        next_imgs, next_propris = self.get_as_tensor(next_imgs, next_propris)
        images = torch.cat([images, next_imgs])
        propris = torch.cat([propris, next_propris])
        vals = []
        with torch.no_grad():
            end = len(images)
            for ind in range(0, end, self.GPU_CHUNK):
                inds = np.arange(ind, min(end, ind + self.GPU_CHUNK))
                img = images[inds].to(self.device)
                prop = propris[inds].to(self.device)
                v = self.critic(images=img, proprioceptions=prop, random_rad=True, detach_encoder=False)
                vals.append(v)

        vals = torch.cat(vals)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        old_lprobs = old_lprobs.to(self.device)
        rets, advs = self.estimate_returns_advantages(rewards=rewards, dones=dones, vals=vals)

        # Normalize advantages
        norm_advs = (advs - advs.mean()) / advs.std()

        inds = np.arange(len(rewards))
        for itr in range(self.cfg.n_epochs):
            np.random.shuffle(inds)
            for i_start in range(0, len(self.buffer), self.cfg.opt_batch_size):
                opt_inds = inds[i_start: min(i_start+self.cfg.opt_batch_size, len(inds)-1)]
                img = images[opt_inds].to(self.device)
                prop = propris[opt_inds].to(self.device)
                a = actions[opt_inds].to(self.device)

                # Policy update preparation
                new_lprobs = self.actor.lprob(img, prop, a, random_rad=True, detach_encoder=True)
                new_vals = self.critic(img, prop, random_rad=True, detach_encoder=False)
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

    def push_and_update(self, imgs, propris, action, reward, log_prob, done, next_imgs, next_propris):
        self.buffer.push(imgs, propris, action, reward, done, lprob=log_prob)
        if len(self.buffer) >= self.batch_size and done:
            tic = time.time()
            self.update(next_imgs, next_propris)
            self.buffer.reset()
            self.n_updates += 1
            print("Update {} took {}s".format(self.n_updates, time.time()-tic))

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
