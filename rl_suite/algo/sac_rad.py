import torch
import time
import pickle
import threading

import numpy as np
import torch.multiprocessing as mp

from copy import deepcopy
from rl_suite.algo.cnn_policies import SACRADActor, SACRADCritic
from rl_suite.algo.replay_buffer import SACRADBuffer


class SAC_RAD:
    """ SAC algorithm. """
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = device
        self.gamma = cfg.gamma
        self.critic_tau = cfg.critic_tau
        self.encoder_tau = cfg.encoder_tau
        self.update_actor_every = cfg.update_actor_every
        self.update_critic_target_every = cfg.update_critic_target_every
        self.rad_offset = cfg.rad_offset

        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.alpha_lr = cfg.alpha_lr

        self.action_dim = cfg.action_shape[0]
        
        self.actor = SACRADActor(cfg.image_shape, cfg.proprioception_shape, cfg.action_shape[0], cfg.net_params,
                                cfg.rad_offset, cfg.freeze_cnn).to(device)

        self.critic = SACRADCritic(cfg.image_shape, cfg.proprioception_shape, cfg.action_shape[0], cfg.net_params,
                                cfg.rad_offset, cfg.freeze_cnn).to(device)

        self.critic_target = deepcopy(self.critic) # also copies the encoder instance

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
            mu, action, _, log_std = self.actor(img, prop)
            # print('mu:', mu.cpu().data.numpy().flatten())
            # print('std:', log_std.exp().cpu().data.numpy().flatten())
            if deterministic:
                return mu.cpu().data.numpy().flatten()
            else:
                return action.cpu().data.numpy().flatten()

    def update_critic(self, img, prop, action, reward, next_img, next_prop, done):
        with torch.no_grad():
            _, policy_action, log_p, _ = self.actor(next_img, next_prop)
            target_Q1, target_Q2 = self.critic_target(next_img, next_prop, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_p
            if self.cfg.bootstrap_terminal:
                # enable infinite bootstrap
                target_Q = reward + (self.cfg.gamma * target_V)
            else:
                target_Q = reward + ((1.0 - done) * self.cfg.gamma * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(img, prop, action)
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

    def update(self, img, prop, action, reward, next_img, next_prop, done):
        # Move tensors to device
        img, prop, action, reward, next_img, next_prop, done = img.to(self.device), prop.to(self.device), \
            action.to(self.device), reward.to(self.device), next_img.to(self.device), \
                next_prop.to(self.device), done.to(self.device)

        # regular update of SAC_RAD, sequentially augment data and train
        stats = self.update_critic(img, prop, action, reward, next_img, next_prop, done)
        if self.num_updates % self.update_actor_every == 0:
            actor_stats = self.update_actor_and_alpha(img, prop)
            stats = {**stats, **actor_stats}
        if self.num_updates % self.update_critic_target_every == 0:
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

    def save(self, model_dir, unique_str):
                model_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha.detach().item(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "log_alpha_opt": self.log_alpha_optimizer.state_dict(),
        }
        torch.save(
            model_dict, '%s/%s.pt' % (model_dir, unique_str)
        )

    def load(self, model_dir, unique_str):
        model_dict = torch.load('%s/%s.pt' % (model_dir, unique_str))
        self.actor.load_state_dict(model_dict["actor"])
        self.critic.load_state_dict(model_dict["critic"])
        self.log_alpha = torch.tensor(model_dict["log_alpha"]).to(self.device)
        self.log_alpha.requires_grad = True
        self.actor_optimizer.load_state_dict(model_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(model_dict["critic_optimizer"])
        self.log_alpha_optimizer.load_state_dict(model_dict["log_alpha_optimizer"])


class SACRADAgent(SAC_RAD):
    def __init__(self, cfg, buffer, device=torch.device('cpu')):
        super().__init__(cfg, device)
        self._replay_buffer = buffer
        self.steps = 0

    def push_and_update(self, image, propri, action, reward, done):
        self._replay_buffer.add(image, propri, action, reward, done)
        self.steps += 1

        stat = {}   
        if self.steps > self.cfg.init_steps and (self.steps % self.cfg.update_every == 0):
            for _ in range(self.cfg.update_epochs):
                # tic = time.time()
                stat = self.update(*self._replay_buffer.sample())
                # print(time.time() - tic)
        return stat


class AsyncSACAgent(SAC_RAD):
    def __init__(self, cfg, device=torch.device('cpu')):
        """ SAC agent with asynchronous updates using multiprocessing
        N.B: Pytorch multiprocessing and replay buffers do not mingle well!
        Buffer operations should be taken care of in the main training script
        Args:
            cfg:
            actor_critic:
            device:
        """
        super(AsyncSACAgent, self).__init__(cfg=cfg, device=device)
        self.cfg = cfg
        # self.pi = deepcopy(self.actor)
        # self.pi.to(device)
        self.device = device

        self.batch_size = cfg.batch_size

        self.running = mp.Value('i', 1)
        self.pause = mp.Value('i', 0)
        self.save_buffer = mp.Value('i', 0)
        self.steps = mp.Value('i', 0)
        self.n_updates = mp.Value('i', 0)
        self._nn_lock = mp.Lock()
        self._buffer_lock = mp.Lock()

        self._state_dict = {
            'actor': self.actor.state_dict(), 
            'critic': self.critic.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'log_alpha_opt': self.log_alpha_optimizer.state_dict(),
        }

    def async_recv_model(self, model_queue):
        """ Update the performance element (i.e., sync with new model after gradient updates)
        Args:
            model_queue:
        Returns:
        """
        while True:
            with self.running.get_lock():
                if not self.running.value:
                    print("Exiting async_recv_model thread!")
                    return

            self._state_dict = model_queue.get()
            with self._nn_lock:
                self.actor.load_state_dict(self._state_dict['actor'])

            with self.n_updates.get_lock():
                n_updates = self.n_updates.value
            if n_updates % 20 == 0:
                print("***** Performance element updated!, # learning updates: {} *****".format(n_updates))


    def async_update(self, tensor_queue, model_queue):
        """ Asynchronous process to update the actor_critic model.
        Relies on one other threads spawned from the main process:
            - async_recv_model
        It also spawns it's own thread `async_recv_data` to receive state transitions from the main interaction process
        Args:
            tensor_queue: Multiprocessing queue to transfer RL interaction data
            model_queue: Multiprocessing queue to send updated models back to main process
        Returns:
        """
        # TODO: Make buffer a configurable object
        # N.B: DO NOT pass the buffer as an arg. Python pickling causes large delays and often crashes
        if self.cfg.load_buffer:
            print("Loading buffer for Vector #: {}".format(self.cfg.robot_serial))
            buffer = pickle.load(open("{}-sac_buffer.pkl".format(self.cfg.robot_serial), "rb"))
            print("Buffer loaded successfully from disk...")
        else:
            buffer = SACRADBuffer(self.cfg.image_shape, self.cfg.proprioception_shape, self.cfg.action_shape,
                                self.cfg.replay_buffer_capacity, self.cfg.batch_size)

        def async_recv_data():
            # TODO: Exit mechanism for buffer
            while True:
                with self._buffer_lock:
                    buffer.add(*tensor_queue.get())
                with self.running.get_lock():
                    if not self.running.value:
                        break
            print("Exiting buffer thread within the async update process")

        def save_buffer_pkl():
            tic = time.time()
            print("Saving buffer thread spawned ...")
            with self._buffer_lock:
                with open("{}-sac_buffer.pkl".format(self.cfg.robot_serial), "wb") as handle:
                    pickle.dump(buffer, handle, protocol=4)
            print("Saved the buffer locally!")
            print("Took: {}s".format(time.time()-tic))

        # Start receiving data from the env interaction process to store in the buffer
        buffer_t = threading.Thread(target=async_recv_data)
        buffer_t.start()
        
        while True:
            with self.running.get_lock():
                if not self.running.value:
                    print("Exiting async_update process!")
                    buffer_t.join()
                    return

            # Warmup block
            with self.steps.get_lock():
                steps = self.steps.value
            if steps < self.cfg.init_steps:
                time.sleep(5)
                print("Waiting to fill up the buffer before making any updates...")
                continue
            
            # Save buffer locally
            with self.save_buffer.get_lock():
                save_buffer = self.save_buffer.value
                if save_buffer:
                    buffer_save = threading.Thread(target=save_buffer_pkl)
                    buffer_save.start()
                self.save_buffer.value = 0

            # Pause learning
            with self.pause.get_lock():
                pause = self.pause.value
            if pause:
                time.sleep(0.25)
                continue

            # Ask for data, make learning updates           
            tic = time.time()
            images, propris, actions, rewards, next_images, next_propris, dones = buffer.sample()
            tic = time.time()
            self.update(images.clone(), propris.clone(), actions.clone(), 
                rewards.clone(), next_images.clone(), next_propris.clone(), dones.clone())
            with self.n_updates.get_lock():
                self.n_updates.value += 1
                if self.n_updates.value % 100 == 0:
                    print("***** SAC learning update {} took {} *****".format(self.n_updates.value, time.time() - tic))
            state_dict = {
                'actor': self.actor.state_dict(), 
                'critic': self.critic.state_dict(),
                'actor_opt': self.actor_optimizer.state_dict(),
                'critic_opt': self.critic_optimizer.state_dict(),
                'log_alpha_opt': self.log_alpha_optimizer.state_dict(),
            }
            # model_queue.put(state_dict)

    def compute_action(self, obs, deterministic=False):
        with torch.no_grad():
            with self._nn_lock:
                action, _ = self.pi(obs, with_lprob=False, det_rad=True)
        return action.detach().cpu().numpy()

    def set_pause(self, val):
        with self.pause.get_lock():
            self.pause.value = val
            print("Learning paused!" if val else "Resuming async learning ...")
