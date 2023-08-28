import torch
import numpy as np

from collections import namedtuple
from threading import Lock


class PPORADBuffer:
    Transition = namedtuple('Transition', ('img', 'prop', 'action', 'reward', 'done', 'lprob'))
    def __init__(self, image_shape, proprioception_shape, action_shape, capacity, store_lprob=False):
        self.buffer = []
        self.done_indices = []

    def push(self, images, proprioception, action, reward, done, lprob):
        """ Saves a transition. """
        self.buffer.append(self.Transition(images, proprioception, action, reward, done, lprob))
        if done:
            self.done_indices.append(len(self.buffer))

    def sample(self, batch_size):
        if batch_size >= len(self.buffer):
            batch = self.Transition(*zip(*self.buffer))
        else:
            raise NotImplemented

        if torch.is_tensor(batch.prop[0]):
            propris = torch.cat(batch.prop, dim=0)
        else:
            propris = torch.from_numpy(np.stack(batch.prop).astype(np.float32))

        images = torch.from_numpy(np.stack(batch.img, axis=0).astype(np.float32))
        actions = torch.from_numpy(np.stack(batch.action).astype(np.float32))
        rewards = torch.from_numpy(np.stack(batch.reward).astype(np.float32))
        dones = torch.from_numpy(np.stack(batch.done).astype(np.float32))
        lprobs = torch.from_numpy(np.stack(batch.lprob).astype(np.float32)).view(-1)

        return images, propris, actions, rewards, dones, lprobs

    @property
    def n_episodes(self):
        return len(self.done_indices)

    def reset(self):
        self.buffer = []
        self.done_indices = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        t = self.Transition(*zip(*self.buffer[item]))
        return t.img, t.prop, t.action, t.reward, t.done, t.lprob


class SACRADBuffer(object):
    """ Buffer to store environment transitions. """
    def __init__(self, image_shape, proprioception_shape, action_shape, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.ignore_image = True
        self.ignore_propri = True

        if image_shape[-1] != 0:
            self.images = np.zeros((capacity, *image_shape), dtype=np.uint8)
            self.ignore_image = False

        if proprioception_shape[-1] != 0:
            self.propris = np.zeros((capacity, *proprioception_shape), dtype=np.float32)
            self.ignore_propri = False

        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.count = 0

        size_of_buffer = (((((self.images.size * self.images.itemsize) + (self.propris.size * self.propris.itemsize) + \
            (self.actions.size * self.actions.itemsize) + (8 * capacity)) / 1024) / 1024)/ 1024)
        print("Size of replay buffer: {:.2f}GB".format(size_of_buffer))

    def add(self, image, propri, action, reward, done):
        if not self.ignore_image:
            self.images[self.idx] = image
        if not self.ignore_propri:
            self.propris[self.idx] = propri

        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.count = self.capacity if self.full else self.idx

    def sample(self):
        idxs = np.random.randint(
            0, self.count-1, size=min(self.count-1, self.batch_size)
        )
        if self.ignore_image:
            images = None
            next_images = None
        else:
            images = torch.as_tensor(self.images[idxs]).float()
            next_images = torch.as_tensor(self.images[idxs+1]).float()
        if self.ignore_propri:
            propris = None
            next_propris = None
        else:
            propris = torch.as_tensor(self.propris[idxs]).float()
            next_propris = torch.as_tensor(self.propris[idxs+1]).float()        

        actions = torch.as_tensor(self.actions[idxs])
        rewards = torch.as_tensor(self.rewards[idxs])
        dones = torch.as_tensor(self.dones[idxs])

        return images, propris, actions, rewards, next_images, next_propris, dones


class SACReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, capacity, batch_size):
        self.batch_size = batch_size
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, capacity
        self.lock = Lock()

        size_of_buffer = (((((self.observations.size * self.observations.itemsize) + \
            (self.actions.size * self.actions.itemsize) + (8 * capacity)) / 1024) / 1024)/ 1024)
        print("Size of replay buffer: {:.2f}GB".format(size_of_buffer))

    def add(self, obs, action, reward, done):
        with self.lock:
            self.observations[self.ptr] = obs
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.ptr = (self.ptr+1) % self.max_size
            self.size = min(self.size+1, self.max_size)

    def sample(self):
        with self.lock:
            idxs = np.random.randint(0, self.size-1, size=self.batch_size)
            observations = torch.from_numpy(self.observations[idxs, :])
            next_observations = torch.from_numpy(self.observations[idxs+1, :])
            actions = torch.from_numpy(self.actions[idxs])
            rewards = torch.from_numpy(self.rewards[idxs])
            dones = torch.from_numpy(self.dones[idxs])
            return (observations, actions, rewards, dones, next_observations)

    def __len__(self):
        return self.size


class ResetSACBuffer(SACReplayBuffer):
    def __init__(self, obs_dim, act_dim, capacity, batch_size, reset_thresh):
        super().__init__(obs_dim, act_dim, capacity, batch_size)
        self.reset_actions = np.zeros(capacity, dtype=np.float32)
        self.reset_thresh = reset_thresh
    
    def add(self, obs, action, reward, done, reset_action):
        super().add(obs, action, reward, done)
        with self.lock:
            self.reset_actions[self.ptr] = reset_action

    def sample(self):
        with self.lock:
            idxs = np.random.randint(0, self.size-1, size=self.batch_size)
            observations = torch.from_numpy(self.observations[idxs, :])
            next_observations = torch.from_numpy(self.observations[idxs+1, :])
            actions = torch.from_numpy(self.actions[idxs])
            rewards = torch.from_numpy(self.rewards[idxs])
            dones = torch.from_numpy(self.dones[idxs])
            reset_actions = torch.from_numpy(self.reset_actions[idxs])
            return (observations, actions, rewards, dones, next_observations, reset_actions)

    def sample(self):
        with self.lock:
            # TODOL Heuristic to ensure we have enough samples
            idxs = np.random.randint(0, self.size-1, size=self.batch_size*4)
            reset_actions = torch.from_numpy(self.reset_actions[idxs])
            idxs = (reset_actions < self.reset_thresh).nonzero().view(-1).numpy()[:self.batch_size]

            observations = torch.from_numpy(self.observations[idxs, :])
            next_observations = torch.from_numpy(self.observations[idxs+1, :])
            actions = torch.from_numpy(self.actions[idxs])
            rewards = torch.from_numpy(self.rewards[idxs])
            dones = torch.from_numpy(self.dones[idxs])
            
            return (observations, actions, rewards, dones, next_observations)
    
    def reset_action_sample(self):
        with self.lock:            
            idxs = np.random.randint(0, self.size-1, size=self.batch_size)                       

            observations = torch.from_numpy(self.observations[idxs, :])
            next_observations = torch.from_numpy(self.observations[idxs+1, :])
            rewards = torch.from_numpy(self.rewards[idxs])
            dones = torch.from_numpy(self.dones[idxs])
            reset_actions = torch.from_numpy(self.reset_actions[idxs]).view((-1, 1))
            
            return (observations, reset_actions, rewards, dones, next_observations)
