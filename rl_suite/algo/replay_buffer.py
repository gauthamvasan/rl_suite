import torch
import numpy as np
from collections import namedtuple


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
