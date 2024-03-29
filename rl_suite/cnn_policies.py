import torch
import logging

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.nn import Parameter
from torch.distributions import Normal
from rl_suite.mlp_policies import orthogonal_weight_init


def random_augment(images, rad_height, rad_width):
    """ RAD from Laskin et al.,

    Args:
        images:
        rad_height:
        rad_width:

    Returns:

    """
    n, c, h, w = images.shape
    _h = h - 2 * rad_height
    _w = w - 2 * rad_width
    w1 = torch.randint(0, rad_width + 1, (n,))
    h1 = torch.randint(0, rad_height + 1, (n,))
    cropped_images = torch.empty((n, c, _h, _w), device=images.device).float()
    for i, (image, w11, h11) in enumerate(zip(images, w1, h1)):
        cropped_images[i][:] = image[:, h11:h11 + _h, w11:w11 + _w]
    return cropped_images

def conv_out_size(input_size, kernel_size, stride, padding=0):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.contiguous().view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints


class SSEncoderModel(nn.Module):
    """Convolutional encoder of pixels observations. Uses Spatial Softmax"""

    def __init__(self, image_shape, proprioception_shape, net_params, rad_offset, spatial_softmax=True):
        super().__init__()
        self.spatial_softmax = spatial_softmax

        if image_shape[-1] != 0:  # use image
            c, h, w = image_shape
            self.rad_h = round(rad_offset * h)
            self.rad_w = round(rad_offset * w)
            image_shape = (c, h - 2 * self.rad_h, w - 2 * self.rad_w)
            self.init_conv(image_shape, net_params)
            if spatial_softmax:
                self.latent_dim = net_params['conv'][-1][1] * 2
            else:
                self.latent_dim = net_params['latent']

            if proprioception_shape[-1] == 0:  # no proprioception readings
                self.encoder_type = 'pixel'

            else:  # image with proprioception
                self.encoder_type = 'multi'
                self.latent_dim += proprioception_shape[0]

        elif proprioception_shape[-1] != 0:
            self.encoder_type = 'proprioception'
            self.latent_dim = proprioception_shape[0]

        else:
            raise NotImplementedError('Invalid observation combination')

    def init_conv(self, image_shape, net_params):
        conv_params = net_params['conv']
        latent_dim = net_params['latent']
        channel, height, width = image_shape
        conv_params[0][0] = channel
        layers = []
        for i, (in_channel, out_channel, kernel_size, stride) in enumerate(conv_params):
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride))
            if i < len(conv_params) - 1:
                layers.append(nn.ReLU())
            width = conv_out_size(width, kernel_size, stride)
            height = conv_out_size(height, kernel_size, stride)

        self.convs = nn.Sequential(
            *layers
        )
        
        if self.spatial_softmax:
            self.ss = SpatialSoftmax(width, height, conv_params[-1][1])
        else:
            self.fc = nn.Linear(conv_params[-1][1] * width * height, latent_dim)
        # self.ln = nn.LayerNorm(latent_dim)
        self.apply(orthogonal_weight_init)

    def forward(self, images, proprioceptions, random_rad=True, detach=False):
        if self.encoder_type == 'proprioception':
            return proprioceptions

        if self.encoder_type == 'pixel' or self.encoder_type == 'multi':
            images = images / 255.
            n, c, h, w = images.shape
            if random_rad:
                images = random_augment(images, self.rad_h, self.rad_w)
            else:                
                images = images[:, :,
                         self.rad_h: h - self.rad_h,
                         self.rad_w: w - self.rad_w,
                         ]

            if self.spatial_softmax:
                h = self.ss(self.convs(images))
            else:
                h = self.fc(self.convs(images).view((n, -1)))

            if detach:
                h = h.detach()

            if self.encoder_type == 'multi':
                h = torch.cat([h, proprioceptions], dim=-1)

            return h
        else:
            raise NotImplementedError('Invalid encoder type')


class ActorModel(nn.Module):
    """ MLP actor network. """
    def __init__(self, image_shape, proprioception_shape, action_dim, net_params, rad_offset, freeze_cnn=False, spatial_softmax=True):
        super().__init__()

        self.encoder = SSEncoderModel(image_shape, proprioception_shape, net_params, rad_offset, spatial_softmax)
        logging.info("Encoder initialized for Actor")
        if freeze_cnn:
            logging.warn("Actor CNN weights won't be trained!")
            for param in self.encoder.parameters():
                param.requires_grad = False

        mlp_params = net_params['mlp']
        mlp_params[0][0] = self.encoder.latent_dim
        mlp_params[-1][-1] = action_dim * 2
        layers = []
        for i, (in_dim, out_dim) in enumerate(mlp_params):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(mlp_params) - 1:
                layers.append(nn.ReLU())
        self.trunk = nn.Sequential(
            *layers
        )

        self.outputs = dict()
        
        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)

    def forward(self, images, proprioceptions, random_rad=True, detach_encoder=False):
        latents = self.encoder(images, proprioceptions, random_rad, detach=detach_encoder)
        mu, log_std = self.trunk(latents).chunk(2, dim=-1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = Normal(mu, std)

        action = dist.sample()
        lprob = dist.log_prob(action).sum(axis=-1)

        return mu, action, lprob

    def lprob(self, images, proprioceptions, actions, random_rad=True, detach_encoder=False):
        latents = self.encoder(images, proprioceptions, random_rad, detach=detach_encoder)
        mu, log_std = self.trunk(latents).chunk(2, dim=-1)
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist.log_prob(actions).sum(axis=-1)


class CriticModel(nn.Module):
    def __init__(self, image_shape, proprioception_shape, net_params, rad_offset, freeze_cnn=False, spatial_softmax=True):
        super().__init__()

        self.encoder = SSEncoderModel(image_shape, proprioception_shape, net_params, rad_offset, spatial_softmax)
        if freeze_cnn:
            logging.warn("Critic CNN weights won't be trained!")
            for param in self.encoder.parameters():
                param.requires_grad = False

        mlp_params = net_params['mlp']
        mlp_params[0][0] = self.encoder.latent_dim
        mlp_params[-1][-1] = 1
        layers = []
        for i, (in_dim, out_dim) in enumerate(mlp_params):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(mlp_params) - 1:
                layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)

        self.outputs = dict()
        self.apply(orthogonal_weight_init)

    def forward(self, images, proprioceptions, random_rad=True, detach_encoder=False):
        latents = self.encoder(images, proprioceptions, random_rad, detach=detach_encoder)
        vals = self.trunk(latents)
        return vals.view(-1)


class SACRADActor(nn.Module):
    def __init__(self, image_shape, proprioception_shape, action_dim, net_params, rad_offset, freeze_cnn=False, spatial_softmax=True):
        super().__init__()
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

        self.encoder = SSEncoderModel(image_shape, proprioception_shape, net_params, rad_offset, spatial_softmax)
        logging.info("Encoder initialized for Actor")
        if freeze_cnn:
            logging.warn("Actor CNN weights won't be trained!")
            for param in self.encoder.parameters():
                param.requires_grad = False

        mlp_params = net_params['mlp']
        mlp_params[0][0] = self.encoder.latent_dim
        mlp_params[-1][-1] = action_dim * 2
        layers = []
        for i, (in_dim, out_dim) in enumerate(mlp_params[:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(
            *layers
        )

        self.mu_log_std = nn.Linear(mlp_params[-1][0], mlp_params[-1][1])

        self.outputs = dict()
        
        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)


    @staticmethod
    def squash(mu, action, log_p):
        """Apply squashing function.
        See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
        """
        mu = torch.tanh(mu)
        if action is not None: 
            action = torch.tanh(action)
        if log_p is not None:
            log_p -= torch.log(F.relu(1 - action.pow(2)) + 1e-6).sum(-1, keepdim=True)
        return mu, action, log_p
    
    @staticmethod
    def gaussian_logprob(noise, log_std):
        """Compute Gaussian log probability."""
        residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
        return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

    def get_features(self, images, proprioceptions):
        with torch.no_grad():
            latents = self.encoder(images, proprioceptions, random_rad=True, detach=False)
            phi = self.trunk(latents)
        return phi

    def get_image_features(self, images, proprioceptions):
        with torch.no_grad():
            latents = self.encoder(images, proprioceptions, random_rad=True, detach=False)            
        return latents

    def forward(self, images, proprioceptions, random_rad=True, detach_encoder=False):
        latents = self.encoder(images, proprioceptions, random_rad, detach=detach_encoder)
        phi = self.trunk(latents)
        mu, log_std = self.mu_log_std(phi).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        # log_std = torch.tanh(log_std)
        # log_std = self.LOG_STD_MIN + 0.5 * (
        #     self.LOG_STD_MAX - self.LOG_STD_MIN
        # ) * (log_std + 1)
        # std = log_std.exp()

        # self.outputs['mu'] = mu
        # self.outputs['std'] = std

        # # Reparametrized action sampling
        # noise = torch.randn_like(mu)
        # action = mu + noise * std

        # log_p = self.gaussian_logprob(noise, log_std)
        # mu, action, log_p = self.squash(mu, action, log_p)

        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        # Reparametrized action sampling
        action = dist.rsample()
        log_p = dist.log_prob(action).sum(axis=-1)
        log_p -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)

        # TODO: Why does spinning up apply tanh here instead of earlier?
        action = torch.tanh(action)

        return mu, action, log_p, log_std


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, latent_dim, action_dim, net_params):
        super().__init__()

        mlp_params = net_params['mlp']
        mlp_params[0][0] = latent_dim + action_dim
        mlp_params[-1][-1] = 1
        layers = []
        for i, (in_dim, out_dim) in enumerate(mlp_params):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(mlp_params) - 1:
                layers.append(nn.ReLU())
        self.trunk = nn.Sequential(
            *layers
        )

    def forward(self, latent, action):
        latent_action = torch.cat([latent, action], dim=1)
        return self.trunk(latent_action)


class SACRADCritic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(self, image_shape, proprioception_shape, action_dim, net_params, rad_offset, freeze_cnn=False, spatial_softmax=True):
        super().__init__()

        self.encoder = SSEncoderModel(image_shape, proprioception_shape, net_params, rad_offset, spatial_softmax)
        if freeze_cnn:
            logging.info("Critic CNN weights won't be trained!")
            for param in self.encoder.parameters():
                param.requires_grad = False


        self.Q1 = QFunction(
            self.encoder.latent_dim, action_dim, net_params
        )
        self.Q2 = QFunction(
            self.encoder.latent_dim, action_dim, net_params
        )

        self.outputs = dict()
        self.apply(orthogonal_weight_init)

    def forward(self, obs, state, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, state, detach=detach_encoder)
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


if __name__ == '__main__':
    net_params = {
        'conv': [
            # in_channel, out_channel, kernel_size, stride
            [-1, 32, 3, 2],
            [32, 32, 3, 2],
            [32, 32, 3, 2],
            [32, 32, 3, 1],
        ],

        'latent': 50,

        'mlp': [
            [-1, 1024],
            [1024, 1024],
            [1024, -1]
        ],
    }
    image_shape = (9, 120, 160)
    proprioception_shape = (6,)
    action_dim = 2
    rad_offset = 0.02
    device = torch.device("cuda")

    img = torch.zeros(image_shape).unsqueeze(0).to(device)
    prop = torch.zeros(proprioception_shape).unsqueeze(0).to(device)

    actor = SACRADActor(image_shape, proprioception_shape, action_dim, net_params, rad_offset).to(device)
    logging.info(actor(img, prop))
    
    critic = CriticModel(image_shape, proprioception_shape, net_params, rad_offset).to(device)
    logging.info(critic(img, prop))
    
    