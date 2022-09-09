from matplotlib.pyplot import axis
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.nn import Parameter
from torch.distributions import Normal

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

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


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
        self.ss = SpatialSoftmax(width, height, conv_params[-1][1])
        self.fc = nn.Linear(conv_params[-1][1] * width * height, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
        self.apply(weight_init)

    def forward(self, images, proprioceptions, random_rad=True, detach=False):
        if self.encoder_type == 'proprioception':
            return proprioceptions

        if self.encoder_type == 'pixel' or self.encoder_type == 'multi':
            images = images / 255.
            if random_rad:
                images = random_augment(images, self.rad_h, self.rad_w)
            else:
                n, c, h, w = images.shape
                images = images[:, :,
                         self.rad_h: h - self.rad_h,
                         self.rad_w: w - self.rad_w,
                         ]

            h = self.ss(self.convs(images))
            if detach:
                h = h.detach()

            if self.encoder_type == 'multi':
                h = torch.cat([h, proprioceptions], dim=-1)

            return h
        else:
            raise NotImplementedError('Invalid encoder type')

class ActorModel(nn.Module):
    """MLP actor network."""
    def __init__(self, image_shape, proprioception_shape, action_dim, net_params, rad_offset, freeze_cnn=False):
        super().__init__()

        self.encoder = SSEncoderModel(image_shape, proprioception_shape, net_params, rad_offset)
        if freeze_cnn:
            print("Actor CNN weights won't be trained!")
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
        self.apply(weight_init)

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
    def __init__(self, image_shape, proprioception_shape, net_params, rad_offset, freeze_cnn=False):
        super().__init__()

        self.encoder = SSEncoderModel(image_shape, proprioception_shape, net_params, rad_offset)
        if freeze_cnn:
            print("Critic CNN weights won't be trained!")
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
        self.apply(weight_init)

    def forward(self, images, proprioceptions, random_rad=True, detach_encoder=False):
        latents = self.encoder(images, proprioceptions, random_rad, detach=detach_encoder)
        vals = self.trunk(latents)
        return vals.view(-1)


class SACRADActor(ActorModel):
    LOG_STD_MIN = -10
    LOG_STD_MAX = 2
    
    def __init__(self, image_shape, proprioception_shape, action_dim, net_params, rad_offset, freeze_cnn=False):
        super().__init__(image_shape, proprioception_shape, action_dim, net_params, rad_offset, freeze_cnn)

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

    def forward(self, images, proprioceptions, random_rad=True, detach_encoder=False):
        latents = self.encoder(images, proprioceptions, random_rad, detach=detach_encoder)
        mu, log_std = self.trunk(latents).chunk(2, dim=-1)

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
    def __init__(self, image_shape, proprioception_shape, action_dim, net_params, rad_offset, freeze_cnn=False):
        super().__init__()

        self.encoder = SSEncoderModel(image_shape, proprioception_shape, net_params, rad_offset)
        # if freeze_cnn:
        #     print("Critic CNN weights won't be trained!")
        #     for param in self.encoder.parameters():
        #         param.requires_grad = False


        self.Q1 = QFunction(
            self.encoder.latent_dim, action_dim, net_params
        )
        self.Q2 = QFunction(
            self.encoder.latent_dim, action_dim, net_params
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, state, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, state, detach=detach_encoder)
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

class ResetActionActorModel(nn.Module):
    def __init__(self, image_shape, proprioception_shape, action_dim, net_params, rad_offset):
        super().__init__()

        self.encoder = SSEncoderModel(image_shape, proprioception_shape, net_params, rad_offset)

        mlp_params = net_params['mlp']
        mlp_params[0][0] = self.encoder.latent_dim
        mlp_params[-1][-1] = action_dim*2 - 2
        layers = []
        for _, (in_dim, out_dim) in enumerate(mlp_params[:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            # change: if i < len(mlp_params) - 1:
            layers.append(nn.ReLU())

        self.trunk = nn.Sequential(
            *layers
        )

        self.action_layer = nn.Linear(mlp_params[-1][0], mlp_params[-1][-1])
        self.reset_action_layer = nn.Linear(mlp_params[-1][0], 2)
        
        self.apply(weight_init)
        # change, initial the last layer to be 0 mean, 0 log std
        self.action_layer.weight.data.fill_(0.0)
        self.action_layer.bias.data.fill_(0.0)
        self.reset_action_layer.weight.data.fill_(0.0)
        self.reset_action_layer.bias.data.fill_(0.0)

    def get_action_module_parameters(self):
        return list(self.trunk.parameters()) + list(self.action_layer.parameters()) 
    
    def get_reset_action_module_parameters(self):
        return self.reset_action_layer.parameters()


class SAC_RAD_ResetActionActor(ResetActionActorModel):
    LOG_STD_MIN = -10
    LOG_STD_MAX = 2
    
    def __init__(self, image_shape, proprioception_shape, action_dim, net_params, rad_offset):
        super().__init__(image_shape, proprioception_shape, action_dim, net_params, rad_offset)

    def forward(self, images, proprioceptions, random_rad=True, compute_log_pi=True, detach_encoder=False):
        
        latents = self.trunk(self.encoder(images, proprioceptions, random_rad, detach=detach_encoder))
        mu, log_std = self.action_layer(latents).chunk(2, dim=-1)
        latents_no_grad = latents.detach()
        reset_mu, reset_log_std = self.reset_action_layer(latents_no_grad).chunk(2, dim=-1)
        # mu, log_std = self.trunk(latents).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (
            self.LOG_STD_MAX - self.LOG_STD_MIN
        ) * (log_std + 1)

        reset_log_std = torch.tanh(reset_log_std)
        reset_log_std = self.LOG_STD_MIN + 0.5 * (
            self.LOG_STD_MAX - self.LOG_STD_MIN
        ) * (reset_log_std + 1)

        # compute_pi:
        std = log_std.exp()
        #noise = torch.randn_like(mu)
        #pi = mu + noise * std
        pi_dist = Normal(mu, std)
        pi = pi_dist.rsample()
        if compute_log_pi:
            #log_pi = self.gaussian_logprob(noise, log_std)
            log_pi = pi_dist.log_prob(pi).sum(axis=-1)
            log_pi -= (2 * (np.log(2) - pi - F.softplus(-2 * pi))).sum(axis=1)
        else:
            log_pi = None

        reset_std = reset_log_std.exp()
        #reset_noise = torch.randn_like(reset_mu)
        #reset_action = reset_mu + reset_noise*reset_std
        reset_dist = Normal(reset_mu, reset_std)
        reset_action = reset_dist.rsample()
        if compute_log_pi:
            #log_reset_action_prob = self.gaussian_logprob(reset_noise, reset_log_std)
            log_reset_action_prob = reset_dist.log_prob(reset_action).sum(axis=-1)
            log_reset_action_prob -= (2 * (np.log(2) - reset_action - F.softplus(-2 * reset_action))).sum(axis=1)
        else:
            log_reset_action_prob = None

        mu, pi = torch.tanh(mu), torch.tanh(pi)
        reset_mu, reset_action = torch.tanh(reset_mu), torch.tanh(reset_action)
        return mu, pi, log_pi, log_std, reset_mu, reset_action, log_reset_action_prob, reset_log_std

    # def forward(self, images, proprioceptions, random_rad=True, compute_log_pi=True, detach_encoder=False):
        
    #     latents = self.trunk(self.encoder(images, proprioceptions, random_rad, detach=detach_encoder))
    #     mu, log_std = self.action_layer(latents).chunk(2, dim=-1)
    #     latents_no_grad = latents.detach()
    #     reset_mu, reset_log_std = self.reset_action_layer(latents_no_grad).chunk(2, dim=-1)

    #     # constrain log_std inside [log_std_min, log_std_max]
    #     log_std = torch.tanh(log_std)
    #     log_std = self.LOG_STD_MIN + 0.5 * (
    #         self.LOG_STD_MAX - self.LOG_STD_MIN
    #     ) * (log_std + 1)

    #     reset_log_std = torch.tanh(reset_log_std)
    #     reset_log_std = self.LOG_STD_MIN + 0.5 * (
    #         self.LOG_STD_MAX - self.LOG_STD_MIN
    #     ) * (reset_log_std + 1)

    #     std = log_std.exp()
    #     dist = Normal(mu, std)
    #     x_action = dist.rsample()
    #     log_p = None
    #     if compute_log_pi:
    #         log_p = dist.log_prob(x_action).sum(axis=-1)
    #         log_p -= (2 * (np.log(2) - x_action - F.softplus(-2 * x_action))).sum(axis=-1)

    #     reset_std = reset_log_std.exp()
    #     reset_dist = Normal(reset_mu, reset_std)
    #     reset_action = reset_dist.rsample()
    #     log_reset_action_prob = None
    #     if compute_log_pi:
    #         log_reset_action_prob = reset_dist.log_prob(reset_action).sum(axis=-1)
    #         log_reset_action_prob -= (2 * (np.log(2) - reset_action - F.softplus(-2 * reset_action))).sum(axis=-1)

        
    #     return torch.tanh(mu), torch.tanh(x_action), log_p, log_std, torch.tanh(reset_mu), torch.tanh(reset_action), log_reset_action_prob, reset_log_std

if __name__ == '__main__':
    ss_config = {
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

    critic = CriticModel(image_shape=(9, 125, 200), proprioception_shape=(6,), net_params=ss_config, rad_offset=0.02)
    img = torch.zeros((1, 9, 125, 200))
    prop = torch.zeros((1, 6))
    print(critic(img, prop))
    