import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.distributions import Normal, Categorical

# dict to enable loading activations based on a string
nn_activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
}


def mlp_hidden_layers(input_dim, hidden_sizes, activation="relu"):
    """ Helper function to create hidden MLP layers.
    N.B: The same activation is applied after every layer

    Args:
        input_dim: An int denoting the input size of the mlp
        hidden_sizes: A list with ints containing hidden sizes
        activation: A str specifying the activation function

    Returns:

    """
    activation = nn_activations[activation]
    dims = [input_dim] + hidden_sizes
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation)
    return layers


def orthogonal_weight_init(m):
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

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        m.bias.data.zero_()
    

class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, nn_params, device):
        super(MLPGaussianActor, self).__init__()
        self.device = device

        layers = mlp_hidden_layers(input_dim=obs_dim, hidden_sizes=nn_params["mlp"]["hidden_sizes"],
                                   activation=nn_params["mlp"]["activation"])
        self.phi = nn.Sequential(*layers)
        self.mu = nn.Linear(nn_params["mlp"]["hidden_sizes"][-1], action_dim)
        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def _dist(self, x):
        phi = self.phi(x)
        mu = self.mu(phi)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def lprob(self, x, a):
        x = x.to(self.device)
        dist = self._dist(x)
        # Last axis sum needed for Torch Normal distribution
        lprob = dist.log_prob(torch.as_tensor(a, device=self.device)).sum(axis=-1)
        return lprob, dist

    def compute_action(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            dist = self._dist(x)
            action = dist.sample()
            lprob = dist.log_prob(action).sum(axis=-1)
        return action.cpu().numpy(), lprob.cpu().item()


class Critic(nn.Module):
    def __init__(self, obs_dim, nn_params, device):
        super(Critic, self).__init__()
        self.device = device

        layers = mlp_hidden_layers(input_dim=obs_dim, hidden_sizes=nn_params["mlp"]["hidden_sizes"],
                                   activation=nn_params["mlp"]["activation"])
        self.phi = nn.Sequential(*layers)
        self.value = nn.Linear(nn_params["mlp"]["hidden_sizes"][-1], 1)

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, x):
        x = x.to(self.device)
        phi = self.phi(x)
        return self.value(phi).view(-1)


class SquashedGaussianMLPActor(nn.Module):
    """ Continous MLP Actor for Soft Actor-Critic """

    def __init__(self, obs_dim, action_dim, nn_params, device):
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        super(SquashedGaussianMLPActor, self).__init__()
        self.device = device

        layers = mlp_hidden_layers(input_dim=obs_dim, hidden_sizes=nn_params["mlp"]["hidden_sizes"],
                                   activation=nn_params["mlp"]["activation"])
        self.phi = nn.Sequential(*layers)

        self.mu = nn.Linear(nn_params["mlp"]["hidden_sizes"][-1], action_dim)
        self.log_std = nn.Linear(nn_params["mlp"]["hidden_sizes"][-1], action_dim)

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        
        self.to(device=device)

    def _dist(self, x):
        phi = self.phi(x)
        mu = self.mu(phi)
        log_std = self.log_std(phi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return Normal(mu, std)

    def forward(self, x, with_lprob=True):
        x = x.to(self.device)
        dist = self._dist(x)
        action = dist.rsample()
        if with_lprob:
            lprob = dist.log_prob(action).sum(axis=-1)
            lprob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)
        else:
            lprob = None
        action = torch.tanh(action)
        return dist.mean, action, lprob, torch.log(dist.stddev)

    def get_features(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            phi = self.phi(x)
        return phi


class SquashedGaussianMLP_ResetActionActor(nn.Module):
    """ Continous MLP Actor for Soft Actor-Critic """

    LOG_STD_MAX = 2
    LOG_STD_MIN = -10

    def __init__(self, obs_dim, action_dim, nn_params, device):
        super().__init__()
        self.device = device

        layers = mlp_hidden_layers(input_dim=obs_dim, hidden_sizes=nn_params["mlp"]["hidden_sizes"],
                                   activation=nn_params["mlp"]["activation"])
        self.phi = nn.Sequential(*layers)

        self.mu = nn.Linear(nn_params["mlp"]["hidden_sizes"][-1], action_dim-1)
        self.log_std = nn.Linear(nn_params["mlp"]["hidden_sizes"][-1], action_dim-1)

        self.reset_mu = nn.Linear(nn_params["mlp"]["hidden_sizes"][-1], 1)
        self.reset_log_std = nn.Linear(nn_params["mlp"]["hidden_sizes"][-1], 1)
        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)

        self.mu.weight.data.fill_(0)
        self.mu.bias.data.fill_(0)

        self.log_std.weight.data.fill_(0)
        self.log_std.bias.data.fill_(0)

        self.reset_mu.weight.data.fill_(0)
        self.reset_mu.bias.data.fill_(0)

        self.reset_log_std.weight.data.fill_(0)
        self.reset_log_std.bias.data.fill_(0)

        self.to(device=device)

    def get_action_module_parameters(self):
        return list(self.phi.parameters()) + list(self.mu.parameters()) + list(self.log_std.parameters())

    def get_reset_action_module_parameters(self):
        return list(self.reset_mu.parameters()) + list(self.reset_log_std.parameters())

    def _dist(self, x):
        phi = self.phi(x)
        mu = self.mu(phi)
        log_std = self.log_std(phi)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (
            self.LOG_STD_MAX - self.LOG_STD_MIN
        ) * (log_std + 1)

        std = torch.exp(log_std)

        phi_no_grad = phi.detach()
        reset_mu = self.reset_mu(phi_no_grad)
        reset_log_std = self.reset_log_std(phi_no_grad)
        reset_log_std = torch.tanh(reset_log_std)
        reset_log_std = self.LOG_STD_MIN + 0.5 * (
            self.LOG_STD_MAX - self.LOG_STD_MIN
        ) * (reset_log_std + 1)

        reset_std = torch.exp(reset_log_std)

        return Normal(mu, std), Normal(reset_mu, reset_std)

    def forward(self, x, with_lprob=True):
        x = x.to(self.device)
        dist, reset_dist = self._dist(x)

        x_action = dist.rsample()
        reset_action = reset_dist.rsample()

        if with_lprob:
            lprob = dist.log_prob(x_action).sum(axis=-1)
            reset_lprob = reset_dist.log_prob(reset_action).sum(axis=-1)

            lprob -= (2 * (np.log(2) - x_action - F.softplus(-2 * x_action))).sum(axis=1)
            reset_lprob -= (2 * (np.log(2) - reset_action - F.softplus(-2 * reset_action))).sum(axis=1)
        else:
            lprob = None
            reset_lprob = None
        
        x_action = torch.tanh(x_action)
        reset_action = torch.tanh(reset_action)

        return torch.tanh(dist.mean), x_action, lprob, torch.log(dist.stddev), torch.tanh(reset_dist.mean), reset_action, reset_lprob, torch.log(reset_dist.stddev)

    def get_features(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            phi = self.phi(x)
        return phi


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, nn_params, device):
        super(MLPQFunction, self).__init__()
        self.device = device

        layers = mlp_hidden_layers(input_dim=obs_dim + action_dim,
                                   hidden_sizes=nn_params["mlp"]["hidden_sizes"], 
                                   activation=nn_params["mlp"]["activation"])
        self.phi = nn.Sequential(*layers)
        self.q = nn.Linear(nn_params["mlp"]["hidden_sizes"][-1], 1)

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs, action):
        obs = obs.to(self.device)
        action = action.to(self.device) 
        phi = self.phi(torch.cat([obs, action], dim=-1))
        q = self.q(phi)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

    def get_features(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            phi = self.phi(x)
        return phi


class SACCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, nn_params, device):
        super(SACCritic, self).__init__()
        self.device = device

        # build value functions
        self.Q1 = MLPQFunction(obs_dim, action_dim, nn_params, device)
        self.Q2 = MLPQFunction(obs_dim, action_dim, nn_params, device)
        self.to(device)

    def forward(self, obs, action):
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)
        return q1, q2


class LinearSquashedPolicy(nn.Module):
    """ Continous MLP Actor for Soft Actor-Critic """

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, obs_dim, action_dim, device):
        super(LinearSquashedPolicy, self).__init__()
        self.device = device       

        self.mu = nn.Linear(obs_dim, action_dim)
        self.log_std = nn.Linear(obs_dim, action_dim)

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def _dist(self, x):
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return Normal(mu, std)

    def forward(self, x, with_lprob=True):
        x = x.to(self.device)
        dist = self._dist(x)
        action = dist.rsample()
        if with_lprob:
            lprob = dist.log_prob(action).sum(axis=-1)
            lprob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)
        else:
            lprob = None
        action = torch.tanh(action)
        return dist.mean, action, lprob, torch.log(dist.stddev)


class MLPDiscreteActor(nn.Module):
    """ Discrete MLP Actor for Soft Actor-Critic """

    def __init__(self, obs_dim, action_dim, nn_params, device):
        super().__init__()
        self.device = device

        layers = mlp_hidden_layers(input_dim=obs_dim, hidden_sizes=nn_params["mlp"]["hidden_sizes"],
                                   activation=nn_params["mlp"]["activation"])
        self.phi = nn.Sequential(*layers)

        self.logits = nn.Linear(nn_params["mlp"]["hidden_sizes"][-1], action_dim)
        
        # Weight Initialization
        self.apply(kaiming_init)
        nn.init.xavier_uniform_(self.logits.weight)
        self.logits.bias.data.zero_()

        self.to(device=device)       

    def forward(self, x):
        x = x.to(self.device)
        phi = self.phi(x)
        logits = self.logits(phi)        
        probs = F.softmax(logits, -1)
        z = probs == 0.0
        z = z.float() * 1e-8
        return Categorical(probs), probs + z

    def get_features(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            phi = self.phi(x)
        return phi


class DiscreteQFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, nn_params, device):
        super(DiscreteQFunction, self).__init__()
        self.device = device

        layers = mlp_hidden_layers(input_dim=obs_dim,
                                   hidden_sizes=nn_params["mlp"]["hidden_sizes"], 
                                   activation=nn_params["mlp"]["activation"])
        self.phi = nn.Sequential(*layers)
        self.q = nn.Linear(nn_params["mlp"]["hidden_sizes"][-1], action_dim)

        # Weight Initialization
        self.apply(kaiming_init)
        nn.init.xavier_uniform_(self.q.weight)
        self.q.bias.data.zero_()

        self.to(device=device)

    def forward(self, obs):
        obs = obs.to(self.device)        
        q = self.q(self.phi(obs))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

    def get_features(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            phi = self.phi(x)
        return phi


class SACDiscreteCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, nn_params, device):
        super(SACDiscreteCritic, self).__init__()
        self.device = device

        # build value functions
        self.Q1 = DiscreteQFunction(obs_dim, action_dim, nn_params, device)
        self.Q2 = DiscreteQFunction(obs_dim, action_dim, nn_params, device)
        self.to(device)

    def forward(self, obs):
        q1 = self.Q1(obs)
        q2 = self.Q2(obs)
        return q1, q2
