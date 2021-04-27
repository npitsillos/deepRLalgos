import torch
import torch.nn as nn
import torch.nn.functional as F

from drl_algos.networks.base import Network, FeedForwardBase, RecurrentBase, ConvolutionalBase, CustomModelBase
from drl_algos.utils.distributions import Delta, Discrete, TanhNormal, MultivariateDiagonalNormal
from drl_algos.utils import utils

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class Policy(Network):
    """
        Base class for all discrete Actor networks.

        :param features_in: Size of latent features.
        :param action_dim: Action output dimension.
        :param is_continuous: Whether action is continuous.
    """
    def get_action(self, state):
        raise NotImplementedError

    def reset(self):
        pass

class StochasticPolicy(Policy):

    def get_action(self, obs):
        obs = utils.to_tensor(obs[None], self.device)
        dist = self(obs)
        actions = dist.sample()
        actions = utils.to_numpy(actions)
        if len(actions.shape) == 0:
            return actions, {}
        return actions[0, :], {}


class DeterministicPolicy(StochasticPolicy):

    def __init__(self, policy):
        super().__init__()
        self.device = policy.device
        self._policy = policy

    def forward(self, *args, **kwargs):
        dist = self._policy(*args, **kwargs)
        return Delta(dist.mle_estimate())


class GaussianPolicy(StochasticPolicy):

    def __init__(self, base, action_dim, std, dist, init_w=1e-3):
        super().__init__()
        self.base = base
        self.log_std = None
        self.std = std
        self.action_dim = action_dim
        self.dist = dist

        self.fc_mean = nn.Linear(self.base.latent_size, *action_dim)
        self.fc_mean.weight.data.uniform_(-init_w, init_w)
        self.fc_mean.bias.data.fill_(0)

        if std is None:
            # Learned std layer
            self.fc_log_std = nn.Linear(self.base.latent_size, *action_dim)
            self.fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            # Fixed std layer
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        features = self.base(obs)
        mean = self.fc_mean(features)
        if self.std is None:
            log_std = self.fc_log_std(features)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = utils.to_tensor(np.array([self.std, ]), self.device)

        return self.dist(mean, std)

    def logprob(self, action, mean, std):
        tanh_normal = self.dist(mean, std)
        log_prob = tanh_normal.log_prob(action)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob


class CategoricalPolicy(nn.Module):

    def __init__(self, base, action_dim, init_w=1e-3):
        super().__init__()
        self.device = "cuda:0"
        self.base = base
        self.dist = Discrete
        self.logits = nn.Linear(self.base.latent_size, *action_dim)
        self.logits.weight.data.uniform_(-init_w, init_w)
        self.logits.bias.data.fill_(0)
    
    def get_action(self, obs):
        obs = utils.to_tensor(obs[None], self.device)
        dist = self(obs)
        actions = dist.sample()
        actions = utils.to_numpy(actions)
        return actions, {}
    
    def forward(self, obs, print_l=False):
        features = self.base(obs)
        logits = self.logits(features)
        return self.dist(logits=logits)
    
    def reset(self):
        pass

class FeedForwardGaussianPolicy(GaussianPolicy):
    """
        Class implementing the FeedForwardBase class for the actor.

        :param state_dim: Input state shape.
        :param action_dim: Action output dimension.
        :param is_continuous: Whether action is continuous.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
        :param activation_fn: Activation function to use.
    """

    def __init__(self, state_dim, action_dim, dist, layers=(256, 256), activation_fn=F.relu, weight_init_fn=utils.fanin_init, bias_init_val=0., std=None):
        super().__init__(FeedForwardBase(state_dim, layers, activation_fn, weight_init_fn, bias_init_val), action_dim, std, dist) 


class FeedForwardCategoricalPolicy(CategoricalPolicy):

    def __init__(self, state_dim, action_dim, layers=(256, 256), activation_fn=F.relu, weight_init_fn=None, bias_init_val=0.):
        super().__init__(FeedForwardBase(state_dim, layers, activation_fn, weight_init_fn, bias_init_val), action_dim)


class RecurrentGaussianPolicy(GaussianPolicy):
    """
        Class implementing the RecurrentBase class for the actor.

        :param state_dim: Input state shape.
        :param action_dim: Action output dimension.
        :param is_continuous: Whether action is continuous.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
    """

    def __init__(self, state_dim, action_dim, dist, layers=(128, 128), std=None):
        super().__init__(RecurrentBase(state_dim, layers), action_dim, std, dist)

        self.base.init_lstm_state()


class CustomGaussianPolicy(GaussianPolicy):
    """
        Class implementing the CustomModelBase class for the actor.
       
        :param state_dim: Input state shape.
        :param action_dim: Action output dimension.
        :param is_continuous: Whether action is continuous.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
        :param activation_fn: Activation function to use.
    """

    def __init__(self, state_dim, action_dim, layers, dist, activation_fn=F.relu, weight_init_fn=utils.fanin_init, bias_init_val=0., std=None):
        super().__init__(CustomModelBase(state_dim, layers, activation_fn, weight_init_fn, bias_init_val), action_dim, std, dist)

        if len(self.base.rnn_layers) > 0:
            self.base.init_lstm_state()