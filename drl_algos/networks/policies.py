import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from drl_algos.networks import Network, Base, Mlp
from drl_algos.distributions import TanhNormal, Delta
from drl_algos import utils


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class Policy(Network):
    """High level policy interface."""

    def get_action(self, obs):
        """Returns an action given a observation.

        Args:
            obs_np: observation batch as a numpy array.

        Returns:
            numpy array of actions,
            dictionary of additional information"""
        raise NotImplementedError

    def reset(self):
        """To be called whenever the agent needs reset."""
        pass


class StochasticPolicy(Policy):
    """Class of Policies containing Distributions."""

    def get_action(self, obs):
        obs = utils.to_tensor(obs[None], self.device)
        dist = self(obs)
        actions = dist.sample()
        # log_p = utils.to_numpy(dist.log_prob(actions))
        actions = utils.to_numpy(actions)
        # return actions[0, :], {"log_p": log_p}
        return actions[0, :], {}


class MakeDeterministic(StochasticPolicy):
    def __init__(
            self,
            policy: StochasticPolicy,
    ):
        super().__init__()
        self.device = policy.device
        self._policy = policy

    def forward(self, *args, **kwargs):
        dist = self._policy(*args, **kwargs)
        return Delta(dist.mle_estimate())


class GaussianPolicy(StochasticPolicy):

    def __init__(
        self,
        base: Base,
        action_dim,
        std=None,
        mean_layer_init=None,
        std_layer_init=None,
    ):
        super().__init__()
        self.base = base
        self.log_std = None
        self.std = std
        self.action_dim = action_dim

        # Action mean layer
        last_hidden_size = base.output_size
        self.fc_mean = nn.Linear(last_hidden_size, action_dim)
        if mean_layer_init is None:
            self.fc_mean.weight.data.uniform_(-1e-3, 1e-3)
        else:
            utils.initialise(self.fc_mean.weight, mean_layer_init, F.tanh)
        self.fc_mean.bias.data.fill_(0)

        if std is None:
            # Learned std layer
            self.fc_log_std = nn.Linear(last_hidden_size, action_dim)
            if std_layer_init is None:
                self.fc_log_std.weight.data.uniform_(-1e-3, 1e-3)
            else:
                utils.initialise(self.fc_log_std.weight, std_layer_init, F.tanh)
            self.fc_log_std.bias.data.fill_(0)
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

        return TanhNormal(mean, std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(action)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob


class MlpGaussianPolicy(GaussianPolicy):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
    ):
        super().__init__(
            base=Mlp(
                layer_sizes=hidden_sizes,
                input_size=input_size,
            ),
            action_dim=output_size,
        )

class MlpGaussianPolicy2(GaussianPolicy):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        base_kwargs,
        **kwargs,
    ):
        super().__init__(
            base=Mlp(
                input_size=input_size,
                layer_sizes=hidden_sizes,
                **base_kwargs,
            ),
            output_size=output_size,
            **kwargs
        )

class MlpSacPolicy(GaussianPolicy):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        layer_init="fanin",
        init_mean=False,
    ):
        super().__init__(
            base=Mlp(
                input_size=input_size,
                layer_sizes=hidden_sizes,
                layer_init=layer_init,
                layer_activation=F.relu,
            ),
            action_dim=output_size,
            mean_layer_init=layer_init if init_mean else None,
        )
