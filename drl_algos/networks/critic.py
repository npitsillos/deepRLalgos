import torch
import torch.nn as nn
import torch.nn.functional as F

from drl_algos.networks.base import Network, FeedForwardBase, RecurrentBase, ConvolutionalBase, CustomModelBase
from drl_algos.utils import utils

class Base_Q(Network):
    """
        Base class for Q function networks.

        :param features_in: Size of latent features.
    """
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.q_fn = nn.Linear(self.base.latent_size, 1)

    def forward(self, inputs):
        inputs = utils.cat(inputs, dim=1)
        inputs = utils.to_tensor(inputs, self.device)
        latent_features = self.base(inputs)
        return self.q_fn(latent_features)


class FeedForwardQ(Base_Q):
    """
        Class implementing the FeedForwardBase class for Q networks.

        :param state_dim: Input state shape.
        :param action_dim: Action output dimension.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
        :param activation_fn: Activation function to use.
    """

    def __init__(self, state_dim, action_dim, layers=(256, 256), activation_fn=F.relu, weight_init_fn=utils.fanin_init, bias_init_val=0.):
        super().__init__(FeedForwardBase((state_dim[0] + action_dim[0],), layers, activation_fn, weight_init_fn, bias_init_val))


class RecurrentQ(Base_Q):
    """
        Class implementing the RecurrentBase class for Q networks.

        :param state_dim: Input state shape.
        :param action_dim: Action output dimension.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
    """

    def __init__(self, state_dim, action_dim, layers=(128, 128)):
        super().__init__(RecurrentBase(state_dim + action_dim, layers))

        self.base.init_lstm_state()


class CustomModelQ(Base_Q):
    """
        Class implementing the CustomModelBase for Q networks.

        :param state_dim: Input state shape.
        :param action_dim: Action output dimension.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
        :param activation_fn: Activation function to use.
    """

    def __init__(self, state_dim, action_dim, layers, activation_fn=F.relu, weight_init_fn=utils.fanin_init, bias_init_val=0.):
        super().__init__(CustomModelBase(state_dim + action_dim, layers, activation_fn, weight_init_fn, bias_init_val))

        if len(self.base.rnn_layers) > 0:
            self.base.is_recurrent = True
            self.base.init_lstm_state()


class Base_Value(Network):
    """
        Base class for value function networks.

        :param features_in: Size of latent features.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.value_fn = nn.Linear(self.base.latent_size, 1)

    def forward(self, inputs):
        inputs = utils.to_tensor(inputs, self.device)
        latent_features = self.base(inputs)
        return self.value_fn(latent_features)

class FeedForwardValue(Base_Value):
    """
        Class implementing the FeedForwardBase for Value networks.

        :param state_dim: Input state shape.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
        :param activation_fn: Activation function to use.
    """

    def __init__(self, state_dim, layers=(256, 256), activation_fn=F.relu, weight_init_fn=utils.fanin_init, bias_init_val=0.):
        super().__init__(FeedForwardBase(state_dim, layers, activation_fn, weight_init_fn, bias_init_val))


class RecurrentValue(Base_Value):
    """
        Class implementing the RecurrentForwardBase for value networks.

        :param state_dim: Input state shape.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
    """

    def __init__(self, state_dim, layers=(128, 128)):
        super().__init__(RecurrentBase(state_dim, layers))

        self.base.init_lstm_state()


class CustomModelValue(Base_Value):
    """
        Class implementing the CustomModelBase for Q networks.

        :param state_dim: Input state shape.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
        :param activation_fn: Activation function to use.
    """

    def __init__(self, state_dim, layers, activation_fn=F.relu, weight_init_fn=utils.fanin_init, bias_init_val=0.):
        super().__init__(CustomModelBase(state_dim, layers, activation_fn, weight_init_fn, bias_init_val))

        if len(self.base.rnn_layers) > 0:
            self.base.init_lstm_state()