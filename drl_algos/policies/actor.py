import torch
import torch.nn as nn
import torch.nn.functional as F

from drl_algos.policies.base import FeedForwardBase, RecurrentBase, ConvolutionalBase, CustomModelBase

class ContinuousMixin:
    """
        Mixin class to add the set_log_std() attribute in a custom model
        with a continuous action.
    """
    def __init__(self):
        def set_log_std(log_std=None):
            if log_std is None:
                self.log_std = nn.Linear(self.action_logits.in_features, self.action_logits.out_features)
            else:
                self._log_std = log_std
                self.log_std = lambda x: self._log_std

        self.set_log_std = set_log_std


class Actor:
    """
        Base class for all discrete Actor networks.

        :param features_in: Size of latent features.
        :param action_dim: Action output dimension.
        :param is_continuous: Whether action is continuous.
    """

    def __init__(self, features_in, action_dim, is_continuous):
        
        self.action_logits = nn.Linear(features_in, *action_dim)
        self.is_continuous = is_continuous
        if self.is_continuous:
            ContinuousMixin.__init__(self)

    def logits_out(self, x):
        x = self._forward_features(x)
        if self.is_continuous:
            return self.action_logits(x), self.log_std(x)
        return self.action_logits(x)


class FeedForwardActor(FeedForwardBase, Actor):
    """
        Class implementing the FeedForwardBase class for the actor.

        :param state_dim: Input state shape.
        :param action_dim: Action output dimension.
        :param is_continuous: Whether action is continuous.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
        :param activation_fn: Activation function to use.
    """

    def __init__(self, state_dim, action_dim, is_continuous=False, layers=(256, 256), activation_fn=F.relu):
        FeedForwardBase.__init__(self, state_dim, layers, activation_fn)
        Actor.__init__(self, layers[-1], action_dim, is_continuous)

    def forward(self, state):
        return self.logits_out(state)


class RecurrentActor(RecurrentBase, Actor):
    """
        Class implementing the RecurrentBase class for the actor.

        :param state_dim: Input state shape.
        :param action_dim: Action output dimension.
        :param is_continuous: Whether action is continuous.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
    """

    def __init__(self, state_dim, action_dim, is_continuous=False, layers=(128, 128)):
        RecurrentBase.__init__(self, state_dim, layers)
        Actor.__init__(self, layers[-1], action_dim, is_continuous)

        self.is_recurrent = True
        self.init_lstm_state()
    
    def forward(self, state):
        return self.logits_out(state)


class CustomActor(CustomModelBase, Actor):
    """
        Class implementing the CustomModelBase class for the actor.
       
        :param state_dim: Input state shape.
        :param action_dim: Action output dimension.
        :param is_continuous: Whether action is continuous.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
        :param activation_fn: Activation function to use.
    """

    def __init__(self, state_dim, action_dim, layers, is_continuous=False, activation_fn=F.relu):
        CustomModelBase.__init__(self, state_dim, layers, activation_fn)
        Actor.__init__(self, layers[-1][1], action_dim, is_continuous)

        if len(self.rnn_layers) > 0:
            self.is_recurrent = True
            self.init_lstm_state()

    def forward(self, state):
        return self.logits_out(state)        