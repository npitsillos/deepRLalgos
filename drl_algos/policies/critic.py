import torch
import torch.nn as nn
import torch.nn.functional as F

from drl_algos.policies.base import FeedForwardBase, RecurrentBase, ConvolutionalBase, CustomModelBase

class Base_Q:
    """
        Base class for Q function networks.
    """
    def __init__(self, features_in):

        self.q_fn = nn.Linear(features_in, 1)

    def q_forward(self, state, action):
        # Ask how Lewis wants to have the action + state passed
        x = torch.cat([state, action], len(state.size()) - 1)
        x = self._forward_features(x)
        return self.q_fn(x)

class FeedForwardQ(FeedForwardBase, Base_Q):
    """
        Class implementing the FeedForwardBase class for Q networks.
    """

    def __init__(self, state_dim, action_dim, layers=(256, 256), activation_fn=F.relu):
        FeedForwardBase.__init__(self, state_dim + action_dim, layers, activation_fn)
        Base_Q.__init__(self, layers[-1])

    def forward(self, state, action):
        return self.q_forward(state, action)

class RecurrentQ(RecurrentBase, Base_Q):
    """
        Class implementing the RecurrentBase class for Q networks.
    """

    def __init__(self, state_dim, action_dim, layers=(128, 128)):
        RecurrentBase.__init__(self, state_dim + action_dim, layers)
        Base_Q.__init__(self, layers[-1])

        self.is_recurrent = True
        self.init_lstm_state()
    
    def forward(self, state, action):
        return self.q_forward(state, action)

class CustomModelQ(CustomModelBase, Base_Q):
    """
        Class implementing the CustomModelBase for Q networks.
    """

    def __init__(self, state_dim, action_dim, layers, activation_fn=F.relu):
        CustomModelBase.__init__(self, state_dim + action_dim, layers, activation_fn)
        Base_Q.__init__(self, layers[-1][1])

        if len(self.rnn_layers) > 0:
            self.is_recurrent = True
            self.init_lstm_state()
        
    def forward(self, state, action):
        return self.q_forward(state, action)


class Base_Value:
    """
        Base class for value function networks.
    """

    def __init__(self, features_in):

        self.value_fn = nn.Linear(features_in, 1)

    def value_forward(self, state):
        x = self._forward_features(state)
        return self.value_fn(x)


class FeedForwardValue(FeedForwardBase, Base_Value):
    """
        Class implementing the FeedForwardBase for Value networks.
    """

    def __init__(self, state_dim, layers=(256, 256), activation_fn=F.relu):
        FeedForwardBase.__init__(self, state_dim, layers, activation_fn)
        Base_Value.__init__(self, layers[-1])

    def forward(self, state):
        return self.value_forward(state)


class RecurrentValue(RecurrentBase, Base_Value):
    """
        Class implementing the RecurrentForwardBase for value networks.
    """

    def __init__(self, state_dim, layers=(128, 128)):
        RecurrentBase.__init__(self, state_dim, layers)
        Base_Value.__init__(self, layers[-1])

        self.is_recurrent = True
        self.init_lstm_state()

    def forward(self, state):
        return self.value_forward(state)


class CustomModelValue(CustomModelBase, Base_Value):
    """
        Class implementing the CustomModelBase for Q networks.
    """

    def __init__(self, state_dim, layers, activation_fn=F.relu):
        CustomModelBase.__init__(self, state_dim, layers, activation_fn)
        Base_Value.__init__(self, layers[-1][1])

        if len(self.rnn_layers) > 0:
            self.is_recurrent = True
            self.init_lstm_state()
        
    def forward(self, state):
        return self.value_forward(state)