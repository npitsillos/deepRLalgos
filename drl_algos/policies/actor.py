import torch
import torch.nn as nn
import torch.nn.functional as F

from drl_algos.policies.base import FeedForwardBase, RecurrentBase, ConvolutionalBase, CustomModelBase

class Actor:
    """
        Base class for all Actor networks.
    """

    def __init__(self, features_in, act_dim):
        
        self.policy = nn.Linear(features_in, act_dim)

    def logits_out(self, x):
        x = self._forward_features(x)
        return self.policy(x)

class FeedForwardActor(FeedForwardBase, Actor):
    """
        Class implementing the FeedForwardBase class for the actor.
    """

    def __init__(self, state_dim, action_dim, layers=(256, 256), activation_fn=F.relu):
        FeedForwardBase.__init__(self, state_dim, layers, activation_fn)
        Actor.__init__(self, layers[-1], action_dim)

    def forward(self, state):
        return self.logits_out(x)


class RecurrentActor(RecurrentBase, Actor):
    """
        Class implementing the RecurrentBase class for the actor.
    """

    def __init__(self, state_dim, action_dim, layers=(128, 128)):
        RecurrentBase.__init__(self, state_dim, layers)
        Actor.__init__(self, layers[-1], action_dim)

        self.is_recurrent = True
        self.init_lstm_state()
    
    def forward(self, state):
        return self.logits_out(state)


class CustomActor(CustomModelBase, Actor):
    """
        Class implementing the CustomModelBase class for the actor.
    """

    def __init__(self, state_dim, action_dim, layers, activation_fn=F.relu):
        CustomModelBase.__init__(self, state_dim, layers, activation_fn)
        Actor.__init__(self, layers[-1][1], act_dim)

        if len(self.rnn_layers) > 0:
            self.is_recurrent = True
            self.init_lstm_state()

    def forward(self, state):
        return self.logits_out(state)        