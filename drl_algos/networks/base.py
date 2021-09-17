import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from drl_algos import utils


class Network(nn.Module):
    """Wraps torch nn.Module to provide device type tracking."""

    def __init__(self):
        super().__init__()
        self.device = "cpu"

    def to(self, *args, **kwargs):
        """Override super method to track device."""
        if kwargs.get("device") is not None:
            self.device = kwargs.get("device")
        elif isinstance(args[0], str):
            self.device = args[0]
        return super().to(*args, **kwargs)

    def cuda(self, device=0):
        """Override super method to track device."""
        self.device = "cuda:" + str(device)
        return super().cuda(device)


class Base(Network):
    """High level class defining network architecture."""

    def __init__(
        self,
        output_size,
    ):
        """Tracks output size."""
        super().__init__()
        self.output_size = output_size

    def reset(self):
        """To be called whenever the agent needs reset."""
        pass


class Mlp(Base):
    """Fully Connected Network with optional Layer Norm"""

    def __init__(
        self,
        layer_sizes,
        input_size,
        layer_activation=F.relu,
        act_last_layer=True,
        layer_init="fanin",
        bias_init_value=0.,
        layer_norm=False,
        layer_norm_kwargs=None,
    ):
        super().__init__(output_size=layer_sizes[-1])

        self.input_size = input_size
        self.layer_activation = layer_activation
        self.act_last_layer = act_last_layer # If false, no activation on last layer
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []

        in_size = input_size
        for i, next_size in enumerate(layer_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            utils.initialise(fc.weight, layer_init, layer_activation)
            fc.bias.data.fill_(bias_init_value)
            self.fcs.append(fc)
            self.__setattr__("fc{}".format(i), fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.layer_norms.append(ln)
                self.__setattr__("layer_norm{}".format(i), ln)

    def forward(self, input):
        features = input
        last_index = len(self.fcs)-1
        for i, fc in enumerate(self.fcs):
            features = fc(features)
            if self.layer_norm and i < last_index:
                features = self.layer_norms[i](features)
            if i < last_index or self.act_last_layer:
                features = self.layer_activation(features)
        return features


class Gru(Base):
    """Gru Network"""

    def __init__(
        self,
        input_size,
        layer_sizes,
    ):
        super().__init__(output_size=layer_sizes[-1])

        self.input_size = input_size
        self.grus = []

        in_size = input_size
        for i, next_size in enumerate(layer_sizes):
            gru = nn.GRUCell(input_size=in_size, hidden_size=next_size)
            self.grus.append(gru)
            self.__setattr__("gru{}".format(i), gru)
            in_size = next_size

    def forward(self, input, hidden):
        """ To extend to stacked GRU, need to take [batch_size, layers, hidden]
        instead of [batch_size, hidden]. Note - would only use final hidden for
        more processing but need rest of hiddens for subsequent calls.
        """
        if len(self.grus) > 1:
            new_hiddens = []
            for i, gru in enumerate(self.grus):
                new_hidden = gru(input, hidden[:,i])
                new_hiddens.append(new_hidden)
                input = new_hidden
            return torch.stack(new_hiddens, dim=1) # TODO - check this is correct
        else:
            return self.grus[0](input, hidden)
        # return new_hiddens # shape should be [batch_size, layers, hidden]
