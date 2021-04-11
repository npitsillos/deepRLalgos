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
        hidden_sizes,
        input_size,
        hidden_activation=F.relu,
        hidden_init=utils.fanin_init,
        b_init_value=0.,
        layer_norm=False,
        layer_norm_kwargs=None,
    ):
        super().__init__(output_size=hidden_sizes[-1])

        self.input_size = input_size
        self.hidden_activation = hidden_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.fcs.append(fc)
            self.__setattr__("fc{}".format(i), fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.layer_norms.append(ln)
                self.__setattr__("layer_norm{}".format(i), ln)

    def forward(self, input):
        features = input
        for i, fc in enumerate(self.fcs):
            features = fc(features)
            if self.layer_norm and i < len(self.fcs) - 1:
                features = self.layer_norms[i](features)
            features = self.hidden_activation(features)
        return features
