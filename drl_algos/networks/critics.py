import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from drl_algos.networks import Network, Base, Mlp
from drl_algos import utils


class Critic(Network):

    def __init__(
            self,
            base: Base,
            output_size,
            layer_init=None,
    ):
        super().__init__()
        self.base = base

        last_hidden_size = base.output_size
        self.fc_q = nn.Linear(last_hidden_size, output_size)
        if layer_init is None:
            self.fc_q.weight.data.uniform_(-3e-3, 3e-3)
        else:
            utils.initialise(self.fc_q.weight, layer_init, F.linear)
        self.fc_q.bias.data.fill_(0)

    def forward(self, *inputs):
        flat_inputs = utils.cat(inputs, dim=1)
        flat_inputs = utils.to_tensor(flat_inputs, self.device)
        features = self.base(flat_inputs)
        return self.fc_q(features)


class MlpCritic(Critic):

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
            output_size=output_size,
        )

class MlpCritic2(Critic):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        base_kwargs,
        *args,
        **kwargs,
    ):
        super().__init__(
            base=Mlp(
                input_size=input_size,
                layer_sizes=hidden_sizes,
                **base_kwargs,
            ),
            output_size=output_size,
            *args,
            **kwargs
        )
