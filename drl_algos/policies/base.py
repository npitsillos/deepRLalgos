import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple

str_to_layer_fn = {
    "fc": nn.Linear,
    "rnn": nn.LSTMCell,
    "cnn": nn.Conv2d
}

def create_fn(input_dim: Tuple[int], layers: Union[Tuple[int, int], Tuple[Tuple[str, int]]], layer_fn: Union[nn.Linear, nn.Conv2d, nn.LSTM] = None) -> nn.ModuleList():
    """
        Creates a ModuleList torch module containing the layers in the base network.

        :param input_dim: A tuple of ints defining the input size.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
        :param layer_fn: Type of layer if a single type, None if a custom
            model is needed.

        :return: nn.ModuleList containing the layers of the base network.
    """
    net = nn.ModuleList()
    if layer_fn is not None:
        # This is the case where a single type of layer policy is needed
        net += [layer_fn(*input_dim, layers[0])]
        for layer in range(len(layers) - 1):
            net += [layer_fn(layers[layer], layers[layer + 1])]
    else:
        net += [str_to_layer_fn[layers[0][0]](*input_dim, layers[0][1])]
        for layer in range(len(layers) - 1):
            net += [str_to_layer_fn[layers[layer + 1][0]](layers[layer][1], layers[layer + 1][1])]
    return net


class RecurrentMixin:
    """
        Mixin class to add the init_hidden_state attribute in a custom model
        with an LSTM layer.
    """
    def __init__(self):
        def init_lstm_state(batch_size=1):
            self.hidden = [torch.zeros(batch_size, self.layers[idx].hidden_size) for idx in self.rnn_layers]
            self.cell = [torch.zeros(batch_size, self.layers[idx].hidden_size) for idx in self.rnn_layers]

        self.init_lstm_state = init_lstm_state


class Net(nn.Module):
    """
        Base class which all policies inherit from.
    """

    def __init__(self):
        super(Net, self).__init__()

        self.is_recurrent = False


class FeedForwardBase(Net):
    """
        Base class for all networks using only linear layers.

        :param input_dim: A tuple of ints defining the input size.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
        :param activation_fn: Activation function to use.
    """

    def __init__(self, input_dim, layers, activation_fn):
        super(FeedForwardBase, self).__init__()

        self.layers = create_fn(input_dim, layers, nn.Linear)
        self.activation_fn = activation_fn

    def _forward_features(self, x):
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return x

class ConvolutionalBase(Net):
    """
        Base class for all networks using only convolutional layers.
        
        :param input_dim: A tuple of ints defining the input size.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
        :param activation_fn: Activation function to use.
    """
    
    def __init__(self, input_dim, layers, activation_fn):

        self.layers = create_fn(input_dim, layers, nn.Conv2d)
        self.activation_fn = activation_fn
    
    def _forward_features(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

class RecurrentBase(Net):
    """
        Base class for all networks using only recurrent layers.

        :param input_dim: A tuple of ints defining the input size.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
    """

    def __init__(self, input_dim, layers):
        super(RecurrentBase, self).__init__()

        self.layers = create_fn(input_dim, layers, nn.LSTMCell)
        self.is_recurrent = True

    def init_lstm_state(self, batch_size=1):
        # Do we also want to init from buffer?
        self.hidden = [torch.zeros(batch_size, layer.hidden_size) for layer in self.layers]
        self.cell = [torch.zeros(batch_size, layer.hidden_size) for layer in self.layers]

    def _forward_features(self, x):
        # Two cases here either 3-dim x -> T x B x F, batch of trajectories having T timesteps
        # or 2-dim/1-dim -> batch of timesteps
        if len(x.shape) == 3: # batch of trajectories
            self.init_lstm_state(x.size(1))

            x_s = []
            for x_t in x:
                for idx, layer in enumerate(self.layers):
                    hidden, cell = self.hidden[idx], self.cell[idx]
                    self.hidden[idx], self.cell[idx] = layer(x_t, (hidden, cell))
                    x_t = self.hidden[idx]
                x_s.append(x_t)
            
            x = torch.stack(x_s)
        else:
            dim = len(x.shape)
            if dim == 1:
                x = x.view(1, -1)

            for idx, layer in enumerate(self.layers):
                hidden, cell = self.hidden[idx], self.cell[idx]
                self.hidden[idx], self.cell[idx] = layer(x, (hidden, cell))
                x = self.hidden[idx]

            if dim == 1:
                x = x.view(-1)

        return x

class CustomModelBase(Net):
    """
        Base class for all networks that use a custom policy.

        :param input_dim: A tuple of ints defining the input size.
        :param layers: A tuple defining the layer sizes if a single layer type
            and the type of layer and their size if a custom model is needed.
        :param activation_fn: Activation function to use.
    """

    def __init__(self, input_dim, layers, activation_fn):
        super(CustomModelBase, self).__init__()

        self.layers = create_fn(input_dim, layers)
        self.activation_fn = activation_fn
        self.rnn_layers = [i for i, layer in enumerate(layers) if layer[0] == "rnn"]
        if len(self.rnn_layers) > 0:
            RecurrentMixin.__init__(self)

    def _forward_features(self, x):

        lstm_cell_idx = 0
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.LSTMCell):
                # Need to take into account the fact that here we pass timesteps x features
                # but need to turn tensor to timesteps x trajectory_batch x features given the indices of
                # trajectory start end.
                if len(x.size()) == 1: x = x.view(1, -1)
                hidden, cell = self.hidden[lstm_cell_idx], self.cell[lstm_cell_idx]
                self.hidden[lstm_cell_idx], self.cell[lstm_cell_idx] = layer(x, (hidden, cell))
                x = self.hidden[lstm_cell_idx]
                lstm_cell_idx += 1
            else:
                x = self.activation_fn(layer(x))
        return x