import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from collections import namedtuple

from algorithms.utils import get_activation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

discrim_params = (
    "hidden_layers",
    "activation",
)
DiscrimParams = namedtuple("DiscrimParams", discrim_params)
"""
    hidden_layers: Tuple of discriminator layer sizes.
    activation: String defining the activation function of every discriminator
                        layer.
"""


class Discriminator(nn.Module):
    ACTIVATIONS = ["tanh", "relu", "sigmoid"]

    def __init__(self, state_dim: Tuple, action_space: int, params: DiscrimParams):
        super(Discriminator, self).__init__()
        assert params.activation in self.ACTIVATIONS

        layers = []
        prev_dimension = np.product(state_dim) + action_space
        for layer_size in params.hidden_layers:
            layers.append(nn.Linear(prev_dimension, layer_size))
            layers.append(get_activation(params.activation))
            prev_dimension = layer_size
        layers.append(nn.Linear(prev_dimension, 2))
        self.softmax_out = nn.Softmax(dim=-1)
        self.log_softmax_out = nn.LogSoftmax(dim=-1)

        self.discrim_layers = nn.Sequential(*layers)

    def forward(self, state_action):
        return self.softmax_out(self.discrim_layers(state_action))

    def prob_expert(self, state_action):
        prob_expert = self.forward(state_action).t()[1]
        return prob_expert

    def logprobs(self, state_action):
        return self.log_softmax_out(self.discrim_layers(state_action))
