import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from typing import Tuple

from algorithms.utils import get_activation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Discriminator(nn.Module):
    def __init__(
        self,
        state_dimension: Tuple,
        action_space: int,
        hidden_layers: Tuple,
        activation: str,
    ):
        super(Discriminator, self).__init__()

        activations = ["tanh", "relu", "sigmoid"]
        assert activation in activations

        layers = []
        prev_dimension = np.product(state_dimension) + action_space
        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev_dimension, layer_size))
            layers.append(get_activation(activation))
            prev_dimension = layer_size
        layers.append(nn.Linear(prev_dimension, 1))
        layers.append(nn.Sigmoid())

        self.discrim_layers = nn.Sequential(*layers)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = self.discrim_layers(state_action)
        return x
