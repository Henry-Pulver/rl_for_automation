import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    def __init__(
        self,
        state_dimension: Tuple,
        hidden_layers: Tuple[int],
        activation: str = "tanh",
    ):
        super(Critic, self).__init__()
        activations = ["tanh", "relu", "sigmoid"]
        assert activation in activations

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

        self.hidden_layers = nn.ModuleList()
        prev_dimension = np.product(state_dimension)
        for layer_size in hidden_layers:
            self.hidden_layers.append(nn.Linear(prev_dimension, layer_size))
            prev_dimension = layer_size

        self.value_head = nn.Linear(prev_dimension, 1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x: np.ndarray):
        x = torch.from_numpy(x).float().to(device)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        value = torch.flatten(self.value_head(x))
        return value
