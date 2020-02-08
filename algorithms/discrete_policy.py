import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class DiscretePolicy(nn.Module):
    def __init__(
        self,
        state_dimension: Tuple,
        action_space: int,
        hidden_layers: Tuple,
        activation: str,
    ):
        super(DiscretePolicy, self).__init__()
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

        self.action_head = nn.Linear(prev_dimension, action_space)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        action_probs = torch.softmax(self.action_head(x), dim=0)
        return action_probs

    def pick_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.multinomial(1)
        return action


# def enable_gpus(model: nn.Module) -> nn.Module:
#     return model
