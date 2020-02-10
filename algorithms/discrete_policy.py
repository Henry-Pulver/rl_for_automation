import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    def forward(self, x: np.ndarray):
        x = torch.from_numpy(x).float().to(device)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        action_probs = torch.softmax(self.action_head(x), dim=0)
        return action_probs

    def pick_action(self, x):
        action_probs = self.forward(x)

        dist = Categorical(action_probs)
        return dist.sample()

    def evaluate(self, state: np.ndarray, action: torch.tensor) -> Tuple:
        action_probs = self.forward(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy


# def enable_gpus(model: nn.Module) -> nn.Module:
#     return model
