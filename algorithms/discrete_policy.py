import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch.distributions import Categorical

from algorithms.utils import get_activation

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

        layers = self.build_actor(
            hidden_layers, state_dimension, action_space, activation
        )
        self.actor_layers = nn.Sequential(*layers)

    def build_actor(
        self,
        hidden_layers: Tuple,
        state_dimension: Tuple,
        action_space: int,
        activation: str,
    ):
        sequential_layers = []
        prev_dimension = np.product(state_dimension)
        for layer in hidden_layers:
            sequential_layers.append(nn.Linear(prev_dimension, layer))
            sequential_layers.append(get_activation(activation))
            prev_dimension = layer
        sequential_layers.append(nn.Linear(prev_dimension, action_space))
        sequential_layers.append(nn.Softmax(dim=-1))
        return sequential_layers

    def forward(self, x):
        action_probs = self.actor_layers(x)
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
