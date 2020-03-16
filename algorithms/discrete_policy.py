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
        param_sharing: bool = False,
    ):
        super(DiscretePolicy, self).__init__()
        self.param_sharing = param_sharing

        act_layers, shared_layers = self.build_actor(
            hidden_layers, state_dimension, action_space, activation
        )
        self.shared_layers = nn.Sequential(*shared_layers)
        self.actor_layers = nn.Sequential(*act_layers)

    def build_actor(
        self,
        hidden_layers: Tuple,
        state_dimension: Tuple,
        action_space: int,
        activation: str,
    ):
        actor_layers, shared_layers = [], []
        prev_dimension = np.product(state_dimension)
        # print(f"len: {len(hidden_layers)}")
        # print(self.param_sharing)
        num_shared = len(hidden_layers) - 1 if self.param_sharing else 0
        # print(f"num shared: {num_shared}")

        for layer_count, layer in enumerate(hidden_layers):
            if layer_count < num_shared:
                shared_layers.append(nn.Linear(prev_dimension, layer))
                shared_layers.append(get_activation(activation))
            else:
                actor_layers.append(nn.Linear(prev_dimension, layer))
                actor_layers.append(get_activation(activation))
            prev_dimension = layer
        actor_layers.append(nn.Linear(prev_dimension, action_space))
        actor_layers.append(nn.Softmax(dim=-1))

        return actor_layers, shared_layers

    def forward(self, x):
        x = self.shared_layers(x) if self.param_sharing else x
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
