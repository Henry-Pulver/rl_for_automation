import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch.distributions import Categorical
from collections import namedtuple

from utils import get_activation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


param_names = (
    "actor_layers",
    "actor_activation",
    "num_shared_layers",
)
try:
    DiscretePolicyParams = namedtuple(
        "DiscretePolicyParams", param_names, defaults=(None,) * len(param_names),
    )
except TypeError:
    DiscretePolicyParams = namedtuple("DiscretePolicyParams", param_names)
    DiscretePolicyParams.__new__.__defaults__ = (None,) * len(param_names)
"""
    actor_layers: Tuple of actor layer sizes. 
    actor_activation: String defining the activation function of every actor layer.
    num_shared_layers: (Optional) number of layers to share across actor and critic.
"""


class DiscretePolicy(nn.Module):
    def __init__(
        self, state_dimension: Tuple, action_space: int, params: DiscretePolicyParams,
    ):
        super(DiscretePolicy, self).__init__()
        self.param_sharing = params.num_shared_layers is not None
        self.params = params

        act_layers, shared_layers = self.build_actor(
            params.actor_layers, state_dimension, action_space, params.actor_activation
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
        num_shared = self.params.num_shared_layers if self.param_sharing else 0

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
