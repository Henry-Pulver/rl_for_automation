import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torch.distributions import Categorical
from collections import namedtuple

from buffer import PPOExperienceBuffer
from utils import get_activation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


policy_params = (
    "actor_layers",
    "actor_activation",
    "num_shared_layers",
)
try:
    DiscretePolicyParams = namedtuple(
        "DiscretePolicyParams", policy_params, defaults=(None,) * len(policy_params),
    )
except TypeError:
    DiscretePolicyParams = namedtuple("DiscretePolicyParams", policy_params)
    DiscretePolicyParams.__new__.__defaults__ = (None,) * len(policy_params)
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
        self.param_sharing = (
            params.num_shared_layers is not None and params.num_shared_layers != 0
        )
        self.params = params

        act_layers, shared_layers = self.build_actor(
            params.actor_layers, state_dimension, action_space, params.actor_activation
        )
        self.shared_layers = nn.Sequential(*shared_layers)
        self.actor_layers = nn.Sequential(*act_layers)
        self.log_softmax = nn.LogSoftmax(dim=-1)

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

    def logprobs(self, x):
        x = self.shared_layers(x) if self.param_sharing else x
        return self.log_softmax(self.actor_layers[:-1](x))

    def evaluate(self, state: np.ndarray, action: torch.tensor) -> Tuple:
        action_probs = self.forward(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, None, dist_entropy, action_probs

    def act(self, state, buffer: Optional[PPOExperienceBuffer] = None):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        try:
            action = dist.sample()
        except Exception as e:
            print(action_probs)
            raise e

        if buffer is not None:
            buffer.update(
                state,
                action,
                log_probs=dist.log_prob(action),
                action_probs=action_probs,
            )

        return action.item()
