import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from collections import namedtuple
from typing import Tuple

from discrete_policy import DiscretePolicy, DiscretePolicyParams
from utils import get_activation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


ac_params = (
    "actor_layers",
    "critic_layers",
    "actor_activation",
    "critic_activation",
    "num_shared_layers",
)
try:
    ActorCriticParams = namedtuple(
        "ActorCriticParams", ac_params, defaults=(None,) * len(ac_params),
    )
except TypeError:
    ActorCriticParams = namedtuple("ActorCriticParams", ac_params)
    ActorCriticParams.__new__.__defaults__ = (None,) * len(ac_params)
"""
    actor_layers: Tuple of actor layer sizes. 
    critic_layers: Tuple of critic layer sizes.
    actor_activation: String defining the activation function of every actor layer.
    critic_activation: String defining the activation function of every critic layer.
    num_shared_layers: (Optional) number of layers to share across actor and critic.
"""


class ActorCritic(DiscretePolicy):
    def __init__(
        self, state_dimension: Tuple, action_space: int, params: ActorCriticParams,
    ):
        super(ActorCritic, self).__init__(
            state_dimension,
            action_space,
            DiscretePolicyParams(
                params.actor_layers, params.actor_activation, params.num_shared_layers
            ),
        )
        if self.param_sharing:
            for act_layer, crit_layer in zip(
                params.actor_layers[: params.num_shared_layers],
                params.critic_layers[: params.num_shared_layers],
            ):
                assert act_layer == crit_layer
        critic_mlp = []
        prev_dimension = (
            params.critic_layers[params.num_shared_layers - 1]
            if self.param_sharing
            else np.product(state_dimension)
        )

        critic_layers = (
            params.critic_layers[params.num_shared_layers :]
            if self.param_sharing
            else params.critic_layers
        )
        for layer in critic_layers:
            critic_mlp.append(nn.Linear(prev_dimension, layer))
            critic_mlp.append(get_activation(params.critic_activation))
            prev_dimension = layer
        critic_mlp.append(nn.Linear(prev_dimension, 1))
        self.critic_layers = nn.Sequential(*critic_mlp)

    def value_and_action_probs(self, x):
        shared_output = self.shared_layers(x) if self.param_sharing else x
        action_probs = self.actor_layers(shared_output)
        value = self.critic_layers(shared_output)
        return value, action_probs

    def evaluate(self, state: np.ndarray, action: torch.tensor) -> Tuple:
        state_value, action_probs = self.value_and_action_probs(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy, action_probs
