import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple

from algorithms.discrete_policy import DiscretePolicy
from algorithms.buffer import PPOExperienceBuffer
from algorithms.utils import get_activation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(DiscretePolicy):
    def __init__(
        self,
        state_dimension: Tuple,
        action_space: int,
        actor_layers: Tuple,
        actor_activation: str,
        critic_layers: Tuple,
        critic_activation: str,
        param_sharing: bool = False,
    ):
        super(ActorCritic, self).__init__(
            state_dimension, action_space, actor_layers, actor_activation, param_sharing
        )
        if self.param_sharing:
            for act_layer, crit_layer in zip(actor_layers[:-1], critic_layers[:-1]):
                assert act_layer == crit_layer
        critic_mlp = []
        prev_dimension = critic_layers[-2] if self.param_sharing else np.product(state_dimension)

        critic_layers = (critic_layers[-1],) if self.param_sharing else critic_layers
        for layer in critic_layers:
            critic_mlp.append(nn.Linear(prev_dimension, layer))
            critic_mlp.append(get_activation(critic_activation))
            prev_dimension = layer
        critic_mlp.append(nn.Linear(prev_dimension, 1))
        self.critic_layers = nn.Sequential(*critic_mlp)
        # print(self.state_dict())

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

    def act(self, state, buffer: PPOExperienceBuffer):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.forward(state)
        # print(action_probs)
        dist = Categorical(action_probs)
        action = dist.sample()

        buffer.update(
            state, action, log_probs=dist.log_prob(action), action_probs=action_probs
        )

        return action.item()
