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
    ):
        super(ActorCritic, self).__init__(
            state_dimension, action_space, actor_layers, actor_activation
        )

        critic_mlp = []
        prev_dimension = np.product(state_dimension)
        for layer in critic_layers:
            critic_mlp.append(nn.Linear(prev_dimension, layer))
            critic_mlp.append(get_activation(critic_activation))
            prev_dimension = layer
        critic_mlp.append(nn.Linear(prev_dimension, 1))
        self.critic_layers = nn.Sequential(*critic_mlp)

    def value(self, x):
        for layer in self.critic_layers:
            x = self.critic_activation(layer(x))
        return x

    def evaluate(self, state: np.ndarray, action: torch.tensor) -> Tuple:
        action_probs = self.forward(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic_layers(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy, action_probs

    def act(self, state, buffer: PPOExperienceBuffer):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        buffer.update(
            state, action, log_probs=dist.log_prob(action), action_probs=action_probs
        )

        return action.item()
