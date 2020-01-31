import numpy as np
import torch
from typing import Iterable, Tuple, Optional
from pathlib import Path
from collections import namedtuple

from algorithms.discrete_policy import DiscretePolicy
from algorithms.buffer import ExperienceBuffer
from algorithms.advantage_estimation import get_gae
from algorithms.base_RL import DiscretePolicyBasedRL
from algorithms.critic import Critic

HyperparametersPPO = namedtuple(
    "HyperparametersPPO",
    ("gamma", "lamda", "c1", "c2", "epsilon", "delta", "alpha", "K"),
)
"""
gamma: Discount factor for time delay in return.
lamda: GAE weighting factor.
c1: Value 
delta: Max KL divergence limit.
alpha: Backtracking coefficient for conjugate gradients method.
max_backtracking_steps: Maximum number of steps that conjugate gradients can backtrack.
"""


class PPO(DiscretePolicyBasedRL):
    def __init__(
        self,
        state_dimension: int,
        action_space: Iterable,
        hyperparameters: namedtuple,
        ref_num: int,
        data_save_location: Path,
        nn_save_location: Path,
        actor_layers: Tuple,
        critic_layers: Tuple,
        actor_activation: str = "tanh",
        critic_activation: str = "tanh",
    ):
        super(PPO, self).__init__(
            state_dimension,
            action_space,
            hyperparameters,
            ref_num,
            data_save_location,
            nn_save_location,
            actor_layers,
            actor_activation,
        )

        self.critic = Critic(
            state_dimension, hidden_layers=critic_layers, activation=critic_activation
        )

    def update_weights(
        self, returns: np.array, save_data: bool = False, step: Optional[int] = None
    ) -> None:
        gae = get_gae(
            self.memory_buffer, self.critic, gamma=self.hyp.gamma, lamda=self.hyp.lamda
        )

        clipped_r = np.clip(r, 1 - self.hyp.epsilon, 1 + self.hyp.epsilon)
        loss_clip = np.min(r * gae, clipped_r * gae)
