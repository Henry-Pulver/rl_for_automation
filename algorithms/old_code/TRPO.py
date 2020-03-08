import numpy as np
import torch
import torch.nn as nn
import gym
from typing import Tuple, Iterable
from pathlib import Path
from collections import namedtuple

from old_code.critic import Critic

HyperparametersTRPO = namedtuple("HyperparametersTRPO", ("delta", "alpha", "K"))
"""
delta: Max KL divergence limit.
alpha: Backtracking coefficient for conjugate gradients method.
max_backtracking_steps: Maximum number of steps that conjugate gradients can backtrack.
"""

hyp = HyperparametersTRPO(delta=0.5, alpha=0.7, K=20)

n = np.array([4, 2])
p = np.array([2, 2])
print(np.sum(np.dot(n, p)))


class DiscreteTRPO:
    """
    Class for learning using TRPO algorithm using GAE for advantage estimate.

    Args:


        nn_save_location: Relative file location where nn will be saved
    """

    def __init__(
        self,
        state_dimension: int,
        action_space: Iterable,
        hyperparameters: namedtuple,
        nn_save_location: Path,
        data_save_location: Path,
        actor_layers: Tuple,
        critic_layers: Tuple,
    ):

        self.critic = Critic(
            state_dimension=state_dimension, hidden_layers=critic_layers
        )
        self.value_loss = nn.MSELoss()

    def train(self, env: gym.Env, episode_limit: int):
        # Gather experience
        self.gather_experience(env, time_limit=episode_limit)

        # Compute rewards from experience

        # Compute advantage estimates GAE

        # Estimate policy gradient

        # Use conjugate gradient algorithm to compute x_hat

        # Fit value function by regression on mean-squared error
        self.value_loss()

    def gather_experience(self, env: gym.Env, time_limit: int) -> float:
        state = env.reset()
        done = False
        total_reward, reward, timestep = 0, 0, 0

        while not done:
            action_chosen = self.actor.pick_action(state)

            self.memory_buffer.update(state, action_chosen, reward)

            state, reward, done, info = env.step(action_chosen)
            total_reward += reward
            timestep += 1

            if timestep >= time_limit:
                break
        env.close()

    def save_network(self):
        filename = "none"
        self.save_location.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{self.save_location}/{filename}.pt")


def main():
    hyp = HyperparametersTRPO(delta=1, alpha=23, K=42)
    DiscreteTRPO(hyperparameters=hyp,)


if __name__ == "__main__":
    main()
