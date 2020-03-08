import os
import numpy as np
import gym
import torch
from pathlib import Path
from collections import namedtuple
import logging
import tqdm

from typing import Optional, Tuple
from algorithms.buffer import ExperienceBuffer
from algorithms.discrete_policy import DiscretePolicy
from algorithms.utils import prod

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DiscretePolicyGradientsRL:
    """
    Base class for RL algorithms.
    """

    def __init__(
        self,
        state_dimension: Tuple,
        action_space: int,
        save_path: Path,
        hyperparameters: namedtuple,
        actor_layers: Tuple,
        actor_activation: str,
        param_plot_num: int,
    ):
        self.hyp = hyperparameters
        self.state_dim_size = prod(state_dimension)

        self.actor_layers = actor_layers
        self.actor = (
            DiscretePolicy(
                state_dimension=state_dimension,
                action_space=action_space,
                hidden_layers=actor_layers,
                activation=actor_activation,
            )
            .float()
            .to(device)
        )

        self.memory_buffer = ExperienceBuffer(state_dimension, action_space)

        # Randomly select 1st layer NN weights to plot during learning
        self.chosen_params_x = np.random.randint(
            low=0, high=actor_layers[0], size=param_plot_num
        )
        self.chosen_params_y = np.random.randint(
            low=0, high=self.state_dim_size, size=param_plot_num
        )
        self.save_path = save_path

        self.policy_plot = []
        self.total_reward_plot = []
        self.ep_length_plot = []

    def sample_nn_params(self):
        """Gets randomly sampled actor NN parameters from 1st layer."""
        return self.actor.state_dict()["actor_layers.0.weight"].numpy()[
            self.chosen_params_x, self.chosen_params_y
        ]

    def choose_action(self, state: np.array) -> torch.tensor:
        """For use with test_solution() function"""
        return self.actor.pick_action(state)

    def calculate_returns(self) -> np.array:
        """Calculates the returns from the experience_buffer."""
        future_rewards = self.memory_buffer.get_rewards()
        returns = []
        future_discounted_return = 0
        for reward in reversed(future_rewards):
            future_discounted_return = reward + (
                self.hyp.gamma * future_discounted_return
            )
            returns.insert(0, future_discounted_return)
        return np.array(returns)

    def gather_experience(self, env: gym.Env, time_limit: int) -> float:
        state = env.reset()
        done = False
        total_reward, reward, timesteps = 0, 0, 0
        while not done:
            action_chosen = self.choose_action(state)
            self.memory_buffer.update(state, action_chosen, reward)
            state, reward, done, info = env.step(action_chosen)
            total_reward += reward
            timesteps += 1
            if timesteps >= time_limit:
                break
        env.close()
        print(f"Episode of experience over, total reward = \t{total_reward}")
        return total_reward

    def save_episode(self, total_episode_reward: float, episode_length: int) -> None:
        self.total_reward_plot.append(total_episode_reward)
        self.ep_length_plot.append(episode_length)
        np.save(f"{self.save_path}/returns.npy", np.array(self.total_reward_plot))
        np.save(f"{self.save_path}/episode_lengths.npy", np.array(self.ep_length_plot))

    def save_policy_params(self):
        self.policy_plot.append(self.sample_nn_params())
        np.save(
            f"{self.save_path}/policy_params.npy", np.array(self.policy_plot),
        )

    def save_network(self):
        torch.save(self.actor.state_dict(), f"{self.save_path}/actor.pt")

    def update_step(self) -> None:
        """Override with update step from relevant algorithm."""
        raise NotImplementedError
