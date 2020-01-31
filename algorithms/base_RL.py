import os
import numpy as np
import gym
import torch
from pathlib import Path
from collections import namedtuple
import logging
import tqdm
import math

from typing import Optional, Tuple
from algorithms.buffer import ExperienceBuffer
from algorithms.discrete_policy import DiscretePolicy
from algorithms.utils import prod


class DiscretePolicyBasedRL:
    """
    Base class for RL algorithms.
    """

    def __init__(
        self,
        state_dimension: Tuple,
        action_space: int,
        hyperparameters: namedtuple,
        ref_num: int,
        data_save_location: Path,
        nn_save_location: Path,
        actor_layers: Tuple,
        actor_activation: str = "tanh",
        param_plot_num: int = 10,
    ):
        self.hyp = hyperparameters
        self.id = ref_num
        self.algorithm_name = "base"
        self.state_dim_size = prod(state_dimension)

        self.actor = DiscretePolicy(
            state_dimension=state_dimension,
            action_space=action_space,
            hidden_layers=actor_layers,
            activation=actor_activation,
        ).float()

        self.overall_save = data_save_location
        self.plots_save = f"{data_save_location}/plots/{self.algorithm_name}/{self.id}"
        os.makedirs(self.plots_save, exist_ok=True)
        self.nn_save = f"{nn_save_location}/weights/{self.algorithm_name}/{self.id}"
        os.makedirs(self.nn_save, exist_ok=True)

        self.memory_buffer = ExperienceBuffer(state_dimension, action_space)
        self.policy_plot = self.actor.parameters()
        tensor_str = "hidden_layers.0.weight"
        chosen_params_x = np.random.randint(low=0, high=self.state_dim_size - 1, size=param_plot_num)
        chosen_params_y = np.random.randint(low=0, high=actor_layers[0] - 1, size=param_plot_num)
        print(self.actor.state_dict()[tensor_str].numpy()[chosen_params_x, chosen_params_y])

    def choose_action(self, state: np.array) -> torch.tensor:
        """For use with test_solution() function"""
        return self.actor.pick_action(state)

    def calculate_returns(self) -> np.array:
        episode_len = self.memory_buffer.get_length()
        try:
            gammas = np.logspace(
                0, np.log10(self.hyp.gamma ** (episode_len - 1)), num=episode_len
            )
        except AttributeError:
            gammas = 1.0
        future_rewards = self.memory_buffer.get_rewards()
        returns = []
        for timestep in range(episode_len):
            returns.append(np.sum(np.dot(future_rewards, gammas)))
            # Remove last element from gammas array, remove 1st element from rewards
            future_rewards = future_rewards[1:]
            gammas = gammas[:-1]
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

    def update_step(
        self, returns: np.array, save_data: bool = False, step: Optional[int] = None
    ) -> None:
        """Override with update step from relevant algorithm."""
        self.memory_buffer.clear()

    def train_network(self, env_name: str, log_level=logging.INFO):
        logging.basicConfig(
            filename=f"{self.overall_save}/{self.id}.log", level=log_level
        )

    def save_(self) -> None:
        np.save(f"{save_path}/returns_{ref_num}.npy", rewards)
        np.save(f"{save_path}/moving_avg_{ref_num}.npy", moving_avg)
        np.save(f"{save_path}/returns_{ref_num}.npy", rewards)

def train_policy(
    env: gym.Env,
    num_episodes: int,
    max_episode_length: int,
    discount_factor: float,
    ref_num: Optional[int] = None,
    random_seed: Optional[int] = None,
):
    torch.manual_seed(random_seed)
    save_path = f"REINFORCE_states/plots/{ref_num}"

    moving_avg = np.array([0])
    rewards = np.array([0])



    try:
        for step in tqdm(range(num_steps)):
            total_reward = policy.gather_experience(env, 10000)
            returns = policy.calculate_returns()
            policy.update_weights(returns, save_data=(step % 100 == 0), step=step)

            rewards = np.append(rewards, total_reward)
            moving_avg = np.append(
                moving_avg, 0.01 * total_reward + 0.99 * moving_avg[-1]
            )

            if step % 10 == 0:
                policy.save()
                save_performance_plots()

                # Output progress message
                print(
                    f"Trial {ref_num}, Step: {step}\tAvg: {moving_avg[-1]}\tAlpha_policy: {policy.ALPHA_POLICY}\tAlpha_baseline: {policy.ALPHA_BASELINE}"
                )

            if abs(moving_avg[-1]) < 150:
                print(
                    f"Problem successfully solved - policy saved at {policy.weights_save}!"
                )
                break

    finally:
        policy.save()
        save_performance_plots()
    return moving_avg[-1]
