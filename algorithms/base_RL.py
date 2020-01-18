import os
import numpy as np
import gym
import torch
from pathlib import Path
from collections import namedtuple

from typing import Optional, Tuple, Iterable
from algorithms.buffer import ExperienceBuffer
from algorithms.discrete_policy import DiscretePolicy


class DiscretePolicyBasedRL:
    """
    Base class for RL algorithms.
    """

    def __init__(
        self,
        state_dimension: int,
        action_space: Iterable,
        hyperparameters: namedtuple,
        ref_num: int,
        data_save_location: Path,
        nn_save_location: Path,
        actor_layers: Tuple,
    ):
        self.hyp = hyperparameters
        self.id = ref_num

        self.actor = DiscretePolicy(
            state_dimension=state_dimension,
            action_space=action_space,
            hidden_layers=actor_layers,
        )

        self.plots_save = f"{data_save_location}/plots/{self.id}"
        os.makedirs(self.plots_save, exist_ok=True)
        self.nn_save = f"{nn_save_location}/weights/{self.id}"
        os.makedirs(self.nn_save, exist_ok=True)

        self.memory_buffer = ExperienceBuffer()
        self.policy_plot = self.policy_weights
        self.baseline_plot = self.baseline_weights

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

    def update_weights(
        self, returns: np.array, save_data: bool = False, step: Optional[int] = None
    ) -> None:
        """"""
        # print(f"Updating weights. Current policy value: {self.policy_weights}\n")
        states, actions, action_probs = self.memory_buffer.recall_memory()
        basic_features = convert_to_feature(states)

        if self.ALPHA_BASELINE is not None:
            values = np.matmul(self.baseline_weights, basic_features.T)
            # print(f"Values: {values}\nValues shape: {values.shape}\n")
            deltas = returns - values
            # print(f"Deltas: {deltas}\n")
            delta_baseline = self.ALPHA_BASELINE * np.matmul(deltas, basic_features)
            # print(f"delta_baseline: {delta_baseline}\n")
            self.baseline_weights += delta_baseline
        next_states = get_next_states(states, self.action_space)
        # print(f"Next states: {next_states}\n")
        next_feature_vectors = convert_to_feature(next_states)
        # print(f"Next feature vectors: {next_feature_vectors}\n")
        steps = np.array(range(next_feature_vectors.shape[0]))
        chosen_features = next_feature_vectors[steps, actions]
        # print(f"Chosen features: {chosen_features}")
        grad_ln_policy = chosen_features - np.sum(
            action_probs.reshape((-1, DISC_CONSTS.ACTION_SPACE.shape[0], 1))
            * next_feature_vectors,
            axis=1,
        )
        # print(f"Grad ln policy: {grad_ln_policy}")

        # Wait 20 steps for baseline to settle
        if step > 20:
            approx_value = deltas if self.ALPHA_BASELINE is not None else returns
            self.policy_weights += self.ALPHA_POLICY * np.matmul(
                approx_value, grad_ln_policy
            )

        # if save_data:
        #     self.save_run_data(values, deltas, returns, step)

        self.policy_plot = np.append(self.policy_plot, self.policy_weights)
        if self.ALPHA_BASELINE is not None:
            self.baseline_plot = np.append(self.baseline_plot, self.baseline_weights)
            self.avg_delta_plot = np.append(self.avg_delta_plot, np.mean(deltas))
            self.ALPHA_BASELINE *= self.ALPHA_DECAY
        self.memory_buffer.clear()
        self.ALPHA_POLICY *= self.ALPHA_DECAY

    # def save_run_data(self, values, deltas, returns, step):
    #     states, actions, action_probs = self.memory_buffer.recall_memory()
    #     os.makedirs(f"REINFORCE_states/plots/{self.id}/{step}", exist_ok=True)
    #     np.save(f"REINFORCE_states/plots/{self.id}/{step}/values.npy", values)
    #     np.save(f"REINFORCE_states/plots/{self.id}/{step}/deltas.npy", deltas)
    #     np.save(
    #         f"REINFORCE_states/plots/{self.id}/{step}/states.npy", states,
    #     )
    #     np.save(
    #         f"REINFORCE_states/plots/{self.id}/{step}/action_probs.npy", action_probs,
    #     )
    #     np.save(
    #         f"REINFORCE_states/plots/{self.id}/{step}/actions.npy", actions,
    #     )
    #     np.save(
    #         f"REINFORCE_states/plots/{self.id}/{step}/rewards.npy",
    #         self.memory_buffer.get_rewards(),
    #     )
    #     np.save(f"REINFORCE_states/plots/{self.id}/{step}/returns.npy", returns)

    def save_(self) -> None:
        np.save(f"{save_path}/returns¬_{ref_num}.npy", rewards)
