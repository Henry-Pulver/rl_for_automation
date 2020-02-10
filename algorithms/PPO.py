import numpy as np
import gym
import torch
from torch.distributions import Categorical
from typing import Tuple, Optional
from pathlib import Path
from collections import namedtuple
import logging

from algorithms.buffer import PPOExperienceBuffer
from algorithms.advantage_estimation import get_gae
from algorithms.base_RL import DiscretePolicyGradientsRL
from algorithms.critic import Critic

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyp_names = (
    "gamma",
    "lamda",
    "actor_learning_rate",
    "critic_learning_rate",
    "epsilon",
    "T",
    "c1",
    "c2",
    "num_epochs",
)
HyperparametersPPO = namedtuple(
    "HyperparametersPPO", hyp_names, defaults=(None,) * len(hyp_names),
)
"""
    gamma: Discount factor for time delay in return.
    lamda: GAE weighting factor.
    T: Time horizon.
    epsilon: PPO clipping parameter.
    actor_learning_rate: Learning rate of Adam optimizer on Actor.
    critic_learning_rate: Learning rate of Adam optimizer on Critic.
    c1: (Optional - only if sharing parameters) Value function loss weighting factor.
    c2: Entropy bonus loss term weighting factor.
    num_epochs: Number of epochs of learning carried out on each T timesteps.
"""

hyperparams_ppo_atari = HyperparametersPPO(
    gamma=0.99,
    lamda=0.95,
    actor_learning_rate=1e-3,
    critic_learning_rate=1e-3,
    T=128,
    epsilon=0.2,
    c2=0.01,
    num_epochs=3,
)


class PPO(DiscretePolicyGradientsRL):
    def __init__(
        self,
        state_dimension: Tuple,
        action_space: int,
        save_path: Path,
        hyperparameters: HyperparametersPPO,
        actor_layers: Tuple,
        critic_layers: Tuple,
        actor_activation: str,
        critic_activation: str,
        param_plot_num: int = 10,
        param_sharing: bool = False,
    ):
        if param_sharing:
            raise NotImplementedError

        super(PPO, self).__init__(
            state_dimension,
            action_space,
            save_path,
            hyperparameters,
            actor_layers,
            actor_activation,
            param_plot_num,
        )
        self.save_path = self.save_path / f"{self.hyp.num_epochs}-epochs"
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.memory_buffer = PPOExperienceBuffer(state_dimension, action_space)

        # Zero the parameter gradients
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.hyp.critic_learning_rate
        )
        self.actor_optimizer.zero_grad()
        self.critic_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.hyp.actor_learning_rate
        )

        self.critic = (
            Critic(
                state_dimension,
                hidden_layers=critic_layers,
                activation=critic_activation,
            )
            .float()
            .to(device)
        )
        self.critic_loss = torch.nn.MSELoss()
        self.loss_plots = {"entropy_loss": [], "clipped_loss": [], "value_loss": []}

    def train_episode(
        self, env: gym.Env, max_episode_length: Optional[int] = None
    ) -> Tuple:
        state = env.reset()
        total_reward, reward, timestep, done = 0, 0, 0, False
        while not done:
            action_chosen = self.act_and_remember(state, reward)
            state, reward, done, info = env.step(action_chosen)
            total_reward += reward
            timestep += 1

            if timestep % self.hyp.T == 0:
                self.update_weights()

            # Checks episode length doesn't exceed specified maximum
            if max_episode_length is not None:
                if timestep >= max_episode_length:
                    done = True

            if done:
                env.close()
                if not timestep % self.hyp.T == 0:
                    self.update_weights()

                self.save_episode(total_reward, timestep)
        return total_reward, timestep

    def update_weights(self) -> None:
        (
            numpy_states,
            actions_taken,
            _,
            old_log_probs,
        ) = self.memory_buffer.recall_memory()
        actions_taken = torch.from_numpy(actions_taken).to(device)
        old_log_probs = torch.from_numpy(old_log_probs).to(device)

        for epoch in range(self.hyp.num_epochs):
            logprobs, dist_entropy = self.actor.evaluate(numpy_states, actions_taken)
            policy_ratios = torch.exp(logprobs - old_log_probs)
            state_values = self.critic(numpy_states).to(device)
            gae = (
                torch.from_numpy(
                    get_gae(
                        self.memory_buffer,
                        state_values.detach().numpy(),
                        gamma=self.hyp.gamma,
                        lamda=self.hyp.lamda,
                    )
                )
                .float()
                .to(device)
            )

            critic_loss = self.critic_loss(state_values, gae)
            self.critic_optimizer.zero_grad()
            critic_loss.mean().backward()
            self.critic_optimizer.step()

            r_clip = torch.clamp(
                policy_ratios, 1 - self.hyp.epsilon, 1 + self.hyp.epsilon
            )
            loss_clip = -torch.min(policy_ratios * gae, r_clip * gae)
            entropy_loss = -self.hyp.c2 * dist_entropy

            actor_loss = loss_clip + entropy_loss

            self.actor_optimizer.zero_grad()
            actor_loss.mean().backward()
            self.actor_optimizer.step()

            save_plots = epoch == (self.hyp.num_epochs - 1)
            self.plot_losses(
                loss_clip, entropy_loss, critic_loss, save_plots,
            )
            self.save_policy_params(save_plots)

        self.memory_buffer.clear()

    def save_network(self):
        super(PPO, self).save_network()
        torch.save(self.critic.state_dict(), f"{self.save_path}/critic.pt")

    def plot_losses(self, clipped_loss, entropy_loss, value_loss, save: bool):
        self.loss_plots["clipped_loss"].append(clipped_loss)
        self.loss_plots["entropy_loss"].append(entropy_loss)
        self.loss_plots["value_loss"].append(value_loss)
        if save:
            np.save(
                f"{self.save_path}/mean_clipped_loss.npy",
                np.array(self.loss_plots["clipped_loss"]),
            )
            np.save(
                f"{self.save_path}/mean_entropy_loss.npy",
                np.array(self.loss_plots["entropy_loss"]),
            )
            np.save(
                f"{self.save_path}/mean_value_loss.npy",
                np.array(self.loss_plots["value_loss"]),
            )

    def act_and_remember(self, state, reward):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_chosen = dist.sample()
        self.memory_buffer.update(
            state,
            action_chosen,
            reward=reward,
            log_probs=dist.log_prob(action_chosen).detach(),
        )
        return action_chosen.numpy()

    def gather_experience(self, env: gym.Env, time_limit: int) -> float:
        raise NotImplementedError


def train_ppo_network(
    env_str: str,
    random_seed: int,
    actor_nn_layers: Tuple,
    save_path: Path,
    hyp: namedtuple,
    actor_activation: str,
    critic_layers: Tuple,
    critic_activation: str,
    max_episodes: int,
    network_save_freq: int = 100,
    max_episode_length: Optional[int] = None,
    log_level=logging.INFO,
):
    save_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=f"{save_path}/log.log", level=log_level)
    if max_episode_length is not None:
        env = gym.make(env_str).env
    else:
        env = gym.make(env_str)

    if random_seed is not None:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    ppo = PPO(
        state_dimension=env.observation_space.shape,
        action_space=env.action_space.n,
        save_path=save_path,
        hyperparameters=hyp,
        actor_layers=actor_nn_layers,
        actor_activation=actor_activation,
        critic_layers=critic_layers,
        critic_activation=critic_activation,
    )

    for ep_num in range(1, max_episodes + 1):
        total_reward, episode_length = ppo.train_episode(env, max_episode_length)

        ep_str = ("{0:0" + f"{len(str(max_episodes))}" + "d}").format(ep_num)
        print(f"Episode {ep_str} of {max_episodes}. \t Total reward = {total_reward}")
        logger.log(
            logging.INFO,
            f"Episode {ep_str} of {max_episodes}.\t Total reward = {total_reward}",
        )
        if ep_num % network_save_freq:
            ppo.save_network()
