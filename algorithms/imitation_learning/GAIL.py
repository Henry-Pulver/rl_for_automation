from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import gym
from collections import namedtuple
import logging
from typing import Tuple, List, Optional

from algorithms.buffer import DemonstrationBuffer, ExperienceBuffer
from algorithms.discriminator import Discriminator
from algorithms.actor_critic import ActorCritic
from algorithms.PPO import PPO
from algorithms.utils import generate_save_location, generate_gail_str

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyp_names = (
    "gamma",
    "T",
    "num_epochs",
    "learning_rate",
    "num_demos",
    "c1",
    "c2",
    "lamda",
    "epsilon",
    "beta",
    "d_targ",
)


try:
    HyperparametersGAIL = namedtuple(
        "HyperparametersGAIL", hyp_names, defaults=(None,) * len(hyp_names),
    )
except TypeError:
    HyperparametersGAIL = namedtuple("HyperparametersGAIL", hyp_names)
    HyperparametersGAIL.__new__.__defaults__ = (None,) * len(hyp_names)
"""
    gamma: Discount factor for time delay in return.
    T: Time horizon.
    num_epochs: Number of epochs of learning carried out on each T timesteps.
    learning_rate: Learning rate of Adam optimizer on Actor and Critic.
    discrim_lr: Discriminator learning rate for Adam optimizer.
    num_demos: Number of demonstrations used.
    c1: Value function loss weighting factor.
    c2: (Optional) Entropy bonus loss term weighting factor.
    lamda: (Optional) GAE weighting factor.
    epsilon: (Optional) PPO clipping parameter.
    beta: (Optional) KL penalty parameter.
    d_targ: (Optional) Adaptive KL target.
"""


def sample_from_buffers(
    demo_buffer: DemonstrationBuffer,
    experience_buffer: ExperienceBuffer,
    batch_size: int,
    action_space: int,
):
    learner_states, learner_actions, _ = experience_buffer.recall_memory()
    learner_action_vectors = np.zeros((len(learner_actions), action_space))
    learner_action_vectors[np.arange(learner_actions.size), learner_actions] = 1
    learner_tuples = np.append(learner_states, learner_action_vectors, axis=1)

    num_expert_samples = int(batch_size - len(learner_actions))
    demo_buffer.to_numpy()
    expert_states, expert_actions, _ = demo_buffer.random_sample(num_expert_samples)
    demo_buffer.from_numpy()
    expert_action_vectors = np.zeros((len(expert_actions), action_space))
    expert_action_vectors[np.arange(expert_actions.size), expert_actions] = 1
    expert_tuples = np.append(expert_states, expert_action_vectors, axis=1)

    data_set = np.append(expert_tuples, learner_tuples, axis=0)
    labels = np.append(
        np.ones(num_expert_samples), np.zeros(batch_size - num_expert_samples)
    )

    return data_set, labels


class GAILTrainer(PPO):
    def __init__(
        self,
        state_dimension: Tuple,
        action_space: int,
        save_path: Path,
        demo_path: Path,
        hyperparameters: HyperparametersGAIL,
        actor_layers: Tuple,
        critic_layers: Tuple,
        discriminator_layers: Tuple,
        actor_activation: str,
        critic_activation: str,
        discriminator_activation: str,
        param_plot_num: int = 2,
        entropy: bool = True,
        ppo_type: str = "clip",
        advantage_type: str = "monte_carlo",
    ):
        super(GAILTrainer, self).__init__(
            state_dimension,
            action_space,
            save_path,
            hyperparameters,
            actor_layers,
            critic_layers,
            actor_activation,
            critic_activation,
            param_plot_num,
            entropy,
            ppo_type,
            advantage_type,
        )

        self.demo_buffer = DemonstrationBuffer(demo_path, state_dimension, action_space)

        self.discrim_lr = self.hyp.discrim_lr
        # self.actor_critic = ActorCritic(
        #     state_dimension,
        #     action_space,
        #     actor_layers,
        #     actor_activation,
        #     critic_layers,
        #     critic_activation,
        # ).to(device)
        self.discriminator = Discriminator(
            state_dimension,
            action_space,
            discriminator_layers,
            discriminator_activation,
        ).to(device)

        self.discrim_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr
        )

        self.discrim_params_x = np.random.randint(
            low=0, high=critic_layers[0], size=param_plot_num
        )
        self.discrim_params_y = np.random.randint(
            low=0, high=state_dimension[0], size=param_plot_num
        )
        self.discrim_plot = []
        self.loss_plots = {
            "discriminator_loss": [],
            "entropy_loss": [],
            "clipped_loss": [],
            "value_loss": [],
        }

    def sample_nn_params(self):
        actor_params, critic_params = super(GAILTrainer, self).sample_nn_params()
        discrim_params = self.discriminator.state_dict()["discrim_layers.0.weight"].numpy()[
            self.discrim_params_x, self.discrim_params_y
        ]
        return actor_params, critic_params, discrim_params

    def record_policy_params(self):
        actor_params, critic_params, discrim_params = self.sample_nn_params()
        self.actor_plot.append(actor_params)
        self.critic_plot.append(critic_params)
        self.discrim_plot.append(discrim_params)
        np.save(
            f"{self.save_path}/policy_params.npy", np.array(self.actor_plot),
        )
        np.save(
            f"{self.save_path}/critic_params.npy", np.array(self.critic_plot),
        )
        np.save(
            f"{self.save_path}/discrim_params.npy", np.array(self.discrim_plot),
        )

    def update(self, buffer: ExperienceBuffer):




    # state_array = np.array([])
    # action_array = np.array([])
    # state_array = state_array.reshape((-1, 2))
    # for demo_num in range(num_demos):
    #     self.demo_buffer.load_demos(demo_num)
    #     states, actions, _ = self.demo_buffer.recall_memory()
    #     state_array = np.append(state_array, states)
    #     action_array = np.append(action_array, actions)
    # state_array = state_array.reshape((-1, 2))
    # states = torch.from_numpy(state_array)
    # actions = torch.from_numpy(action_array)
    #
    # for epoch in range(num_epochs):
    #     print(f"Epoch number: {epoch + 1}")
    #     sum_loss, num_steps = 0, 0
    #     for state, action in zip(states, actions):
    #         # forward + backward + optimize
    #         action_probs = self.network(state.float()).unsqueeze(dim=0)
    #         action_probs = action_probs.float()
    #         action = action.unsqueeze(dim=0)
    #         action = action.type(torch.long)
    #
    #         loss = self.criterion(action_probs, action)
    #         loss.backward()
    #         self.optimizer.step()
    #         num_steps += 1
    #         sum_loss += loss
    #     print(f"Avg loss: {sum_loss / num_steps}")


def train_network(
    demo_path: Path,
    env_name: str,
    actor_layers: Tuple,
    critic_layers: Tuple,
    discrim_layers: Tuple,
    actor_activation: str,
    critic_activation: str,
    discrim_activation: str,
    max_episodes: int,
    update_timestep: int,
    log_interval: int,
    hyp: HyperparametersGAIL,
    solved_reward: float,
    random_seeds: List,
    max_timesteps: Optional[int] = None,
    render: bool = False,
    verbose: bool = False,
    ppo_type: str = "clip",
    advantage_type: str = "monte_carlo",
    log_level=logging.INFO,
    date: Optional[str] = None,
):
    env = gym.make(env_name).env
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    episode_numbers = []
    for random_seed in random_seeds:
        if random_seed is not None:
            torch.manual_seed(random_seed)
            env.seed(random_seed)
            print(f"Set random seed to: {random_seed}")

        buffer = ExperienceBuffer(state_dim, action_dim)

        hyp_str = generate_gail_str(ppo_type, hyp)
        save_path = generate_save_location(
            Path("data"),
            actor_layers,
            f"GAIL-{ppo_type}",
            env_name,
            random_seed,
            hyp_str,
            date,
        )
        save_path.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=f"{save_path}/log.log", level=log_level)

        gail_trainer = GAILTrainer(
            state_dimension=state_dim,
            action_space=action_dim,
            hyperparameters=hyp,
            save_path=save_path,
            demo_path=demo_path,
            actor_layers=actor_layers,
            critic_layers=critic_layers,
            discriminator_layers=discrim_layers,
            actor_activation=actor_activation,
            critic_activation=critic_activation,
            discriminator_activation=discrim_activation,
            entropy=True,
            ppo_type=ppo_type,
            advantage_type=advantage_type,
        )

        # logging variables
        running_reward = 0
        avg_length = 0
        timestep = 0

        # training loop
        for ep_num in range(1, max_episodes + 1):
            state = env.reset()
            ep_total_reward = 0
            for t in range(max_timesteps):
                timestep += 1

                # Running policy_old:
                action = ppo.policy_old.act(state, buffer)
                state, reward, done, _ = env.step(action)

                # Saving reward and is_terminal:
                buffer.rewards.append(reward)
                buffer.is_terminal.append(done)

                # update if its time
                if timestep % update_timestep == 0:
                    ppo.update(buffer)
                    buffer.clear()
                    timestep = 0

                running_reward += reward
                ep_total_reward += reward
                if render:
                    env.render()
                if done:
                    break

            avg_length += t

            ep_str = ("{0:0" + f"{len(str(max_episodes))}" + "d}").format(ep_num)
            if verbose:
                print(
                    f"Episode {ep_str} of {max_episodes}. \t Total reward = {ep_total_reward}"
                )

            # logging
            if ep_num % log_interval == 0:
                avg_length = int(avg_length / log_interval)
                running_reward = int((running_reward / log_interval))
                print(
                    f"Episode {ep_str} of {max_episodes}. \t Avg length: {avg_length} \t Reward: {running_reward}"
                )

                # stop training if avg_reward > solved_reward
                if running_reward > solved_reward:
                    print("########## Solved! ##########")
                    torch.save(ppo.policy.state_dict(), f"./PPO_{env_name}.pth")
                    break

                running_reward = 0
                avg_length = 0
        episode_numbers.append(ep_num)
    print(f"episode_numbers: {episode_numbers}")
    return episode_numbers


