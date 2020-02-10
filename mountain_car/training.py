import numpy as np
import torch
from pathlib import Path
from typing import Tuple

from algorithms.buffer import DemonstrationBuffer
from algorithms.utils import generate_save_location
from algorithms.PPO import HyperparametersPPO, train_ppo_network


def train_ppo():
    env_str = "MountainCar-v0"
    random_seed = 0
    actor_nn_layers = (32, 32)
    actor_activation = "relu"
    critic_nn_layers = (16, 16)
    critic_activation = "relu"
    max_episodes = 10000
    episode_length = 10000
    save_path = generate_save_location(
        Path("data"), actor_nn_layers, "PPO", env_str, random_seed
    )
    hyp = HyperparametersPPO(
        gamma=0.99,
        lamda=0.95,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        T=512,
        epsilon=0.2,
        c2=0.01,
        num_epochs=3,
    )

    train_ppo_network(
        env_str=env_str,
        random_seed=random_seed,
        actor_nn_layers=actor_nn_layers,
        save_path=save_path,
        hyp=hyp,
        actor_activation=actor_activation,
        critic_layers=critic_nn_layers,
        critic_activation=critic_activation,
        max_episodes=max_episodes,
        max_episode_length=episode_length,
    )


def train_deep_q_from_demonstrations(
    k: int,
    demo_path: Path,
    random_seed: int,
    network_update_freq: int,
    minibatch_size: int,
    num_demos: int,
    state_dimension: Tuple,
    action_space_size: int,
    num_epochs: int,
):
    ########## DQfD Hyperparams #########
    # k = 2048
    # demo_path = Path("imitation/expert_demos")
    # random_seed = 0
    # network_update_freq = 32
    # minibatch_size = 32
    # train_deep_q_from_demonstrations(
    #     k, demo_path, random_seed, network_update_freq, minibatch_size
    # )

    # Set random seed
    torch.manual_seed(random_seed)

    # Load in demos
    demo_buffer = DemonstrationBuffer(demo_path, state_dimension, action_space_size)
    for demo_num in range(num_demos):
        demo_buffer.load_demos(demo_num)

    # Phase 1
    for step in range(k):
        pass

    # Phase 2
    for epoch in range(num_epochs):
        pass


def main():
    train_ppo()


if __name__ == "__main__":
    main()
