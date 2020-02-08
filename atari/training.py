import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
import gym
import datetime

from algorithms.discrete_policy import DiscretePolicy
from algorithms.imitation_learning.behavioural_cloning import train_network
from algorithms.buffer import DemonstrationBuffer

from atari.checking import get_average_score
from atari.consts import GAME_NAMES, GAME_STRINGS_TEST


def demonstration_rewards():
    game_ref = 0
    env = gym.make(GAME_STRINGS_TEST[game_ref]).env

    demo_path = Path(f"expert_demos/{GAME_NAMES[game_ref]}/")
    demo_buffer = DemonstrationBuffer(
        demo_path, env.observation_space.shape, env.action_space.n
    )
    reward_list = []
    for demo_num in range(12):
        demo_buffer.load_demos(demo_num)
        rewards = demo_buffer.get_rewards()

        states, actions, _ = demo_buffer.recall_memory()
        print(len(states))
        #
        #     # for state, reward in zip(states, rewards):
        #         # print(reward)
        #     # state_count = 0
        #     # for action in actions:
        #     #     if action == 1:
        #     #         state_count += 1
        #     # print(state_count)
        reward_sum = np.sum(rewards)
        reward_list.append(reward_sum)
        demo_buffer.clear()
    # print(reward_list)
    # print(np.mean(reward_list))
    # print(np.mean(reward_list) * 25)


def train_bc():
    game_ref = 0
    # date = datetime.date.today().strftime("%d-%m-%Y")
    date = "03-02-2020"
    num_demos = [50, 50, 50, 50]
    # num_epochs = list(1000 // np.array(num_demos))
    num_epochs = [5, 10, 20, 40]
    minibatch_sizes = [16, 32, 32, 64]

    outer_outer_means = []

    # NN Architecture - hidden layer sizes
    hidden_layers_list = [(128, 128, 128, 128), (256, 256, 256), (128, 128, 128)]
    for hidden_layers in hidden_layers_list:
        architecture_str = ""
        for layer in hidden_layers:
            architecture_str += str(layer) + "-"
        save_location = Path(
            f"data/BC/{GAME_NAMES[game_ref]}/{date}/{architecture_str[:-1]}"
        )

        demo_path = Path(f"expert_demos/{GAME_NAMES[game_ref]}/")

        env = gym.make(GAME_STRINGS_TEST[game_ref]).env

        # logging.basicConfig(filename=f"{save_location}/{date}.log", level=logging.INFO)
        max_random_seed = 4
        outer_means = []
        for demo, epoch_num, minibatch_size in zip(
            num_demos, num_epochs, minibatch_sizes
        ):
            means = []
            for random_seed in range(max_random_seed):
                torch.manual_seed(random_seed)
                filename = f"seed_{random_seed}"
                file_save_location = (
                    save_location / f"{epoch_num}_epochs" / f"{demo}_demos"
                )
                file_save_location.mkdir(parents=True, exist_ok=True)

                train_network(
                    file_save_location,
                    filename,
                    num_demos=demo,
                    num_epochs=epoch_num,
                    minibatch_size=minibatch_size,
                    demo_path=demo_path,
                    action_space_size=env.action_space.n,
                    state_space_size=env.observation_space.shape,
                    hidden_layers=hidden_layers,
                    learning_rate=1e-7,
                    activation="relu",
                )

                network_load = Path(f"{save_location}/{filename}.pt")
                means.append(
                    get_average_score(
                        network_load=network_load,
                        env=env,
                        episode_timeout=10000,
                        show_solution=True,
                        num_trials=2,
                        hidden_layers=hidden_layers,
                        activation="relu",
                    )
                )
            outer_means.append(means)
        outer_outer_means.append(means)


def main():
    train_bc()
    # demonstration_rewards()


if __name__ == "__main__":
    main()
