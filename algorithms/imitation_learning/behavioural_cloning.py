import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Tuple
import logging

from algorithms.discrete_policy import DiscretePolicy
from algorithms.buffer import DemonstrationBuffer


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


class BCTrainer:
    def __init__(
        self,
        demo_path: Path,
        learning_rate: float,
        state_space_size: Tuple,
        action_space_size: int,
        hidden_layers: Tuple,
        activation: str,
    ):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.network = DiscretePolicy(
            state_dimension=state_space_size,
            action_space=action_space_size,
            hidden_layers=hidden_layers,
            activation=activation,
        ).float()

        self.demo_buffer = DemonstrationBuffer(
            demo_path, state_space_size, action_space_size
        )

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

        # zero the parameter gradients
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()

    def train_network(self, num_epochs: int, num_demos: int, minibatch_size: int):
        avg_loss_plot = []
        for epoch in range(num_epochs):
            sum_loss, num_steps = 0, 0
            demo_list = np.array(range(num_demos))
            np.random.shuffle(demo_list)
            for demo_num in demo_list:
                self.demo_buffer.load_demos(demo_num)
                self.demo_buffer.to_numpy()

                while self.demo_buffer.get_length() > minibatch_size:
                    sampled_states, sampled_actions, _ = self.demo_buffer.sample(
                        minibatch_size
                    )
                    states = torch.from_numpy(sampled_states).to(device)
                    actions = torch.from_numpy(sampled_actions).to(device)
                    # forward + backward + optimize
                    action_probs = self.network(states.float())
                    action_probs = action_probs.float()
                    actions = actions.type(torch.long)

                    loss = self.loss_fn(action_probs, actions)
                    loss.backward()
                    self.optimizer.step()
                    num_steps += 1
                    sum_loss += loss
                self.demo_buffer.from_numpy()
            avg_loss = sum_loss / num_steps
            # logging.debug(f"Epoch number: {epoch + 1}\tAvg loss: {avg_loss}")
            print(
                f"Epoch number: {epoch + 1}\tAvg loss: {avg_loss}\t Num steps: {num_steps}"
            )
            avg_loss_plot.append(avg_loss)
            if len(avg_loss_plot) > 10:
                if (
                    abs(avg_loss - avg_loss_plot[0]) < 1e-9
                    and abs(avg_loss - avg_loss_plot[-3]) < 1e-9
                    and abs(avg_loss - avg_loss_plot[-1]) < 1e-9
                ):
                    # logging.info(
                    #     f"\n\nAvg loss hasn't changed for 10 iterations.\n Skipping to next seed.\n"
                    # )
                    print(
                        f"\n\nAvg loss hasn't changed for 10 iterations.\n Skipping to next seed.\n"
                    )
                    break


def train_network(
    save_location: Path,
    network_filename: str,
    num_demos: int,
    minibatch_size: int,
    num_epochs: int,
    demo_path: Path,
    action_space_size: int,
    state_space_size: Tuple,
    hidden_layers: Tuple,
    learning_rate: float,
    activation: str,
):
    trainer = BCTrainer(
        demo_path=demo_path,
        learning_rate=learning_rate,
        action_space_size=action_space_size,
        state_space_size=state_space_size,
        hidden_layers=hidden_layers,
        activation=activation,
    )
    trainer.train_network(
        num_epochs=num_epochs, num_demos=num_demos, minibatch_size=minibatch_size
    )
    save_location.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.network.state_dict(), f"{save_location}/{network_filename}.pt")


def get_action_probs(state, network):
    state = torch.from_numpy(state)
    return network(state.float()).detach().numpy()


def pick_action(state, network):
    action_probs = get_action_probs(state, network)
    chosen_action = np.array(
        [np.random.choice(range(len(action_probs)), p=action_probs)]
    )
    return chosen_action
