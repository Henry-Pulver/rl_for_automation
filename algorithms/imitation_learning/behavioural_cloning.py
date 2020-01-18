import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from algorithms.discrete_policy import DiscretePolicy
from algorithms.buffer import DemonstrationBuffer
from typing import Tuple


class BCTrainer:
    def __init__(
        self,
        demo_path: Path,
        learning_rate: float,
        state_space_size: int,
        action_space_size: int,
        hidden_layers: Tuple = (32, 32),
    ):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.network = DiscretePolicy(
            state_dimension=state_space_size,
            action_space=action_space_size,
            hidden_layers=hidden_layers,
        ).float()

        self.demo_buffer = DemonstrationBuffer(
            demo_path, state_space_size, action_space_size
        )

        self.criterion = nn.CrossEntropyLoss()

        # zero the parameter gradients
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()

    def train_network(self, num_epochs: int, num_demos: int):
        state_array = np.array([])
        action_array = np.array([])
        state_array = state_array.reshape((-1, self.state_space_size))
        for demo_num in range(num_demos):
            self.demo_buffer.load_demos(demo_num)
            states, actions, _ = self.demo_buffer.recall_memory()
            state_array = np.append(state_array, states)
            action_array = np.append(action_array, actions)
        state_array = state_array.reshape((-1, self.state_space_size))
        states = torch.from_numpy(state_array)
        actions = torch.from_numpy(action_array)

        for epoch in range(num_epochs):
            print(f"Epoch number: {epoch + 1}")
            sum_loss, num_steps = 0, 0
            for state, action in zip(states, actions):
                # forward + backward + optimize
                action_probs = self.network(state.float()).unsqueeze(dim=0)
                action_probs = action_probs.float()
                action = action.unsqueeze(dim=0)
                action = action.type(torch.long)

                loss = self.criterion(action_probs, action)
                loss.backward()
                self.optimizer.step()
                num_steps += 1
                sum_loss += loss
            print(f"Avg loss: {sum_loss / num_steps}")


def train_network(
    save_location: Path,
    network_filename: str,
    num_demos: int,
    num_epochs: int,
    action_space_size: int,
    state_space_size: int,
    learning_rate: float = 1e-5,
):
    trainer = BCTrainer(
        demo_path=Path("../expert_demos/"),
        learning_rate=learning_rate,
        action_space_size=action_space_size,
        state_space_size=state_space_size,
    )
    trainer.train_network(num_epochs=num_epochs, num_demos=num_demos)
    save_location.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.network.state_dict(), f"{save_location}/{network_filename}.pt")


def get_action_probs(state, network):
    state = torch.from_numpy(state)
    return network(state.float()).detach().numpy()


def pick_action(state, network):
    action_probs = get_action_probs(state, network)
    chosen_action = np.array([np.random.choice(action_probs, p=action_probs)])
    return chosen_action
