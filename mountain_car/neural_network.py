import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from buffer import DemonstrationBuffer


class MountainCarNeuralNetwork(nn.Module):
    def __init__(self):
        super(MountainCarNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        # x = self.fc2(x)
        x = nn.functional.softmax(self.fc3(x))
        # print(x.shape)
        # _, x = torch.max(x)  # , dim=1)
        # print(x.shape)
        return x


def run_epoch(num_demos: int):
    demo_buffer = DemonstrationBuffer(Path("expert_demos/"))
    network = MountainCarNeuralNetwork(learning_rate=1e-2)
    state_array = np.array([])
    action_array = np.array([])

    # zero the parameter gradients
    network.optimizer.zero_grad()

    # forward + backward + optimize
    outputs = network(state_array)
    loss = network.criterion(outputs, action_array)
    loss.backward()
    optimizer.step()
    for demo_num in range(num_demos):
        demo_buffer.load_demos(demo_num)
        states, actions, _ = demo_buffer.recall_memory()
        state_array = np.append(state_array, states)
        action_array = np.append(action_array, actions)
    state_array = state_array.reshape((-1, 2))
    action_array = to_categorical(action_array)
    # print(state_array.shape)
    # print(action_array.shape)
    network.model.fit(state_array, action_array, epochs=5)

# run_epoch(20)
