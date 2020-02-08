import numpy as np
from pathlib import Path
import torch
from typing import Tuple
from collections import namedtuple

from algorithms.buffer import DemonstrationBuffer


HyperparametersDQfD = namedtuple(
    "HyperparametersDQfD",
    ("pre_training_updates", "network_update_freq", "minibatch_size"),
)
"""
    pre_training_updates: Number of steps to train purely on demonstrations.
    network_update_freq: Frequency at which to update target net.
    minibatch_size: Size of mini-batch samples.
"""


class QLearningFromDemos:
    def __init__(
        self,
        state_dimension: Tuple,
        action_space: int,
        hyperparameters: namedtuple,
        demo_path: Path,
        random_seed: int,
        data_save_location: Path,
        nn_save_location: Path,
        network_layers: Tuple,
        network_activation: str,
    ):
        self.hyp = hyperparameters
        self.data_save_path = data_save_location
        self.nn_save_location = nn_save_location

        # Set random seed
        torch.manual_seed(random_seed)

        self.demo_buffer = DemonstrationBuffer(demo_path, state_dimension, action_space)
        self.experience_buffer =

    def train(self, num_demos: int, num_epochs: int):
        # Load up the right number of demos
        for demo_num in range(num_demos):
            self.demo_buffer.load_demos(demo_num)

        # Phase 1
        for step in range(self.hyp.pre_training_updates):
            if step % self.hyp.network_update_frequency == 0:
                pass


        # Phase 2
        for epoch in range(num_epochs):
            pass
