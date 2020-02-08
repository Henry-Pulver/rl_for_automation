import os
import numpy as np
from pathlib import Path
import torch
from algorithms.buffer import DemonstrationBuffer
from typing import Tuple


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
    k = 2048
    demo_path = Path("imitation/expert_demos")
    random_seed = 0
    network_update_freq = 32
    minibatch_size = 32
    train_deep_q_from_demonstrations(
        k, demo_path, random_seed, network_update_freq, minibatch_size
    )


if __name__ == "__main__":
    main()
