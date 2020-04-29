import os
import numpy as np
import gym
from pathlib import Path

from buffer import DemonstrationBuffer


def output_demo_data(env_name: str, demo_path: Path):
    """
    Prints out the rewards from all expert demonstrations at `demo_path`/env_name.

    Args:
        env_name: Name of environment
        demo_path: Path to directory containing
    """
    env = gym.make(env_name).env
    demo_path = demo_path / env_name

    demo_buffer = DemonstrationBuffer(
        demo_path, env.observation_space.shape, env.action_space.n
    )
    num_demos = len(os.listdir(f"{demo_path}"))

    reward_list = []
    for demo_num in range(num_demos):
        demo_buffer.load_demo(demo_num)
        rewards = demo_buffer.get_rewards()

        states, actions, _ = demo_buffer.recall_memory()
        reward_sum = np.sum(rewards)
        reward_list.append(reward_sum)
        demo_buffer.clear()
    print(reward_list)
    print(np.mean(reward_list))
    print(f"\nDemo path: {demo_path}\n Rewards: {reward_list}\nReward mean: {np.mean(reward_list)}")
    print(f"{np.sqrt(np.var(reward_list))}")


def main():
    env_name = "Breakout-ram-v4"
    demo_path = Path("../expert_demos")
    output_demo_data(env_name, demo_path)


if __name__ == "__main__":
    main()
