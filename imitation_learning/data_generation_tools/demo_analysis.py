import os
import numpy as np
import gym
from pathlib import Path

from buffer import DemonstrationBuffer

from envs.atari.consts import GAME_STRINGS_LEARN


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
        reward_sum = np.sum(demo_buffer.get_rewards())
        reward_list.append(reward_sum)
        demo_buffer.clear()

    print(f"\nEnv name: {env_name}\nNumber of demos: {num_demos}")
    print(f"Demo path: {demo_path}\nScores: {reward_list}")
    print(
        f"Score mean: {np.mean(reward_list)}\nScore std dev: {np.sqrt(np.var(reward_list))}"
    )


def main():
    # env_name = "Breakout-ram-v4"
    # n = 2
    env_names = ["MountainCar-v0", "CartPole-v1", "Acrobot-v1"]
    # for n in range(4, 5):
    #     env_name = GAME_STRINGS_LEARN[n]
    # env_name = env_names[n]
    for env_name in env_names:
        demo_path = Path("../expert_demos")
        output_demo_data(env_name, demo_path)


if __name__ == "__main__":
    main()
