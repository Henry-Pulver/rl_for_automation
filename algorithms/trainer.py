import numpy as np
import gym
import torch
from typing import Optional, List
from shutil import rmtree
from pathlib import Path
import datetime


class RunLogger:
    LOG_TYPES = ["legacy", "moving_avg"]

    def __init__(
        self,
        max_episodes: int,
        log_type: str = "legacy",
        moving_avg_factor: Optional[float] = None,
        verbose: bool = False,
    ):
        assert log_type in self.LOG_TYPES
        self.max_episodes = max_episodes
        self.log_type = log_type
        self.ep_length = 0
        self.ep_reward = 0
        self.verbose = verbose

        if log_type == "legacy":
            self.avg_ep_len = 0
            self.avg_reward = 0
            self.total_reward = 0
            self.total_ep_len = 0
        elif log_type == "moving_avg":
            self.avg_ep_len = None
            self.avg_reward = None
            assert 0 < moving_avg_factor < 1
            self.ma_factor = moving_avg_factor

    def update(self, num_steps: int, reward: float):
        self.ep_length += num_steps
        self.ep_reward += reward

    def end_episode(self, ep_num: int = 0):
        if self.log_type == "legacy":
            self.total_reward += self.ep_reward
            self.total_ep_len += self.ep_length
        elif self.log_type == "moving_avg":
            self.ma_update()
        if self.verbose:
            self.avg_reward = self.ep_reward
            self.avg_ep_len = self.ep_length
            self.print_logs(ep_num)
        self.ep_reward = 0
        self.ep_length = 0

    def print_logs(self, ep_num: int):
        ep_str = ("{0:0" + f"{len(str(self.max_episodes))}" + "d}").format(ep_num)
        print(
            f"Episode {ep_str} of {self.max_episodes}. \t Avg length: {int(self.avg_ep_len)}"
            f" \t Reward: {np.round(self.avg_reward, 1)}"
        )

    def output_logs(self, ep_num: int, log_interval: int):
        if self.log_type == "legacy":
            self.avg_reward = self.total_reward / log_interval
            self.avg_ep_len = self.total_ep_len / log_interval
            self.total_reward = 0
            self.total_ep_len = 0
        self.print_logs(ep_num)

    def ma_update(self):
        assert self.log_type == "moving_avg"
        if self.avg_reward is not None:
            self.avg_reward = self.avg_reward * self.ma_factor + self.ep_reward * (
                1 - self.ma_factor
            )
        else:
            self.avg_reward = self.ep_reward
        if self.avg_ep_len is not None:
            self.avg_ep_len = self.avg_ep_len * self.ma_factor + self.ep_length * (
                1 - self.ma_factor
            )
        else:
            self.avg_ep_len = self.ep_length


class Trainer:
    def __init__(
        self,
        env_name: str,
        save_base_path: Path,
        action_space: Optional[List] = None,
        date: Optional[str] = None,
    ):
        self.env_name = env_name
        self.env = gym.make(env_name).env
        self.state_dim = self.env.observation_space.shape
        self.action_dim = (
            self.env.action_space.n if action_space is None else len(action_space)
        )
        self.date = (
            date if date is not None else datetime.date.today().strftime("%d-%m-%Y")
        )
        self.save_base_path = save_base_path

    def set_seed(self, random_seed: Optional[int]):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            self.env.seed(random_seed)
            np.random.seed(random_seed)
            print(f"Set random seed to: {random_seed}")

    @staticmethod
    def restart(save_path: Path, restart: bool):
        if restart:
            if save_path.exists():
                print("Old data removed!")
                rmtree(save_path)
