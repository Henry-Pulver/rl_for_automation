import unittest
import numpy as np

from algorithms.trainer import RunLogger

from tools import captured_output


class RunLoggerTests(unittest.TestCase):
    def test_logger_legacy(self):
        log_type = "legacy"
        max_episodes = 100
        logger = RunLogger(max_episodes, log_type)
        num_eps = 5
        ep_lengths = np.random.randint(low=0, high=100, size=num_eps)
        reward_list = []
        for ep_length in ep_lengths:
            rewards = np.random.randint(low=0, high=100, size=ep_length)
            reward_list.append(np.sum(rewards))
            for reward in rewards:
                logger.update(1, reward)
            logger.end_episode()

        ep_num = np.random.randint(0, 100)
        with captured_output() as (out, err):
            logger.output_logs(ep_num, num_eps)
        output = out.getvalue().strip()
        ep_str = ("{0:0" + f"{len(str(max_episodes))}" + "d}").format(ep_num)
        self.assertEqual(
            output,
            f"Episode {ep_str} of {max_episodes}. \t Avg length: "
            f"{int(np.mean(ep_lengths))} \t Reward: {np.round(float(np.mean(reward_list)), 1)}",
        )

    def test_logger_moving_avg(self):
        log_type = "moving_avg"
        max_episodes = 100
        ma_factor = 0.95
        logger = RunLogger(max_episodes, log_type, ma_factor)
        num_eps = 5
        ep_lengths = np.random.randint(low=0, high=100, size=num_eps)
        reward_list = []
        for ep_length in ep_lengths:
            rewards = np.random.randint(low=0, high=100, size=ep_length)
            reward_list.append(np.sum(rewards))
            for reward in rewards:
                logger.update(1, reward)
            logger.end_episode()

        ep_num = np.random.randint(0, 100)
        ma_reward = reward_list[0]
        ma_length = ep_lengths[0]
        for count in range(1, num_eps):
            ma_reward = ma_factor * ma_reward + (1 - ma_factor) * reward_list[count]
            ma_length = ma_factor * ma_length + (1 - ma_factor) * ep_lengths[count]

        with captured_output() as (out, err):
            logger.output_logs(ep_num, num_eps)
        output = out.getvalue().strip()
        ep_str = ("{0:0" + f"{len(str(max_episodes))}" + "d}").format(ep_num)
        self.assertEqual(
            output,
            f"Episode {ep_str} of {max_episodes}. \t Avg length: {int(ma_length)} \t "
            f"Reward: {np.round(float(ma_reward), 1)}",
        )
