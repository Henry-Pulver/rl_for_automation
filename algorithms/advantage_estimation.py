from typing import Callable
import numpy as np

from algorithms.buffer import ExperienceBuffer


def get_advantage(experience_buffer: ExperienceBuffer, value_fn: Callable) -> np.array:
    """

    Args:
        experience_buffer: Experience from which to calculate the advantage.

    Returns:
        Advantage at every time
    """
    raise NotImplementedError


def get_td_error(values: np.array, buffer, gamma: float):
    rewards = np.array(buffer.rewards)
    if not values.shape == rewards.shape:
        print(f"values.shape: {values.shape}")
        print(f"values: {values}")
        print(f"rewards.shape: {rewards.shape}")
    assert values.shape == rewards.shape
    td_errors = []
    next_value = 0
    for reward, value, is_terminal in zip(
        reversed(buffer.rewards), reversed(values), reversed(buffer.is_terminal)
    ):
        if is_terminal:
            next_value = 0
        td_errors.insert(0, reward + (gamma * next_value) - value)
        next_value = value
    return td_errors


def get_gae(td_errors, is_terminals, gamma: float, lamda: float) -> np.array:
    """
    Class for implementing Generalised Advantage Estimation (https://arxiv.org/pdf/1506.02438.pdf)

    Args:
        td_errors: TD-errors for each timestep in batch.
        gamma: Return discount factor for future rewards. 0 < gamma < 1. Special cases:
               gamma = 1 gives undiscounted returns. gamma = 0 gives return = next
               reward.
        lamda: GAE TD-error discount factor. 0 < lamda < 1. Special cases: lamda = 1
                gives the true discounted return (by looking as far into the future as
                possible). lamda = 0 gives 1 step TD-error.
    """
    assert 0 <= gamma <= 1
    assert 0 <= lamda <= 1
    gae = []
    future_gen_adv = 0
    for td_error, is_terminal in zip(reversed(td_errors), reversed(is_terminals)):
        if is_terminal:
            future_gen_adv = 0
        future_gen_adv = td_error + (gamma * lamda * future_gen_adv)
        gae.insert(0, future_gen_adv)
    return np.array(gae)
