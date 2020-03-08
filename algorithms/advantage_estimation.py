from typing import Callable
import numpy as np
import torch
from algorithms.buffer import ExperienceBuffer
import copy


def get_advantage(experience_buffer: ExperienceBuffer, value_fn: Callable) -> np.array:
    """

    Args:
        experience_buffer: Experience from which to calculate the advantage.

    Returns:
        Advantage at every time
    """
    raise NotImplementedError


def get_td_error(values: np.array, rewards: np.array, gamma: float):
    if not values.shape == rewards.shape:
        print(f"values.shape: {values.shape}")
        print(f"values: {values}")
        print(f"rewards.shape: {rewards.shape}")
    assert values.shape == rewards.shape
    values_copy = copy.deepcopy(values)
    next_step_values = np.append(values_copy, values_copy[-1])[1:]
    td_errors = rewards + gamma * next_step_values - values.reshape((-1))
    return td_errors


def get_gae(td_errors, gamma: float, lamda: float) -> np.array:
    """
    Class for implementing Generalised Advantage Estimation (https://arxiv.org/pdf/1506.02438.pdf)

    Args:
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
    for td_error in reversed(td_errors):
        future_gen_adv = td_error + (gamma * lamda * future_gen_adv)
        gae.insert(0, future_gen_adv)
    return np.array(gae)
