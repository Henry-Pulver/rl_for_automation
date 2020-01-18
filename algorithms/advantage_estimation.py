from typing import Callable
import numpy as np
from algorithms.buffer import ExperienceBuffer
import copy


def get_advantage(experience_buffer: ExperienceBuffer) -> np.array:
    """

    Args:
        experience_buffer: Experience from which to calculate the advantage.

    Returns:
        Advantage at every time
    """
    pass


def get_gae(
    experience_buffer: ExperienceBuffer,
    value_fn: Callable,
    gamma: float,
    lamda: float,
    approximate: bool = False,
) -> np.array:
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
    time_horizon = (
        int(np.ceil(-3 / (np.log10(gamma * lamda))))
        if approximate
        else experience_buffer.get_length()
    )
    states, actions, action_probs = experience_buffer.recall_memory()
    rewards = experience_buffer.get_rewards()
    values = value_fn(states)

    def td_error():
        values_copy = copy.deepcopy(values)
        next_step_values = np.append(values_copy, 0)[1:]
        td_errors = rewards + gamma * next_step_values - values
        return td_errors
