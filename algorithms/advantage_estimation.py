from typing import Callable
import numpy as np
from algorithms.buffer import ExperienceBuffer
import copy


def get_advantage(experience_buffer: ExperienceBuffer, value_fn: Callable) -> np.array:
    """

    Args:
        experience_buffer: Experience from which to calculate the advantage.

    Returns:
        Advantage at every time
    """
    pass


def get_td_error(values: np.array, rewards: np.array, gamma: float):
    assert values.shape == rewards.shape
    values_copy = copy.deepcopy(values)
    next_step_values = np.append(values_copy, 0)[1:]
    td_errors = rewards + gamma * next_step_values - values.reshape((-1))
    return td_errors


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
    gamlam = gamma * lamda
    time_horizon = (
        int(np.ceil(-3 / (np.log10(gamlam))))
        if approximate
        else experience_buffer.get_length()
    )
    states, _, _ = experience_buffer.recall_memory()
    rewards = experience_buffer.get_rewards()
    values = value_fn(states)
    td_error = get_td_error(values, rewards, gamma)

    gae = []
    gamma_lamdas = np.logspace(
        start=0, stop=np.log10(gamlam ** (time_horizon - 1)), num=time_horizon
    )
    for step in range(time_horizon):
        if len(td_error[step:]) >= time_horizon:
            gae.append(np.dot(gamma_lamdas, td_error[step : step + time_horizon]))
        else:
            gae.append(np.dot(gamma_lamdas[: len(td_error[step:])], td_error[step:]))
        # if step == 0:
        #     gae.append(np.dot(gamma_lamdas, td_error[:time_horizon]))
        # else:
        #     gae.append((gae[-1] - td_error[step - 1]) / gamlam)
        # gae.append(((gae[-1] - td_error[step - 1]) / gamlam) + gamma_lamdas[-1] * td_error[step+time_horizon])
    return np.array(gae)
