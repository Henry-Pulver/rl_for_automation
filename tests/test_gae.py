import unittest
import pytest
import numpy as np
from algorithms.advantage_estimation import get_gae, get_td_error
from algorithms.buffer import ExperienceBuffer


def test_td_error():
    values = [8, 8, 7, 6, 5, 4, 3, 2, 1]
    rewards = [2, 0, 4, 0, 0, 1, 4, 2, 0]
    gamma = 0.8
    td_error = get_td_error(values, rewards, gamma)
    true_td_error = [0.4, -2.4,  1.8, -2.0, -1.8, -0.6, 2.6, 0.8, -1.0]
    for step in range(len(td_error)):
        assert abs(td_error[step] - true_td_error[step]) < 1e-5


def test_gae():
    test_buffer = ExperienceBuffer()
    get_gae()
