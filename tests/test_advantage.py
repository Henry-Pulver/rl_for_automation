import numpy as np
from algorithms.advantage_estimation import get_gae, get_td_error
from algorithms.buffer import ExperienceBuffer


def test_td_error():
    values = np.array([8, 8, 7, 6, 5, 4, 3, 2, 1])
    rewards = np.array([2, 0, 4, 0, 0, 1, 4, 2, 0])
    gamma = 0.8
    td_error = get_td_error(values, rewards, gamma)
    true_td_error = [0.4, -2.4, 1.8, -2.0, -1.8, -0.6, 2.6, 0.8, -0.2]
    for step in range(len(td_error)):
        assert abs(td_error[step] - true_td_error[step]) < 1e-5


def test_gae():
    test_buffer = ExperienceBuffer(state_dimension=(1,), action_space_size=2)
    states = np.array([-10, -8, -9, -6, -5, -3, -3, -2, -1])
    rewards = np.array([2, 0, 4, 0, 0, 1, 4, 2, 0])
    actions = [1, 1, 0, 1, 1, 0, 0, 0, 1]
    for state, action, reward in zip(states, actions, rewards):
        test_buffer.update(state=state, action=action, reward=reward)
    gamma = 0.8
    lamda = 0.8
    value_fn = lambda x: np.array(-0.7 * x).reshape(-1)
    td_errors = get_td_error(value_fn(states), rewards, gamma)
    gae = get_gae(td_errors=td_errors, gamma=gamma, lamda=lamda)
    true_gae = [
        -0.7996843958104229,
        -0.437006868,
        0.1921767680409604,
        -1.3559738,
        0.0687909376,
        2.95123584,
        3.705056,
        1.0704,
        -0.14,
    ]
    for true_value, gae_value in zip(true_gae, gae):
        assert abs(true_value - gae_value) < 1e-5
