import numpy as np
from algorithms.advantage_estimation import get_gae, get_td_error
from algorithms.buffer import ExperienceBuffer


def test_td_error():
    values = np.array([8, 8, 7, 6, 5, 4, 3, 2, 1])
    rewards = np.array([2, 0, 4, 0, 0, 1, 4, 2, 0])
    gamma = 0.8
    td_error = get_td_error(values, rewards, gamma)
    true_td_error = [0.4, -2.4, 1.8, -2.0, -1.8, -0.6, 2.6, 0.8, -1.0]
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
    gae = get_gae(
        experience_buffer=test_buffer, gamma=gamma, lamda=lamda, value_fn=value_fn
    )
    true_gae = [
        -0.81544699,
        -0.46163593,
        0.15369386,
        -1.41610334,
        -0.02516147,
        2.8044352,
        3.47568,
        0.712,
        -0.7,
    ]
    for step in range(len(true_gae)):
        assert abs(true_gae[step] - gae[step]) < 1e-5
