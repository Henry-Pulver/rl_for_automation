import numpy as np
from algorithms.advantage_estimation import get_gae, get_td_error
from algorithms.buffer import PPOExperienceBuffer


def test_td_error():
    values = np.array([8, 8, 7, 6, 5, 4, 3, 2, 1])
    rewards = np.array([2, 0, 4, 0, 0, 1, 4, 2, 0])
    is_terminal = np.array([False,] * 9)
    buffer = PPOExperienceBuffer(action_space_size=3, state_dimension=(2,))
    buffer.is_terminal = is_terminal
    buffer.rewards = rewards
    gamma = 0.8
    td_error = get_td_error(values, buffer, gamma)
    true_td_error = [0.4, -2.4, 1.8, -2.0, -1.8, -0.6, 2.6, 0.8, -1.0]
    for step in range(len(td_error)):
        assert abs(td_error[step] - true_td_error[step]) < 1e-5

    # Keep values and rewards the same, make both reward = 4 the end of the episode
    is_terminal[2] = True
    is_terminal[6] = True
    buffer.is_terminal = is_terminal
    td_error = get_td_error(values, buffer, gamma)
    true_td_error = [0.4, -2.4, -3, -2.0, -1.8, -0.6, 1, 0.8, -1.0]
    for step in range(len(td_error)):
        assert abs(td_error[step] - true_td_error[step]) < 1e-5


def test_gae():
    test_buffer = PPOExperienceBuffer(state_dimension=(1,), action_space_size=2)
    states = np.array([-10, -8, -9, -6, -5, -3, -3, -2, -1])
    rewards = np.array([2, 0, 4, 0, 0, 1, 4, 2, 0])
    is_terminals = np.array([False,] * 9)
    test_buffer.rewards = rewards
    test_buffer.is_terminal = is_terminals
    # actions = [1, 1, 0, 1, 1, 0, 0, 0, 1]
    # for state, action, reward in zip(states, actions, rewards):
    #     test_buffer.update(state=state, action=action, reward=reward)
    gamma = 0.8
    lamda = 0.8
    value_fn = lambda x: np.array(-0.7 * x).reshape(-1)
    td_errors = get_td_error(value_fn(states), test_buffer, gamma)
    gae = get_gae(
        td_errors=td_errors, is_terminals=is_terminals, gamma=gamma, lamda=lamda
    )
    true_gae = [
        -0.8154469945062197,
        -0.4616359289159674,
        0.1536938610688,
        -1.41610334208,
        -0.025161472,
        2.8044352,
        3.47568,
        0.712,
        -0.7,
    ]
    for true_value, gae_value in zip(true_gae, gae):
        assert abs(true_value - gae_value) < 1e-5
