import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.figure_factory as ff

from mountain_car.mountain_car_runner import CONSTS, DISC_CONSTS
from mountain_car.policy_iteration import policy_improvement


def load_data(load_path: Path) -> np.array:
    if load_path.exists():
        return np.load(str(load_path), allow_pickle=True)
    else:
        raise FileNotFoundError("load_path variable doesn't match a file that exists")


def show_contours(x_data: np.array, y_data: np.array, z_data: np.array, title: str = 'Mountain Car Value Function', filename: str = 'mountain_car_value.html'):
    fig = go.Figure(data=[go.Surface(x=x_data, y=y_data, z=z_data)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig.update_layout(title=title,
                      margin=dict(l=65, r=50, b=65, t=60),
                      scene=dict(
                          xaxis=dict(nticks=10),
                          yaxis=dict(nticks=10),
                          xaxis_title='Position',
                          yaxis_title='Velocity',
                          zaxis_title='Value',
                      ))
    fig.write_html(filename, auto_open=True)


# def show_quiver_plot():
#     pass


def show_policy(raw_policy: np.array):
    for count, item in enumerate(raw_policy):
        if item == [0, 1, 2]:
            raw_policy[count] = 0.5
        elif item == [0]:
            raw_policy[count] = -2
        elif item == [1]:
            raw_policy[count] = 0
        elif item == [2]:
            raw_policy[count] = 2
        elif item == [0, 1]:
            raw_policy[count] = -1
        elif item == [1, 2]:
            raw_policy[count] = 1

    return raw_policy.reshape((140, 200))


def main():
    value_fn_data = load_data(Path("mountain_car/value_fn/v140_x200.npy"))
    # _, policy = policy_improvement([ACTION_SPACE] * STATE_SPACE.shape[0], value_fn_data, save=True,
    #                                save_location="mountain_car/value_fn/policy_v140_x200.npy")
    policy = load_data(Path("mountain_car/value_fn/policy_v140_x200.npy"))
    policy = show_policy(policy)

    # value_fn_data = value_fn_data.reshape((140, 200))
    # print(STATE_SPACE)
    positions = np.array([state[0] for state in DISC_CONSTS.STATE_SPACE]).reshape((len(DISC_CONSTS.VELOCITY_VALUES), len(DISC_CONSTS.POSITION_VALUES)))
    # print(positions)
    velocities = np.array([state[1] for state in DISC_CONSTS.STATE_SPACE]).reshape((140, 200))
    # print(velocities[:100])
    # print(velocities)
    # show_contours(x_data=positions, y_data=velocities, z_data=value_fn_data)
    show_contours(x_data=positions, y_data=velocities, z_data=policy, title="Mountain Car Policy", filename="mountain_car_policy.html")


if __name__ == "__main__":
    main()
