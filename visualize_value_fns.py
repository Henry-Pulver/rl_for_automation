import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.figure_factory as ff

from mountain_car.value_iteration import STATE_SPACE


def load_data(load_path: Path) -> np.array:
    if load_path.exists():
        return np.load(str(load_path), allow_pickle=True)
    else:
        raise FileNotFoundError("load_path variable doesn't match a file that exists")


def show_contours(x_data: np.array, y_data: np.array, z_data: np.array):
    fig = go.Figure(data=[go.Surface(x=x_data, y=y_data, z=z_data)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig.update_layout(title='Mountain Car Value Function',
                      margin=dict(l=65, r=50, b=65, t=60),
                      scene=dict(
                          xaxis=dict(nticks=10),
                          yaxis=dict(nticks=10),
                          xaxis_title='Position',
                          yaxis_title='Velocity',
                          zaxis_title='Value',
                      ))
    fig.write_html('mountain_car_value.html', auto_open=True)


def show_quiver_plot():
    pass


# def show_policy(value_fn: np.array, action_space: np.array):



def main():
    value_fn_data = load_data(Path("mountain_car/value_fn/v140_x200.npy"))
    value_fn_data = value_fn_data.reshape((140, 200))
    # print(STATE_SPACE)
    positions = np.array([state[0] for state in STATE_SPACE]).reshape((140, 200))
    # print(positions)
    velocities = np.array([state[1] for state in STATE_SPACE]).reshape((140, 200))
    # print(velocities[:100])
    # print(velocities)
    show_contours(x_data=positions, y_data=velocities, z_data=value_fn_data)


if __name__ == "__main__":
    main()
