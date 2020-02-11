import numpy as np
import plotly.graph_objects as go
import os
from typing import List
from pathlib import Path

from algorithms.utils import generate_save_location
from mountain_car.REINFORCE_next_states import (
    FEATURE_POLYNOMIAL_ORDER as state_poly_order,
)
from mountain_car.REINFORCE_actions import FEATURE_POLYNOMIAL_ORDER as action_poly_order


def plot(load_path: Path, files: List, len_vector: List):
    for count, file in enumerate(files):
        fig = go.Figure()
        y = np.load(f"{load_path}/{file}", allow_pickle=True).T
        if not len_vector[count] == 1:
            y = y.reshape((-1, len_vector[count])).T
            x = np.linspace(0, y.shape[1], y.shape[1] + 1)
            for theta in y:
                fig.add_trace(go.Scatter(x=x, y=theta))
        else:
            x = np.linspace(0, y.shape[0], y.shape[0] + 1)
            fig.add_trace(go.Scatter(x=x, y=y))
        fig.write_html(f"{load_path}/{file}.html", auto_open=True,)


def plot_reinforce_concatenated_weights_and_performance(
    ref_num_list: List,
    opt: bool,
    feature_vector_size: int,
    action_feature: bool,
    baseline: bool,
) -> None:

    REINFORCE_type = "REINFORCE_actions" if action_feature else "REINFORCE_states"
    path = f"mountain_car/{REINFORCE_type}/plots/concatenated/"
    opt_str = "optuna_" if opt else ""
    action_feature_multiplier = 3 if action_feature else 1

    if baseline:
        fig = go.Figure()
        y = np.load(
            f"{path}/{opt_str}baseline_plot_{ref_num_list[0]}-{ref_num_list[1]}.npy"
        ).T
        y = y.reshape((-1, feature_vector_size)).T
        x = np.linspace(0, y.shape[1], y.shape[1] + 1)
        for theta in y:
            fig.add_trace(go.Scatter(x=x, y=theta))
        fig.write_html(
            f"{path}/{opt_str}baseline_plot_{ref_num_list[0]}-{ref_num_list[1]}.html",
            auto_open=True,
        )

        fig = go.Figure()
        y = np.load(
            f"{path}/{opt_str}avg_delta_plot_{ref_num_list[0]}-{ref_num_list[1]}.npy"
        )
        x = np.linspace(0, y.shape[0], y.shape[0] + 1)
        fig.add_trace(go.Scatter(x=x, y=y))
        fig.write_html(
            f"{path}/{opt_str}avg_delta_plot_{ref_num_list[0]}-{ref_num_list[1]}.html",
            auto_open=True,
        )

    fig = go.Figure()
    y = np.load(
        f"{path}/{opt_str}policy_plot_{ref_num_list[0]}-{ref_num_list[1]}.npy"
    ).T
    y = y.reshape((-1, feature_vector_size * action_feature_multiplier)).T
    x = np.linspace(0, y.shape[1], y.shape[1] + 1)
    for theta in y:
        fig.add_trace(go.Scatter(x=x, y=theta))
    fig.write_html(
        f"{path}/{opt_str}policy_plot_{ref_num_list[0]}-{ref_num_list[1]}.html",
        auto_open=True,
    )

    fig = go.Figure()
    y = np.load(f"{path}/{opt_str}moving_avg_{ref_num_list[0]}-{ref_num_list[1]}.npy").T
    x = np.linspace(0, y.shape[0], y.shape[0] + 1)
    fig.add_trace(go.Scatter(x=x, y=y))
    fig.write_html(
        f"{path}/{opt_str}moving_avg_{ref_num_list[0]}-{ref_num_list[1]}.html",
        auto_open=True,
    )

    fig = go.Figure()
    y = np.load(f"{path}/{opt_str}returns_{ref_num_list[0]}-{ref_num_list[1]}.npy").T
    x = np.linspace(0, y.shape[0], y.shape[0] + 1)
    fig.add_trace(go.Scatter(x=x, y=y))
    fig.write_html(
        f"{path}/{opt_str}returns_{ref_num_list[0]}-{ref_num_list[1]}.html",
        auto_open=True,
    )


def plot_reinforce_weights_and_performance(
    ref_num: int,
    opt: bool,
    feature_vector_size: int,
    action_feature: bool,
    baseline: bool,
) -> None:
    REINFORCE_type = "REINFORCE_actions" if action_feature else "REINFORCE_states"
    path = f"mountain_car/{REINFORCE_type}/plots/{ref_num}"
    opt_str = "optuna_" if opt else ""
    action_feature_multiplier = 3 if action_feature else 1

    if baseline:
        fig = go.Figure()
        y = np.load(f"{path}/{opt_str}baseline_plot_{ref_num}.npy").T
        y = y.reshape((-1, feature_vector_size)).T
        x = np.linspace(0, y.shape[1], y.shape[1] + 1)
        for count, theta in enumerate(y):
            fig.add_trace(go.Scatter(x=x, y=theta, name=f"theta_{count}"))
        fig.write_html(f"{path}/{opt_str}baseline_plot_{ref_num}.html", auto_open=True)

        fig = go.Figure()
        y = np.load(f"{path}/{opt_str}avg_delta_plot_{ref_num}.npy")
        x = np.linspace(0, y.shape[0], y.shape[0] + 1)
        fig.add_trace(go.Scatter(x=x, y=y))
        fig.write_html(f"{path}/{opt_str}avg_delta_plot_{ref_num}.html", auto_open=True)

    fig = go.Figure()
    y = np.load(f"{path}/{opt_str}policy_plot_{ref_num}.npy").T
    y = y.reshape((-1, feature_vector_size * action_feature_multiplier)).T
    x = np.linspace(0, y.shape[1], y.shape[1] + 1)
    for count, theta in enumerate(y):
        fig.add_trace(go.Scatter(x=x, y=theta, name=f"theta_{count}"))
    fig.write_html(f"{path}/{opt_str}policy_plot_{ref_num}.html", auto_open=True)

    fig = go.Figure()
    y = np.load(f"{path}/{opt_str}moving_avg_{ref_num}.npy").T
    x = np.linspace(0, y.shape[0], y.shape[0] + 1)
    fig.add_trace(go.Scatter(x=x, y=y))
    fig.write_html(f"{path}/{opt_str}moving_avg_{ref_num}.html", auto_open=True)

    fig = go.Figure()
    y = np.load(f"{path}/{opt_str}returns_{ref_num}.npy").T
    x = np.linspace(0, y.shape[0], y.shape[0] + 1)
    fig.add_trace(go.Scatter(x=x, y=y))
    fig.write_html(f"{path}/{opt_str}returns_{ref_num}.html", auto_open=True)


def plot_run(ref_num: int, trial_num: int) -> None:
    root_dir = f"mountain_car/REINFORCE_states/plots/{ref_num}/{trial_num}"
    files = os.listdir(root_dir)
    for file in files:
        if not file.endswith(".npy"):
            pass
        else:
            fig = go.Figure()
            y = np.load(f"{root_dir}/{file}", allow_pickle=True)
            if len(y.shape) > 1:
                y = y.T
                x = np.linspace(0, y.shape[1], y.shape[1] + 1)
                for theta in y:
                    fig.add_trace(go.Scatter(x=x, y=theta,))
            elif y.shape[0] > 50000:
                y = y.reshape((10000, 9)).T
                x = np.linspace(0, y.shape[1], y.shape[1] + 1)
                for count, theta in enumerate(y):
                    fig.add_trace(go.Scatter(x=x, y=theta, name=f"theta_{count}"))
            else:
                x = np.linspace(0, y.shape[0], y.shape[0] + 1)
                fig.add_trace(go.Scatter(x=x, y=y))
            file = os.path.splitext(file)[0]
            fig.write_html(f"{root_dir}/{file}.html", auto_open=True)


def main():
    load_path = generate_save_location(
        Path("../mountain_car/data"),
        algo="PPO",
        env_name="MountainCar-v0",
        nn_layers=(32, 32),
        seed=1,
    ) / "3-epochs"
    files = ["mean_clipped_loss.npy", "mean_entropy_loss.npy", "mean_value_loss.npy", "policy_params.npy", "critic_params.npy", "returns.npy"]
    # files = ["episode_lengths.npy", "policy_params.npy"]
    len_vector = [1, 1, 1, 2, 2, 1]
    # len_vector = [1, 2]
    plot(load_path, files, len_vector)


    # plot_reinforce_weights_and_performance(
    #     ref_num=51,
    #     opt=False,
    #     feature_vector_size=(action_poly_order + 1) ** 2,
    #     action_feature=False,
    #     baseline=False,
    # )

    # plot_reinforce_concatenated_weights_and_performance(
    #     ref_num_list=[40, 41],
    #     opt=False,
    #     feature_vector_size=(action_poly_order + 1) ** 2,
    #     action_feature=False,
    #     baseline=True
    # )

    # plot_run(ref_num=20, trial_num=800)


if __name__ == "__main__":
    main()
