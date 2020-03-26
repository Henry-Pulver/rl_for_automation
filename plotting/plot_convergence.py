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


def print_counts(load_path: Path, counts: List):
    for count in counts:
        print(f"{count}: {np.load(f'{load_path}/{count}.npy')}")


def plot(
    load_path: Path,
    plots: List,
    single_plots: List,
    min_score: float,
    reward_smoothing_weight: float,
):
    for single_plot, plot in zip(single_plots, plots):
        fig = go.Figure()
        y = np.load(f"{load_path}/{plot}/{plot}.npy", allow_pickle=True).T
        if not single_plot:
            x = np.linspace(0, y.shape[1], y.shape[1] + 1)
            for theta in y:
                fig.add_trace(go.Scatter(x=x, y=theta))
        else:
            x = np.linspace(0, y.shape[0], y.shape[0] + 1)
            fig.add_trace(go.Scatter(x=x, y=y))
        fig.write_html(
            f"{load_path}/{plot}.html", auto_open=True,
        )
        if plot == "rewards.":
            fig2 = go.Figure()
            x = np.linspace(0, y.shape[0], y.shape[0] + 1)
            y_smoothed = smooth_rewards(y, reward_smoothing_weight, min_score)
            fig2.add_trace(go.Scatter(x=x, y=y_smoothed))
            fig2.write_html(
                f"{load_path}/rewards/smoothed_rewards.html", auto_open=True,
            )


def smooth_rewards(
    rewards: np.ndarray, smoothing_weight: float, min_score: float
) -> np.ndarray:
    assert 0 < smoothing_weight < 1
    smoothed_rewards = np.zeros(rewards.shape)
    prev_reward = min_score
    for count, reward in enumerate(rewards):
        smoothed_rewards[count] = (
            smoothing_weight * prev_reward + (1 - smoothing_weight) * reward
        )
        prev_reward = smoothed_rewards[count]
    return smoothed_rewards


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
    #### PPO 1 ###
    # load_path = (
    #     generate_save_location(
    #         Path("../mountain_car/data"),
    #         algo="PPO",
    #         env_name="MountainCar-v0",
    #         nn_layers=(32, 32),
    #         seed=0,
    #     )
    #     / "3-epochs"
    # )
    # files = [
    #     "mean_clipped_loss.npy",
    #     "mean_entropy_loss.npy",
    #     "mean_value_loss.npy",
    #     "policy_params.npy",
    #     "critic_params.npy",
    #     "returns.npy",
    # ]
    # len_vector = [1, 1, 1, 2, 2, 1]

    #### PPO 2 ###
    # load_path = Path("../algorithms/data/PPO-clip/Acrobot-v1/12-03-2020/hyp-0.2/32-32/seed-0/")

    # files = [
    #     "mean_clipped_loss.npy",
    #     "mean_entropy_loss.npy",
    #     "mean_value_loss.npy",
    #     "policy_params.npy",
    #     "critic_params.npy",
    # ]
    # single_plots = [True, True, True, False, False]

    # for seed in range(1):
    #     load_path = Path(
    #         f"../algorithms/data/PPO-fixed_KL/Acrobot-v1/"
    #         f"12-03-2020/hyp-0.003/32-32/seed-{seed}/"
    #     )
    #     files = [
    #         "policy_params.npy",
    #     ]
    #     single_plots = [False]
    #
    #     plot(load_path, files, single_plots)

    load_path = Path(
        f"../atari/data/colab_PPO/Pong-ram-v4/24-03-2020/hyp-0.2/128-128-128-128/seed-0/"
    )
    plots = [
        "mean_clipped_loss",
        "mean_entropy_loss",
        "mean_value_loss",
        "shared_layers.0.weight",
        "shared_layers.2.weight",
        "shared_layers.4.weight",
        "actor_layers.0.weight",
        "critic_layers.0.weight",
        "rewards",
    ]
    counts = ["num_steps_taken", "episode_num"]
    print_counts(load_path, counts)
    single_plots = [True, True, True, False, False, False, True]

    plot(load_path, plots, single_plots, min_score=-21, reward_smoothing_weight=0.99)

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
