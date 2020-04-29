import numpy as np
import plotly.graph_objects as go
import os
from typing import List, Optional, Dict
from pathlib import Path

from algorithms.utils import generate_save_location


def print_counts(load_path: Path, counts: List):
    for count in counts:
        print(f"{count}: {np.load(f'{load_path}/{count}.npy')}")


def load_plot(plot_dir: Path):
    # Retrieve data from save files
    file_count = 0
    for file in plot_dir.iterdir():
        if file.suffix == ".npy":
            file_count += 1

    for file_num in range(1, file_count + 1):
        file = plot_dir / f"{plot_dir.name}_{file_num}.npy"
        if file_num == 1:
            plot = np.load(f"{file}", allow_pickle=True).T
        else:
            plot = np.append(plot, np.load(f"{file}", allow_pickle=True).T, axis=-1)
    plot = np.squeeze(plot)
    return plot


def plot_data(
    plot: np.array,
    plot_name: str,
    load_path: Path,
    reward_smoothing_weight: Optional[float] = None,
    min_score: Optional[float] = None,
    graph_labels: Optional[Dict] = None,
):
    graph_labels = (
        {"Title": "", "x": "", "y": "", "Legend": [""]}
        if graph_labels is None
        else graph_labels
    )
    fig = go.Figure()
    x = np.linspace(0, plot.shape[-1], plot.shape[-1] + 1)

    # If it has multiple plots
    if len(plot.shape) > 1:
        if not len(plot) == len(graph_labels["Legend"]):
            graph_labels["Legend"] = [""] * len(plot)
        for count, theta in enumerate(plot):
            fig.add_trace(go.Scatter(x=x, y=theta, name=graph_labels["Legend"][count]))
    else:
        fig.add_trace(go.Scatter(x=x, y=plot))

    fig.update_layout(
        title=graph_labels["Title"],
        xaxis_title=graph_labels["x"],
        yaxis_title=graph_labels["y"],
    )
    fig.write_html(
        f"{load_path}/{plot_name}.html", auto_open=True,
    )
    if load_path.stem == "rewards":
        fig2 = go.Figure()
        plot_smoothed = smooth_rewards(plot, reward_smoothing_weight, min_score)
        fig2.add_trace(go.Scatter(x=x, y=plot_smoothed))
        fig2.update_layout(
            title=graph_labels["Title"],
            xaxis_title=graph_labels["x"],
            yaxis_title=graph_labels["y"],
        )
        fig2.write_html(
            f"{load_path}/smoothed_rewards.html", auto_open=True,
        )


def output_plots_and_counts(
    load_path: Path,
    reward_smoothing_weight: float,
    min_score: float,
    plots_to_plot: Optional[List] = None,
    graph_labels: Optional[Dict] = None,
):
    for entry in load_path.iterdir():
        if plots_to_plot is None:
            show_plot = True
        else:
            show_plot = entry.stem in plots_to_plot

        # Counts
        if not entry.is_dir() and not entry.suffix == ".pth":
            print(f"{entry.stem}: {np.load(f'{entry}', allow_pickle=True)}")

        # Plots
        elif show_plot and entry.is_dir() and not entry.name == "params":
            plot = load_plot(entry)
            plot_data(
                plot,
                entry.stem,
                entry,
                reward_smoothing_weight,
                min_score,
                graph_labels,
            )


def load_seed_plots(load_path: Path, plots_to_plot: List):
    plots = {plot_name: [] for plot_name in plots_to_plot}
    seeds = []
    for seed_file in load_path.iterdir():
        if seed_file.is_dir():
            seeds.append(seed_file.name[5:])
            for entry in seed_file.iterdir():
                show_plot = entry.name in plots_to_plot or plots_to_plot is None

                # Plots
                if show_plot and entry.is_dir() and not entry.name == "params":
                    plot = load_plot(entry)
                    plots[entry.name].append(plot)
    return plots, seeds


def plot_random_seed_avgs(
    load_path: Path,
    plots_to_plot: List,
    reward_smoothing_weight: Optional[float] = None,
    min_score: Optional[float] = None,
    graph_labels: Optional[Dict] = None,
):
    graph_labels = (
        {"Title": "", "x": "", "y": "", "Legend": [""]}
        if graph_labels is None
        else graph_labels
    )
    plots, seeds = load_seed_plots(load_path, plots_to_plot)
    for plot_name, plot in plots.items():
        plot_data(
            np.mean(np.array(plot), axis=0),
            f"{plot_name}-seed-avg",
            load_path,
            reward_smoothing_weight=reward_smoothing_weight,
            min_score=min_score,
            graph_labels=graph_labels,
        )
        graph_labels["Legend"] = [f"Seed: {seed}" for seed in seeds]
        plot_data(
            np.array(plot),
            plot_name,
            load_path,
            reward_smoothing_weight=reward_smoothing_weight,
            min_score=min_score,
            graph_labels=graph_labels,
        )


def plot_demo_num_avgs(
    load_path: Path,
    arch_str: str,
    plot_to_plot: str,
    graph_labels: Optional[Dict] = None,
):
    nums_list = []
    for num_demos_path in load_path.iterdir():
        # Remove any files here!
        if not num_demos_path.is_dir():
            num_demos_path.unlink()

        # Find the numbers of demos used
        else:
            nums_from_end = 1
            num_string = ""
            while True:
                try:
                    next_char = num_demos_path.name[-nums_from_end]
                    int(next_char)
                    num_string = next_char + num_string
                    nums_from_end += 1
                except ValueError:
                    hyp_prefix = num_demos_path.name[: 1 - nums_from_end]
                    nums_list.append(int(num_string))
                    break
    nums_list = sorted(nums_list)
    print(nums_list)
    # Load in plots
    plots = [
        np.array(
            load_seed_plots(
                load_path / f"{hyp_prefix}{num}" / arch_str, [plot_to_plot]
            )[0][plot_to_plot]
        )
        for num in nums_list
    ]

    graph_labels = {} if graph_labels is None else graph_labels
    graph_labels["Legend"] = np.array([f"# demos used: {num}" for num in nums_list])

    # Cut short any unnecessarily long plots, take mean over random seeds
    for count, plot in enumerate(plots):
        plots[count] = np.mean(
            np.array([seed[-len(min(plot, key=len)) :] for seed in plot]), axis=0
        )

    plot_data(
        plot=np.array(plots),
        plot_name="demo_num_plot",
        load_path=load_path,
        graph_labels=graph_labels,
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

    #### PPO 2 ###
    # load_path = Path("../algorithms/data/PPO-clip/Acrobot-v1/12-03-2020/hyp-0.2/32-32/seed-0/")

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

    # load_path = Path(
    #     f"../atari/data/colab_PPO/Pong-ram-v4/24-03-2020/hyp-0.2/128-128-128-128/seed-0/"
    # )

    # load_path = Path("../imitation_learning/data/GAIL-clip/Acrobot-v1/05-04-2020/hyp-0.2-num_demos_100/32-32/seed-0/")
    # load_path = Path(
    #     "../imitation_learning/data/BC/CartPole-v1/23-04-2020/hyp-demos-100/32-32/seed-0/epoch_loss/epoch_loss_1.npy"
    # )
    hyps = [0, 0.2]
    for hyp in hyps:
        base_path = Path(
            f"../reinforcement_learning/data/PPO-clip/CartPole-v1/28-04-2020/hyp-{hyp}/32-32/seed-0/"
        )
        # param_name = "actor_layers.0.weight"
        # actor_params = np.load(base_path/param_name/ f"{param_name}_1.npy")
        # print(actor_params.shape)

        graph_labels = {
            "Title": f"Atari Ms Pacman clipped PPO smoothed rewards, clipping parameter: {hyp}",
            # "x": "",
            "x": "Episode number",
            # "y": "",
            "y": "Score",
            "Legend": [""]
        }
        load_path = base_path
        output_plots_and_counts(
            load_path,
            min_score=22,
            reward_smoothing_weight=0.99,
            graph_labels=graph_labels,
            # plots_to_plot=["rewards"],
        )

    # for seed in range(5):
    #     graph_labels = {
    #         "Title": f"Mountain Car rewards Random Seed: {seed}",
    #         "x": "Episode number",
    #         "y": "Score",
    #     }
    #     load_path = base_path / f"seed-{seed}"
    #     output_plots_and_counts(
    #         load_path,
    #         min_score=-9000,
    #         reward_smoothing_weight=0.98,
    #         graph_labels=graph_labels,
    #         plots_to_plot=["rewards"],
    #     )

    # BC plotting the effect of the number of demos
    # alg = "BC"
    # algorithm_name = "Behavioural Cloning"
    # env_names = [
    #     "CartPole-v1",
    #     "MountainCar-v0",
    #     "Acrobot-v1",
    # ]
    # for env_name in env_names:
    #     graph_labels = {
    #         "Title": f"{algorithm_name} effect of number of demos on avg score for {env_name}",
    #         "x": "Epochs trained",
    #         "y": "Avg Score",
    #     }
    #     plot_demo_num_avgs(
    #         load_path=Path(f"../imitation_learning/data/{alg}/{env_name}/23-04-2020/"),
    #         arch_str="32-32",
    #         plot_to_plot="avg_score",
    #         graph_labels=graph_labels,
    #     )

    # plot_random_seed_avgs(
    #     Path("../imitation_learning/data/BC/CartPole-v1/23-04-2020/hyp-demos-1/32-32/"),
    #     ["avg_score", "epoch_loss"],
    # )

    # plot_to_change = np.load(load_path)
    # print(plot_to_change)
    # print(plot_to_change.shape)
    # plot_to_change = plot_to_change[20:]
    # print(plot_to_change)
    # print(plot_to_change.shape)
    # np.save(f"{load_path}", plot_to_change)
    # plot_to_change = np.load(load_path)
    # print(plot_to_change)
    # print(plot_to_change.shape)

    # # for plot in plots:
    # for entry in load_path.iterdir():
    #     if entry.is_dir():
    #         for filename in entry.iterdir():
    #             print(filename)

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
    # num_demos_path = Path("../asdfg/sDEGWee1234")
    # nums_from_end = 1
    # nums_list = []
    # num_string = ""
    # while True:
    #     try:
    #         next_char = num_demos_path.name[-nums_from_end]
    #         int(next_char)
    #         num_string = next_char + num_string
    #         nums_from_end += 1
    #     except ValueError:
    #         hyp_prefix = num_demos_path.name[:1 - nums_from_end]
    #         nums_list.append(int(num_string))
    #         break
    # nums_list = sorted(nums_list)
    # print(nums_list)

    # borken_plot = np.load(f"../imitation_learning/data/BC/MountainCar-v0/23-04-2020/hyp-demos-100/32-32/seed-2/avg_score/avg_score_1.npy")
    # print(borken_plot)
    # n = [np.array([2, 3, 1]),
    #      np.array([6, 2, 3, 7])]
    # print(min(n, key=len))
    # print(np.array([thing[-len(min(n, key=len)):] for thing in n]))
