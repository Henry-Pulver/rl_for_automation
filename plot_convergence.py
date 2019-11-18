import numpy as np
import plotly.graph_objects as go
import os
from REINFORCE_next_states import FEATURE_POLYNOMIAL_ORDER


def plot_weights_and_performance(
    ref_num: int, opt: bool, feature_vector_size: int
) -> None:
    path = f"mountain_car/REINFORCE_states/plots"
    opt_str = "optuna_" if opt else ""

    fig = go.Figure()
    y = np.load(f"{path}/{opt_str}baseline_plot_{ref_num}.npy").T
    y = y.reshape((-1, feature_vector_size)).T
    x = np.linspace(0, y.shape[1], y.shape[1] + 1)
    for theta in y:
        fig.add_trace(go.Scatter(x=x, y=theta))
    fig.write_html(f"{path}/{opt_str}baseline_plot_{ref_num}.html", auto_open=True)

    fig = go.Figure()
    y = np.load(f"{path}/{opt_str}policy_plot_{ref_num}.npy").T
    y = y.reshape((-1, feature_vector_size)).T
    x = np.linspace(0, y.shape[1], y.shape[1] + 1)
    for theta in y:
        fig.add_trace(go.Scatter(x=x, y=theta))
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


def plot_run() -> None:
    root_dir = "mountain_car/REINFORCE_states/plots/0/1000"
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
                for theta in y:
                    fig.add_trace(go.Scatter(x=x, y=theta,))
            else:
                x = np.linspace(0, y.shape[0], y.shape[0] + 1)
                fig.add_trace(go.Scatter(x=x, y=y))
            file = os.path.splitext(file)[0]
            fig.write_html(f"{root_dir}/{file}.html", auto_open=True)


def main():
    plot_weights_and_performance(
        ref_num=0, opt=True, feature_vector_size=(FEATURE_POLYNOMIAL_ORDER + 1) ** 2
    )
    # plot_run()


if __name__ == "__main__":
    main()
