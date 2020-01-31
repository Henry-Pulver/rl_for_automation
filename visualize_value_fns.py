import numpy as np
import torch
from pathlib import Path
import plotly.graph_objects as go

from mountain_car.consts import (
    DISC_CONSTS,
    VELOCITY_VALUES,
    POSITION_VALUES,
    NUM_VELOCITIES,
    NUM_POSITIONS,
)
from REINFORCE_actions import Policy
from algorithms.discrete_policy import DiscretePolicy
from imitation.behavioural_cloning import action_probs


def load_data(load_path: Path) -> np.array:
    if load_path.exists():
        return np.load(str(load_path), allow_pickle=True)
    else:
        raise FileNotFoundError("load_path variable doesn't match a file that exists")


def show_contours(
    x_data: np.array,
    y_data: np.array,
    z_data: np.array,
    title: str,
    filename: str,
    project_z: bool = True,
):
    fig = go.Figure(data=[go.Surface(x=x_data, y=y_data, z=z_data)])
    fig.update_traces(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="limegreen", project_z=project_z
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=65, r=50, b=65, t=60),
        scene=dict(
            xaxis=dict(nticks=10),
            yaxis=dict(nticks=10),
            xaxis_title="Position",
            yaxis_title="Velocity",
            zaxis_title="Value",
        ),
    )
    fig.write_html(filename, auto_open=True)


# def show_quiver_plot():
#     pass


def show_discrete_policy(raw_policy: np.array):
    for count, item in enumerate(raw_policy):
        if item == [0, 1, 2]:
            raw_policy[count] = 0
        elif item == [0]:
            raw_policy[count] = -1
        elif item == [1]:
            raw_policy[count] = 1
        elif item == [2]:
            raw_policy[count] = 1
        elif item == [0, 1]:
            raw_policy[count] = -1
        elif item == [1, 2]:
            raw_policy[count] = 1

        # if item == [0, 1, 2]:
        #     raw_policy[count] = 0.5
        # elif item == [0]:
        #     raw_policy[count] = -2
        # elif item == [1]:
        #     raw_policy[count] = 0
        # elif item == [2]:
        #     raw_policy[count] = 2
        # elif item == [0, 1]:
        #     raw_policy[count] = -1
        # elif item == [1, 2]:
        #     raw_policy[count] = 1

    return raw_policy.reshape((NUM_VELOCITIES, NUM_POSITIONS))


def show_cts_policy(title: str, x_data, y_data, z_data: np.array):
    fig = go.Figure()
    for count, data in enumerate(z_data):
        fig.add_trace(go.Surface(x=x_data, y=y_data, z=data, name=f"{count}"))
    # fig.update_traces(
    #     contours_z=dict(
    #         show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
    #     )
    # )
    fig.update_layout(
        title=title,
        margin=dict(l=65, r=50, b=65, t=60),
        scene=dict(
            xaxis=dict(nticks=10),
            yaxis=dict(nticks=10),
            xaxis_title="Position",
            yaxis_title="Velocity",
            zaxis_title="Value",
        ),
    )
    fig.write_html(f"{title}.html", auto_open=True)


def cts_to_discrete(z_data: np.array):
    print(z_data.shape)
    new_values = np.dot(z_data.T, np.array([-1, 0, 1]))
    return new_values
    # return raw_policy.reshape((NUM_VELOCITIES, NUM_POSITIONS))


def main():
    # value_iteration_save_location = "mountain_car/value_fn"
    # filename = f"v{NUM_VELOCITIES}_x{NUM_POSITIONS}"
    # value_fn_data = load_data(Path(f"{value_iteration_save_location}/{filename}.npy"))
    # value_fn_data = value_fn_data.reshape((NUM_VELOCITIES, NUM_POSITIONS))
    positions = POSITION_VALUES
    velocities = VELOCITY_VALUES

    #################### View policy ####################
    # policy_save_location = (
    #     f"{value_iteration_save_location}/policy_v{NUM_VELOCITIES}_x{NUM_POSITIONS}.npy"
    # )
    # _, policy = policy_improvement([DISC_CONSTS.ACTION_SPACE] * DISC_CONSTS.STATE_SPACE.shape[0], value_fn_data, save=True,
    #                                save_location=policy_save_location)
    # policy = load_data(Path(policy_save_location))
    # policy = show_discrete_policy(policy)
    # show_contours(
    #     x_data=positions,
    #     y_data=velocities,
    #     z_data=policy,
    #     title=f"Mountain Car Policy - Value iteration, v={NUM_VELOCITIES}, x={NUM_POSITIONS}",
    #     filename=f"policy_{filename}.html",
    #     project_z=False,
    # )

    #################### View Value Function #############
    # show_contours(x_data=positions,
    #               y_data=velocities,
    #               z_data=value_fn_data,
    #               title=f"Mountain Car Value Function - Value iteration, v={NUM_VELOCITIES}, x={NUM_POSITIONS}",
    #               filename=f"{filename}.html",
    #               project_z=False,
    #               )

    ##################### View stochastic policy ########################
    ref_num = 2001
    filename = "REINFORCE_actions_baseline_order_2"
    load_path = f"mountain_car/REINFORCE_actions/weights/{ref_num}"
    # load_path = f"mountain_car/REINFORCE_states/weights/human"
    policy = Policy(
        ref_num=0,
        alpha_baseline=1,
        alpha_policy=1,
        policy_load=f"{load_path}/policy_weights_{ref_num}.npy",
    )
    z_data = np.array([policy.action_probs(state) for state in DISC_CONSTS.STATE_SPACE])
    final_data = np.array(
        [action.reshape((NUM_VELOCITIES, NUM_POSITIONS)) for action in z_data.T]
    )

    # policy = cts_to_discrete(
    #     z_data=final_data,
    # )

    # show_contours(
    #     x_data=positions[4:-4],
    #     y_data=velocities[4:-4],
    #     z_data=policy.T[4:-4,4:-4],
    #     title=f"Mountain Car Policy - REINFORCE, 2nd order polynomial",
    #     filename=f"policy_{filename}.html",
    #     project_z=False,
    # )
    show_cts_policy(
        # title="REINFORCE States polynomial order 2 Policy without baseline",
        title="REINFORCE Actions Current",
        x_data=positions,
        y_data=velocities,
        z_data=final_data,
    )

    ##################### Neural Network Policy ######################
    # filename = "neural_network"
    # neural_net_path = "mountain_car/imitation_learning/BC/2020/01/15/demos_18_seed_4.pt"
    # network = DiscretePolicy(
    #     action_space=DISC_CONSTS.ACTION_SPACE,
    #     state_dimension=DISC_CONSTS.STATE_SPACE.shape[-1],
    #     hidden_layers=(32, 32),
    # ).float()
    # network.load_state_dict(torch.load(neural_net_path))
    # network.eval()
    # z_data = np.array(
    #     [action_probs(state, network) for state in DISC_CONSTS.STATE_SPACE]
    # )
    # final_data = np.array(
    #     [action.reshape((NUM_VELOCITIES, NUM_POSITIONS)) for action in z_data.T]
    # )
    # policy = cts_to_discrete(z_data=final_data,)
    #
    # show_contours(
    #     x_data=positions,
    #     y_data=velocities,
    #     z_data=policy.T,
    #     title=f"Mountain Car Policy - BC",
    #     filename=f"bc_policy_{filename}.html",
    #     project_z=False,
    # )


if __name__ == "__main__":
    # policy_weights = np.array([0, 100, 100000, 1, 0, 0, 5, 0, 0])
    # baseline_weights = np.array([-1000, 1.753, 1.673, 40, 0, 0, 200, 0, 0])
    # np.save("mountain_car/REINFORCE_states/weights/human/policy_weights_0.npy", policy_weights)
    # np.save("mountain_car/REINFORCE_states/weights/human/baseline_weights_0.npy", baseline_weights)
    main()
