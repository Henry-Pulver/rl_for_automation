import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

from envs.gridworld.grid import Grid, PlayableGrid, get_new_state

STATE_SPACE = [(x, y) for x in range(10) for y in range(10)]
# Update state space by removing wall positions and finish position
for wall_position in Grid.WALL_POSITIONS:
    STATE_SPACE.remove(wall_position)
STATE_SPACE.remove((0, 9))


def value_iteration() -> np.ndarray:
    grid = Grid()

    # Initialise variables
    theta = 0.5
    value_fn = np.array([np.zeros(10)] * 10)

    while True:
        delta = 0
        for state in STATE_SPACE:
            prev_value = value_fn[state]
            possible_actions = grid.find_possible_actions(state)
            outcome_states = [
                get_new_state(state, action) for action in possible_actions
            ]
            rewards = [
                grid.get_reward(state) + value_fn[outcome_state]
                for outcome_state in outcome_states
            ]
            value_fn[state] = max(rewards)
            delta = max(delta, prev_value - value_fn[state])
        if delta < theta:
            break

    print("value function: \n", value_fn)
    return value_fn


def policy_iteration() -> np.ndarray:
    grid = Grid()

    # Initialise variables
    theta = 1
    value_fn = np.array([np.zeros(10)] * 10)
    policy = np.array(
        [[grid.find_possible_actions((y, x)) for x in range(10)] for y in range(10)]
    )

    policy_unchanged = False
    while not policy_unchanged:
        # Policy evaluation
        while True:
            delta = 0
            for state in STATE_SPACE:
                prev_value = value_fn[state]
                outcome_states = [
                    get_new_state(state, action) for action in policy[state]
                ]
                outcome_values = [value_fn[state] for state in outcome_states]
                value_fn[state] = grid.get_reward(state) + (
                    1 / len(outcome_values)
                ) * sum(outcome_values)
                delta = max(delta, abs(prev_value - value_fn[state]))
            if delta < theta:
                print("\n Policy evaluated succesfully!")
                break
        visualise_value_fn(value_fn)

        # Policy improvement
        policy_unchanged = True
        for state in STATE_SPACE:
            prev_action = policy[state]
            possible_actions = grid.find_possible_actions(state)
            outcome_states = [
                get_new_state(state, action) for action in possible_actions
            ]
            outcome_values = np.array(
                [
                    grid.get_reward(state) + value_fn[outcome_state]
                    for outcome_state in outcome_states
                ]
            )
            action_ref = np.where(outcome_values == max(outcome_values))[0]
            policy[state] = possible_actions[action_ref]
            policy_unchanged = (
                np.all(prev_action == policy[state]) if policy_unchanged else False
            )
        print("\n Policy improved succesfully!")

    print("value function: \n", value_fn)
    return value_fn


def visualise_value_fn(value_fn: np.ndarray):
    value_fn = deepcopy(value_fn)
    nrows, ncols = value_fn.shape
    values = np.linspace(np.min(value_fn), np.max(value_fn), num=9)
    value_fn[np.where(value_fn == 0)] = None
    value_fn[0, 9] = 0

    im = plt.imshow(
        value_fn,
        extent=(0, nrows, 0, ncols),
        interpolation="nearest",
        cmap=cm.gist_rainbow,
    )
    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in reversed(values)]
    # create a patch (proxy artist) for every color
    patches = [
        mpatches.Patch(color=colors[i], label=f"{np.round(values[-(i + 1)], 1)}")
        for i in range(len(values))
    ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.xlabel("x Position")
    plt.ylabel("y Position")
    plt.show()


def main():
    value_function = policy_iteration()
    playable_grid = PlayableGrid()
    path_taken = np.array([["0"] * 10] * 10)
    path_taken[0, 9] = "F"
    path_taken[9, 0] = "S"

    while True:
        possible_actions = playable_grid.find_possible_actions(
            playable_grid.agent_state
        )
        outcome_states = [
            get_new_state(playable_grid.agent_state, action)
            for action in possible_actions
        ]
        rewards = [
            playable_grid.get_reward(playable_grid.agent_state)
            + value_function[outcome_state]
            for outcome_state in outcome_states
        ]
        best_actions = [
            action
            for action, reward in zip(possible_actions, rewards)
            if reward == max(rewards)
        ]
        action_chosen = random.choice(best_actions)
        playable_grid.take_action(action_chosen)
        if playable_grid.done:
            break
        symbol = "-" if action_chosen in [(0, 1), (0, -1)] else "|"
        path_taken[playable_grid.agent_state] = symbol

    print("final reward = ", playable_grid.agent_reward)
    print("final state = ", playable_grid.agent_state)
    print("Path: \n", path_taken)


if __name__ == "__main__":
    main()
