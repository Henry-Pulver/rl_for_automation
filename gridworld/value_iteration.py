from gridworld.grid import Grid, PlayableGrid, get_new_state
import random
import numpy as np

STATE_SPACE = [(x, y) for x in range(10) for y in range(10)]
# Update state space by removing wall positions and finish position
for wall_position in Grid.WALL_POSITIONS:
    STATE_SPACE.remove(wall_position)
STATE_SPACE.remove((0, 9))


def value_iteration() -> np.array:
    grid = Grid()

    # Initialise variables
    theta = 0.5
    value_fn = np.array([np.zeros(10)] * 10)

    while True:
        delta = 0
        for state in STATE_SPACE:
            v = value_fn[state]
            possible_actions = grid.find_possible_actions(state)
            outcome_states = [
                get_new_state(state, action) for action in possible_actions
            ]
            rewards = [
                grid.get_reward(state) + value_fn[outcome_state]
                for outcome_state in outcome_states
            ]
            value_fn[state] = max(rewards)
            delta = max(delta, v - value_fn[state])
        if delta < theta:
            break

    # print("delta=", delta)
    print("value function: \n", value_fn)
    return value_fn


def main():
    value_function = value_iteration()
    playable_grid = PlayableGrid()
    # trajectory = []
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
        # trajectory.append(action_chosen)
        if playable_grid.done:
            break
        symbol = "-" if action_chosen in [(0, 1), (0, -1)] else "|"
        path_taken[playable_grid.agent_state] = symbol

    # print("Trajectory = ", trajectory)
    print("final reward = ", playable_grid.agent_reward)
    print("final state = ", playable_grid.agent_state)
    print("Path: \n", path_taken)


if __name__ == "__main__":
    main()
