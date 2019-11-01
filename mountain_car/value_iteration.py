import numpy as np
import gym
import random


ACTION_SPACE = np.array([x for x in range(3)])
GRAVITY = 0.0025
FORCE = 0.001
MIN_POSITION = -1.2
MAX_POSITION = 0.6
MAX_SPEED = 0.07
GOAL_POSITION = 0.5

POSITION_VALUES = np.linspace(MIN_POSITION, MAX_POSITION, 200)  # 100 possible positions
VELOCITY_VALUES = np.linspace(-MAX_SPEED, MAX_SPEED, 100)  # 50 possible velocities
STATE_SPACE = np.array([np.array([pos, vel]) for vel in VELOCITY_VALUES for pos in POSITION_VALUES])

POSITION_SPACING = (POSITION_VALUES[1] - POSITION_VALUES[0]) / 2
VELOCITY_SPACING = (VELOCITY_VALUES[1] - VELOCITY_VALUES[0]) / 2

SAVE_LOCATION1 = "value_fn/value.npy"
SAVE_LOCATION2 = "value_fn/v100_x200.npy"
# print("pspace", POSITION_SPACING)
# print("vspace", VELOCITY_SPACING)

# class ValueIterator:
#     def __init__(self, state_space: np.array, ):
#         self.state_space = state_space
#         self.value_fn = np.zeros(state_space.shape)
#
#     def discretise_state(self) -> np.array:
#         """If necessary, discretises the state into bins."""
#         return self.state
#
#     def calculate_reward(self) -> int:
#         return 0
#
#     def calculate_optimal_value_fn(self) -> np.array:
#         return self.value_fn


# class MountainCarValueIterator(ValueIterator):
#     def __init__(self, state_space: np.array):

def get_state_refs(states: np.array) -> np.array:
    return np.where(np.prod(abs(STATE_SPACE - states) <= np.array([POSITION_SPACING, VELOCITY_SPACING]), axis=-1))[1]


def calculate_rewards(states: np.array) -> np.array:
    return np.prod(states >= [0.5, -MAX_SPEED], axis=-1) - 1


def get_new_state_refs(state: np.array, actions: np.array) -> np.array:
    positions, velocities = np.zeros(actions.shape), np.zeros(actions.shape)
    positions += state[0]
    velocities += state[1]

    velocities += (actions - 1) * FORCE + np.cos(3 * positions) * (-GRAVITY)
    velocities = np.clip(velocities, -MAX_SPEED, MAX_SPEED)
    positions += velocities
    positions = np.clip(positions, MIN_POSITION, MAX_POSITION)
    return get_state_refs(np.array([[positions], [velocities]]).T)


def value_iteration() -> np.array:
    # Initialise variables
    theta = 0.5
    value_fn = np.load(SAVE_LOCATION2)
    # value_fn = np.zeros(STATE_SPACE.shape[0])
    iteration_count = 0

    try:
        while True:
            delta = 0
            for count, state in enumerate(STATE_SPACE):
                v = value_fn[count]
                outcome_state_refs = get_new_state_refs(state, ACTION_SPACE)
                rewards = calculate_rewards(STATE_SPACE[outcome_state_refs]) + value_fn[outcome_state_refs]
                value_fn[count] = max(rewards)
                delta = max(delta, abs(v - value_fn[count]))

            iteration_count += 1
            print(iteration_count)

            if iteration_count % 2 == 1:
                print("delta = ", delta)
                if iteration_count % 10 == 1:
                    print(value_fn)
                if iteration_count % 20 == 1:
                    print("saving...")
                    np.save(SAVE_LOCATION2, value_fn)

            if delta < theta:  # or iteration_count > 200:
                break

    finally:
        np.save(SAVE_LOCATION2, value_fn)

    print("value function: \n", value_fn)
    np.save(SAVE_LOCATION2, value_fn)
    return value_fn


def main():
    # value_fn = value_iteration()
    value_fn = np.load(SAVE_LOCATION2)

    env = gym.make('MountainCar-v0').env
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        outcome_state_refs = get_new_state_refs(state, ACTION_SPACE)
        outcome_state_values = value_fn[outcome_state_refs]
        best_actions = [action for action, value in
                        zip(ACTION_SPACE, outcome_state_values)
                        if value == max(outcome_state_values)]
        action_chosen = random.choice(best_actions)
        state, reward, done, info = env.step(action_chosen)
        total_reward += reward

    env.close()
    print("final reward = ", total_reward)
    print("final state = ", state)


if __name__ == "__main__":
    main()
