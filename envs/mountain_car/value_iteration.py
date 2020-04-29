import numpy as np
from tqdm import tqdm
from consts import (
    CONSTS,
    DISC_CONSTS,
    NUM_POSITIONS,
    NUM_VELOCITIES,
)
from mountain_car_runner import test_solution

save_folder = "value_fn"
SAVE_LOCATION1 = f"{save_folder}/value.npy"
SAVE_LOCATION2 = f"{save_folder}/v100_x200.npy"
FINAL_SAVE_LOCATION = f"{save_folder}/v{NUM_VELOCITIES}_x{NUM_POSITIONS}.npy"
SAVE_LOCATION4 = f"{save_folder}/v200_x400.npy"

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
    return np.where(
        np.prod(
            abs(DISC_CONSTS.STATE_SPACE - states)
            <= np.array([DISC_CONSTS.POSITION_SPACING, DISC_CONSTS.VELOCITY_SPACING]),
            axis=-1,
        )
    )[1]


def get_state_ref(state: np.array) -> np.array:
    return np.where(
        np.prod(
            abs(DISC_CONSTS.STATE_SPACE - state)
            <= np.array([DISC_CONSTS.POSITION_SPACING, DISC_CONSTS.VELOCITY_SPACING]),
            axis=-1,
        )
    )


def calculate_rewards(states: np.array) -> np.array:
    return np.prod(states >= [0.5, -CONSTS.MAX_SPEED], axis=-1) - 1


def get_new_state_refs(state: np.array, actions: np.array) -> np.array:
    positions, velocities = np.zeros(actions.shape), np.zeros(actions.shape)
    positions += state[0]
    velocities += state[1]

    velocities += (actions - 1) * CONSTS.FORCE + np.cos(3 * positions) * (
        -CONSTS.GRAVITY
    )
    velocities = np.clip(velocities, -CONSTS.MAX_SPEED, CONSTS.MAX_SPEED)
    positions += velocities
    positions = np.clip(positions, CONSTS.MIN_POSITION, CONSTS.MAX_POSITION)
    return get_state_refs(np.array([[positions], [velocities]]).T)


def value_iteration() -> np.array:
    # Initialise variables
    theta = 0.5
    # value_fn = np.load(FINAL_SAVE_LOCATION)
    value_fn = np.zeros(DISC_CONSTS.STATE_SPACE.shape[0])
    iteration_count = 0

    try:
        while True:
            delta = 0
            for count, state in enumerate(DISC_CONSTS.STATE_SPACE):
                v = value_fn[count]
                outcome_state_refs = get_new_state_refs(state, DISC_CONSTS.ACTION_SPACE)
                rewards = (
                    calculate_rewards(DISC_CONSTS.STATE_SPACE[outcome_state_refs])
                    + value_fn[outcome_state_refs]
                )
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
                    np.save(FINAL_SAVE_LOCATION, value_fn)

            if delta < theta:  # or iteration_count > 200:
                break

    finally:
        np.save(FINAL_SAVE_LOCATION, value_fn)

    print("value function: \n", value_fn)
    np.save(FINAL_SAVE_LOCATION, value_fn)
    return value_fn


def pick_action(state, value_fn, verbose: bool = False):
    outcome_state_refs = get_new_state_refs(state, DISC_CONSTS.ACTION_SPACE)
    outcome_state_values = value_fn[outcome_state_refs]
    if verbose:
        print(f"\nValue: {value_fn[get_state_ref(state)][0]}\t State: {state}")
    return [
        action
        for action, value in zip(DISC_CONSTS.ACTION_SPACE, outcome_state_values)
        if value == max(outcome_state_values)
    ]


def main():
    # value_fn = value_iteration()
    value_fn = np.load(FINAL_SAVE_LOCATION)
    print(np.min(value_fn))
    num_trials = 100
    rewards = []
    timeout = 500
    for _ in tqdm(range(num_trials)):
        reward = test_solution(
            lambda state: pick_action(state, value_fn),
            show_solution=False,
            record_video=False,
            episode_timeout=timeout,
        )
        rewards.append(reward)
    print(f"rewards: {rewards}")
    print(f"Mean reward: {np.mean(rewards)}")


if __name__ == "__main__":
    main()
