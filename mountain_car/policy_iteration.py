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

POSITION_VALUES = np.linspace(MIN_POSITION, MAX_POSITION, 200)  # 200 discrete positions
VELOCITY_VALUES = np.linspace(-MAX_SPEED, MAX_SPEED, 100)  # 100 discrete velocities
STATE_SPACE = np.array([np.array([pos, vel])
                        for vel in VELOCITY_VALUES
                        for pos in POSITION_VALUES])

POSITION_SPACING = (POSITION_VALUES[1] - POSITION_VALUES[0]) / 2
VELOCITY_SPACING = (VELOCITY_VALUES[1] - VELOCITY_VALUES[0]) / 2

VALUE_SAVE_LOCATION = "policy_iteration/value.npy"
POLICY_SAVE_LOCATION = "policy_iteration/policy.npy"


def get_state_refs(states: np.array) -> np.array:
    return np.where(np.prod(abs(STATE_SPACE - states)
                            <= np.array([POSITION_SPACING, VELOCITY_SPACING]), axis=-1))[1]


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


def policy_evaluation(value_fn, policy):
    print("Starting policy evaluation!")
    theta = 4
    iteration_count = 0
    try:
        while True:
            delta = 0
            for count, state in enumerate(STATE_SPACE):
                v = value_fn[count]
                outcome_state_refs = get_new_state_refs(state, policy[count])
                rewards = calculate_rewards(STATE_SPACE[outcome_state_refs]) \
                          + value_fn[outcome_state_refs]
                value_fn[count] = np.sum(rewards) / len(policy[count])
                delta = max(delta, abs(v - value_fn[count]))

            iteration_count += 1
            print(iteration_count)

            if iteration_count % 2 == 1:
                print("delta = ", delta)
                if iteration_count % 20 == 1:
                    print("saving...")
                    np.save(VALUE_SAVE_LOCATION, value_fn)

            if delta < theta:  # or iteration_count > 200:
                print("Policy evaluation is complete!")
                print("Delta value = ", delta)
                break

    finally:
        np.save(VALUE_SAVE_LOCATION, value_fn)

    return value_fn


def policy_improvement(policy, value_fn):
    print("Policy improvement started!")
    policy_stable = True
    for count, state in enumerate(STATE_SPACE):
        old_action = policy[count]
        outcome_state_refs = get_new_state_refs(state, policy[count])
        policy[count] = np.where(value_fn[outcome_state_refs] ==
                                 max(value_fn[outcome_state_refs]))[0]
        if np.all(old_action == policy[count]):
            policy_stable = False
    np.save(POLICY_SAVE_LOCATION, policy)
    print("Policy improvement is complete!")
    return policy_stable, policy


def policy_iteration() -> np.array:
    value_fn = np.load(VALUE_SAVE_LOCATION)
    policy = np.load(POLICY_SAVE_LOCATION, allow_pickle=True)
    # value_fn = np.zeros(STATE_SPACE.shape[0])
    # policy = [ACTION_SPACE] * STATE_SPACE.shape[0]

    while True:
        value_fn = policy_evaluation(value_fn, policy)
        policy_stable, policy = policy_improvement(policy, value_fn)
        if policy_stable:
            break

    np.save(VALUE_SAVE_LOCATION, value_fn)
    np.save(POLICY_SAVE_LOCATION, policy)
    return policy


def main():
    policy = policy_iteration()

    policy = np.load(POLICY_SAVE_LOCATION, allow_pickle=True)

    env = gym.make('MountainCar-v0').env
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        best_actions = policy[np.where(np.prod(abs(STATE_SPACE - state)
                         <= np.array([POSITION_SPACING, VELOCITY_SPACING]), axis=-1))[0]][0]
        action_chosen = random.choice(best_actions)
        print(action_chosen)
        state, reward, done, info = env.step(action_chosen)
        total_reward += reward

    env.close()
    print("final reward = ", total_reward)
    print("final state = ", state)


if __name__ == "__main__":
    main()
