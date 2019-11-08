import random
import numpy as np
import gym

ALPHA_BASELINE = 0.02
ALPHA_POLICY = 0.01


def convert_to_basic_feature(observation: np.array) -> np.array:
    assert observation.shape == (2,)
    p, v = observation
    return np.array([1,
                     p,
                     v,
                     p * v,
                     p ** 2,
                     v ** 2,
                     p * (v ** 2),
                     v * (p ** 2),
                     (p ** 2) * (v ** 2)])


def feature_to_action_feature(actions: np.array, feature: np.array):
    all_action_features = np.zeros((len(actions), len(actions) * len(feature)))
    for action in actions:
        all_action_features[action][action * len(feature): len(feature) + action * len(feature)] = feature
    return all_action_features




def get_action_probs(actions, policy_weights, observation):
    feature_vector = convert_to_basic_feature(observation)
    all_action_feature_vectors = feature_to_action_feature(actions, feature_vector)
    action_exponents = np.exp(np.matmul(all_action_feature_vectors, policy_weights))
    return action_exponents / np.sum(action_exponents)


def run_episode(action_space, policy_weights, env):
    state = env.reset()
    done = False
    total_reward = 0
    state_list, action_list, reward_list = [], [], []

    while not done:
        # env.render()
        state_list.append(state)
        action_probs = get_action_probs(action_space, policy_weights, state)
        action_chosen = np.random.choice(action_space, p=action_probs)

        state, reward, done, info = env.step(action_chosen)
        action_list.append(action_chosen)
        reward_list.append(reward)
        total_reward += reward

    env.close()
    print("Episode over, total reward = ", total_reward)
    state_list.append(state)
    return state_list, action_list, reward_list


def update_baseline(baseline_weights, states, rewards):
    pass


def update_policy(policy_weights, states, actions, rewards):
    pass


def update_weights(policy_weights, baseline_weights, states, actions, rewards):
    update_baseline(baseline_weights, states, rewards)
    update_policy(policy_weights, states, actions, rewards)



def main():
    baseline_weights = np.random.normal(size=9)
    policy_weights = 0.05 * np.random.normal(size=27)
    action_space = np.array([n for n in range(3)])

    env = gym.make('MountainCar-v0')

    states, actions, rewards = run_episode(action_space, policy_weights, env)
    update_weights()

    # np.save("REINFORCE/baseline_weights.npy", baseline_weights)
    # np.save("REINFORCE/policy_weights.npy", policy_weights)


if __name__ == "__main__":
    main()
    # av = feature_to_action_feature(np.array([0, 1, 2]), convert_to_basic_feature(np.array([0.2, 0.07])))
    # rv = 0.05 * np.random.normal(size=27)
    # print(get_action_probs(np.array([0, 1, 2]), rv, np.array([2, 4])))
    # print(rv)
    # print(np.matmul(av, rv))
