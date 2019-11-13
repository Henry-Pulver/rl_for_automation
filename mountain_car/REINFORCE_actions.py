import numpy as np
import gym
from mountain_car_runner import DISC_CONSTS, test_solution
from buffer import ExperienceBuffer

FEATURE_POLYNOMIAL_ORDER = 1


def convert_to_basic_feature(observation: np.array) -> np.array:
    assert observation.shape == (2,)
    p, v = observation
    return np.array([(p ** n1) * (v ** n2) for n1 in range(FEATURE_POLYNOMIAL_ORDER + 1) for n2 in range(FEATURE_POLYNOMIAL_ORDER + 1)])


def feature_to_action_feature(actions: np.array, feature: np.array):
    all_action_features = np.zeros((len(actions), len(actions) * len(feature)))
    for action in actions:
        all_action_features[action][action * len(feature): len(feature) + action * len(feature)] = feature
    return all_action_features


def get_action_probs(policy_weights, action_feature_vectors):
    action_exponents = np.exp(np.matmul(action_feature_vectors, policy_weights))
    return action_exponents / np.sum(action_exponents)


class Policy:
    # ALPHA_BASELINE = 2e-8
    # ALPHA_POLICY = 1e-8
    ALPHA_DECAY = 0.9999
    GAMMA = 0.995
    action_space = DISC_CONSTS.ACTION_SPACE
    baseline_save = "REINFORCE/baseline_weights_p3.npy"
    policy_save = "REINFORCE/policy_weights_p3.npy"

    def __init__(self, load_weights: bool, alpha_baseline: float, alpha_policy: float):
        feature_size = (FEATURE_POLYNOMIAL_ORDER + 1) ** 2
        self.baseline_weights = np.load(self.baseline_save) if load_weights else np.random.normal(size=feature_size)
        self.policy_weights = np.load(self.policy_save) if load_weights else np.random.normal(size=len(self.action_space) * feature_size)
        self.memory_buffer = ExperienceBuffer()
        self.ALPHA_BASELINE = alpha_baseline
        self.ALPHA_POLICY = alpha_policy

    def choose_action(self, state):
        feature_vector = convert_to_basic_feature(state)
        action_feature_vectors = feature_to_action_feature(self.action_space,
                                                           feature_vector)
        action_probs = get_action_probs(self.policy_weights, action_feature_vectors)
        return [np.random.choice(self.action_space, p=action_probs)]

    def calculate_returns(self):
        episode_len = self.memory_buffer.get_length()
        gammas = np.logspace(0, np.log10(self.GAMMA ** (episode_len - 1)),
                             num=episode_len)
        returns = np.array([])
        for timestep in range(episode_len):
            future_rewards = self.memory_buffer.rewards[timestep:]
            # print("future rewards: ", future_rewards)
            returns = np.append(returns, np.sum(np.dot(future_rewards, gammas)))
            # print("returns: ", returns[-1])
            gammas = gammas[:-1]
        # print(returns)
        return returns

    def gather_experience(self, env, time_limit):
        state = env.reset()
        done = False
        total_reward, reward, timesteps = 0, 0, 0

        while not done:
            feature_vector = convert_to_basic_feature(state)
            action_feature_vectors = feature_to_action_feature(self.action_space,
                                                               feature_vector)
            action_probs = get_action_probs(self.policy_weights, action_feature_vectors)
            action_chosen = np.random.choice(self.action_space, p=action_probs)

            self.memory_buffer.update(state, action_chosen, action_probs, reward)

            state, reward, done, info = env.step(action_chosen)
            total_reward += reward
            timesteps += 1

            if timesteps >= time_limit:
                break
        # if not done:
        #     self.memory_buffer.rewards[-1] -= 199
        env.close()
        # print("Episode of experience over, total reward = ", total_reward)
        return total_reward

    def update_weights(self, returns: np.array) -> None:
        """"""
        episode_len = self.memory_buffer.get_length()
        gammas = np.logspace(0, np.log10(self.GAMMA ** (episode_len - 1)),
                             num=episode_len)
        for timestep, state in enumerate(self.memory_buffer.states):
            basic_feature = convert_to_basic_feature(state)
            value = np.dot(self.baseline_weights, basic_feature)
            # print("v", value)
            delta = returns[timestep] - value
            # print("d", delta)
            # self.baseline_weights.astype(np.float64)
            self.baseline_weights += self.ALPHA_BASELINE * delta * basic_feature
            action_taken = self.memory_buffer.actions[timestep]
            # print("action_taken", action_taken)
            action_features = feature_to_action_feature(self.action_space, basic_feature)
            # print("action_features", action_features)
            action_probs = self.memory_buffer.action_probs[timestep]
            # print("action_probs", action_probs)
            # print(action_features * action_probs.reshape((3,1)))
            # print(action_features[action_taken])
            # print(np.sum(action_features * action_probs.reshape((3,1)), axis=0))
            grad_ln_policy = action_features[action_taken] - np.sum(action_features * action_probs.reshape((3,1)), axis=0)
            # print(grad_ln_policy)
            self.policy_weights += self.ALPHA_POLICY * gammas[timestep] * delta * grad_ln_policy
        self.ALPHA_BASELINE *= self.ALPHA_DECAY
        self.ALPHA_POLICY *= self.ALPHA_DECAY
        # print("BL weights: ", self.baseline_weights)
        # print("Policy weights: ", self.policy_weights)
        self.memory_buffer.clear()

    def save_weights(self):
        np.save(self.baseline_save, self.baseline_weights)
        np.save(self.policy_save, self.policy_weights)


def sanity_check(baseline_load, policy_load):
    actions = DISC_CONSTS.ACTION_SPACE
    # baseline_params = np.load(baseline_load)
    policy_params = np.load(policy_load)

    points = [np.array([-1, 0]),
              np.array([0, 0.065]),
              np.array([0.3, -0.02])]

    for count, point in enumerate(points):
        feature = convert_to_basic_feature(point)
        afv = feature_to_action_feature(actions, feature)
        print(f"Point {count}:\n{point}\n{get_action_probs(policy_params, afv)}\n")


def objective(num_steps: int, load: bool, trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-12, 1e-2)
    print(f"Initialising trial {trial.number}, alpha policy = {learning_rate}\n")
    env = gym.make('MountainCar-v0').env
    policy = Policy(load, 2 * learning_rate, learning_rate, trial.number)
    step = 0
    moving_avg = -10000
    try:
        while True:
            total_reward = policy.gather_experience(env, 10000)
            returns = policy.calculate_returns()
            policy.update_weights(returns)

            step += 1
            moving_avg = 0.01 * total_reward + 0.99 * moving_avg

            # Output progress
            if step % 10 == 0:
                print(f"Trial {trial.number}, Step: {step}\tAvg:{moving_avg}")
                policy.save_weights()

                # Report progress to pruner
                trial.report(moving_avg, step)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.structs.TrialPruned()

            if step >= num_steps:
                break
    finally:
        policy.save_weights()
    return moving_avg


def optimise_alpha(num_steps, load_weights):
    study = optuna.create_study(pruner=optuna.pruners.PercentilePruner(66, n_warmup_steps=50), direction="maximize")
    study.optimize(lambda trial: objective(num_steps, load_weights, trial), n_trials=200, n_jobs=-1)


def main():
    # policy = np.array([0.0, 0.0, -10.0, 0.0, 0.0, -100.0, 0.0, 0.0, 0.0,
    #                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                    2.0, 0.0, 10.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
    # baseline = np.array([-50.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # np.save("REINFORCE/policy_weights_human.npy", policy)
    # np.save("REINFORCE/baseline_weights_human.npy", baseline)

    env = gym.make('MountainCar-v0').env
    policy = Policy(load_weights=False, alpha_baseline=5e-3, alpha_policy=2.5e-3)
    iteration_number = 0
    moving_avg = -10000

    try:
        while True:
            total_reward = policy.gather_experience(env, 10000)
            returns = policy.calculate_returns()
            policy.update_weights(returns)

            moving_avg = 0.01 * total_reward + 0.99 * moving_avg
            if iteration_number % 10 == 0:
                # policy.save_weights()
                print(f"Step: {iteration_number} \t Avg: {moving_avg} \t Alpha_policy: {policy.ALPHA_POLICY}")
                # print("BL weights: ", policy.baseline_weights)
                # print("Policy weights: ", policy.policy_weights)

            iteration_number += 1

            if abs(moving_avg) < 150:
                break
            if iteration_number >= 10000:
                    break
    finally:
        policy.save_weights()

    # test_solution(policy.choose_action)
    # sanity_check("REINFORCE/baseline_weights2.npy", "REINFORCE/policy_weights2.npy")

if __name__ == "__main__":
    main()
    # av = feature_to_action_feature(np.array([0, 1, 2]), convert_to_basic_feature(np.array([0.2, 0.07])))
    # rv = 0.05 * np.random.normal(size=27)
    # print(get_action_probs(np.array([0, 1, 2]), rv, np.array([2, 4])))
    # print(rv)
    # print(np.matmul(av, rv))
