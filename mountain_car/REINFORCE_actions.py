import numpy as np
from typing import Optional, List
from mountain_car.consts import DISC_CONSTS
from algorithms.buffer import ExperienceBuffer
import optuna
import math
import gym
import os
from tqdm import tqdm

FEATURE_POLYNOMIAL_ORDER = 2


def convert_to_basic_feature(state: np.array) -> np.array:
    p, v = state.T
    return np.array(
        [
            (p ** n1) * (v ** n2)
            for n1 in range(FEATURE_POLYNOMIAL_ORDER + 1)
            for n2 in range(FEATURE_POLYNOMIAL_ORDER + 1)
        ]
    )


def feature_to_action_feature(actions: np.array, feature: np.array) -> np.array:
    all_action_features = np.zeros(
        (actions.shape[-1], actions.shape[-1] * feature.shape[-1])
    )
    for action in actions:
        all_action_features[action][
            action * len(feature) : len(feature) + action * len(feature)
        ] = feature
    return all_action_features


def get_action_probs(policy_weights, action_feature_vectors) -> np.array:
    action_exponents = np.matmul(action_feature_vectors, policy_weights)
    unnormalised_probs = np.exp(action_exponents - np.mean(action_exponents))
    return unnormalised_probs / np.sum(unnormalised_probs)


class Policy:
    action_space = DISC_CONSTS.ACTION_SPACE

    def __init__(
        self,
        alpha_baseline: float,
        alpha_policy: float,
        ref_num: int,
        baseline_load: Optional[str] = None,
        policy_load: Optional[str] = None,
        alpha_decay: Optional[float] = None,
        discount_factor: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.ALPHA_DECAY = alpha_decay if alpha_decay else 1
        assert self.ALPHA_DECAY <= 1
        self.GAMMA = discount_factor if discount_factor else 1
        assert self.GAMMA <= 1

        self.plots_save = f"REINFORCE_actions/plots/{ref_num}"
        self.weights_save = f"REINFORCE_actions/weights/{ref_num}"
        os.makedirs(self.weights_save, exist_ok=True)
        self.id = ref_num

        self.feature_size = (FEATURE_POLYNOMIAL_ORDER + 1) ** 2
        if random_seed:
            np.random.seed(random_seed)
        self.baseline_weights = (
            np.load(baseline_load, allow_pickle=True)
            if baseline_load
            else np.random.normal(size=self.feature_size)
        )
        self.policy_weights = (
            np.load(policy_load, allow_pickle=True)
            if policy_load
            else np.random.normal(size=len(self.action_space) * self.feature_size)
        )
        self.memory_buffer = ExperienceBuffer(action_space_size=3, state_dimension=(2,))
        self.ALPHA_BASELINE = alpha_baseline
        self.ALPHA_POLICY = alpha_policy
        self.policy_plot = self.policy_weights
        self.baseline_plot = self.baseline_weights
        self.avg_delta_plot = np.array([])

    def action_probs(self, state: np.array) -> np.array:
        feature_vector = convert_to_basic_feature(state)
        action_feature_vectors = feature_to_action_feature(
            self.action_space, feature_vector
        )
        action_probs = get_action_probs(self.policy_weights, action_feature_vectors)
        action_probs = np.array(
            [1 if math.isnan(prob) else prob for prob in action_probs]
        )
        return action_probs

    def choose_action(self, state: np.array) -> List:
        return [np.random.choice(self.action_space, p=self.action_probs(state))]

    def calculate_returns(self) -> np.array:
        episode_len = self.memory_buffer.get_length()
        gammas = np.logspace(
            0, np.log10(self.GAMMA ** (episode_len - 1)), num=episode_len
        )
        future_rewards = self.memory_buffer.get_rewards()
        returns = []
        for timestep in range(episode_len):
            returns.append(np.sum(np.dot(future_rewards, gammas)))
            # Remove last element from gammas array, remove 1st element from rewards
            np.delete(future_rewards, 0)
            np.delete(gammas, -1)
        return np.array(returns)

    def gather_experience(self, env: gym.Env, time_limit: int) -> float:
        state = env.reset()
        done = False
        total_reward, reward, timesteps = 0, 0, 0

        while not done:
            # print(action_probs)
            action_probs = self.action_probs(state)
            # print(action_probs)

            action_chosen = np.random.choice(self.action_space, p=action_probs)

            self.memory_buffer.update(state, action_chosen, action_probs, reward)

            state, reward, done, info = env.step(action_chosen)
            total_reward += reward
            timesteps += 1

            if timesteps >= time_limit:
                break
        if not done:
            self.memory_buffer.rewards[-1] = -(1 / (1 - self.GAMMA))
            # self.memory_buffer.rewards[-1] = -5000
        env.close()
        # print("Episode of experience over, total reward = ", total_reward)
        return total_reward

    def update_weights(self, returns: np.array, step: int) -> None:
        """"""
        delta_sum = 0
        for timestep, state in enumerate(self.memory_buffer.states):
            basic_feature = convert_to_basic_feature(state)
            value = np.dot(self.baseline_weights, basic_feature)
            # print("v", value)
            delta = returns[timestep] - value
            # print("d", delta)
            self.baseline_weights += self.ALPHA_BASELINE * delta * basic_feature
            action_taken = self.memory_buffer.actions[timestep]
            # print("action_taken", action_taken)
            action_features = feature_to_action_feature(
                self.action_space, basic_feature
            )
            # print("action_features", action_features)
            action_probs = self.memory_buffer.action_probs[timestep]
            # print("action_probs", action_probs)
            grad_ln_policy = action_features[action_taken] - np.sum(
                action_features * action_probs.reshape((3, 1)), axis=0
            )
            # print(grad_ln_policy)
            self.policy_weights += self.ALPHA_POLICY * delta * grad_ln_policy
            delta_sum += delta
        self.baseline_plot = np.append(self.baseline_plot, self.baseline_weights)
        self.policy_plot = np.append(self.policy_plot, self.policy_weights)
        self.avg_delta_plot = np.append(
            self.avg_delta_plot, delta_sum / self.memory_buffer.get_length()
        )
        self.memory_buffer.clear()
        self.ALPHA_BASELINE *= self.ALPHA_DECAY
        self.ALPHA_POLICY *= self.ALPHA_DECAY

    def save(self) -> None:
        np.save(
            f"{self.weights_save}/baseline_weights_{self.id}.npy", self.baseline_weights
        )
        np.save(
            f"{self.weights_save}/policy_weights_{self.id}.npy", self.policy_weights
        )
        np.save(f"{self.plots_save}/baseline_plot_{self.id}.npy", self.baseline_plot)
        np.save(f"{self.plots_save}/policy_plot_{self.id}.npy", self.policy_plot)
        np.save(f"{self.plots_save}/avg_delta_plot_{self.id}.npy", self.avg_delta_plot)


def sanity_check(policy_load):
    actions = DISC_CONSTS.ACTION_SPACE
    policy_params = np.load(policy_load)
    points = [np.array([-1, 0]), np.array([0, 0.065]), np.array([0.3, -0.02])]

    for count, point in enumerate(points):
        feature = convert_to_basic_feature(point)
        afv = feature_to_action_feature(actions, feature)
        print(f"Point {count}:\n{point}\n{get_action_probs(policy_params, afv)}\n")


def train_policy(
    alpha_baseline: float,
    alpha_policy: float,
    num_steps: int,
    episode_length: int,
    discount_factor: float,
    alpha_decay: float,
    policy_load: Optional[str] = None,
    baseline_load: Optional[str] = None,
    trial: Optional = None,
    ref_num: Optional[int] = None,
    random_seed: Optional[int] = None,
):
    ref_num = ref_num if ref_num else trial.number if trial else 0
    env = gym.make("MountainCar-v0").env
    save_path = f"REINFORCE_actions/plots/{ref_num}"
    os.makedirs(save_path, exist_ok=True)

    policy = Policy(
        alpha_baseline=alpha_baseline,
        alpha_policy=alpha_policy,
        policy_load=policy_load,
        baseline_load=baseline_load,
        discount_factor=discount_factor,
        alpha_decay=alpha_decay,
        ref_num=ref_num,
        random_seed=random_seed,
    )
    moving_avg = np.array([-episode_length])
    rewards = np.array([-episode_length])

    def save_performance_plots():
        np.save(f"{save_path}/moving_avg_{ref_num}.npy", moving_avg)
        np.save(f"{save_path}/returns_{ref_num}.npy", rewards)

    try:
        for step in tqdm(range(num_steps)):
            total_reward = policy.gather_experience(env, episode_length)
            returns = policy.calculate_returns()
            policy.update_weights(returns, step)

            rewards = np.append(rewards, total_reward)
            moving_avg = np.append(
                moving_avg, 0.01 * total_reward + 0.99 * moving_avg[-1]
            )

            if step % 10 == 0:
                policy.save()
                save_performance_plots()

                # Output progress message
                print(
                    f"Trial {ref_num}, Step: {step}\tAvg: {moving_avg[-1]}\tAlpha_policy: {policy.ALPHA_POLICY}\tAlpha_baseline: {policy.ALPHA_BASELINE}"
                )

                if trial:
                    # Report progress to pruner
                    trial.report(moving_avg[-1], step)

                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.structs.TrialPruned()

            if abs(moving_avg[-1]) < 150:
                print(
                    f"Problem successfully solved - policy saved at {policy.plots_save}!"
                )
                break

    finally:
        policy.save()
        save_performance_plots()
    return moving_avg[-1]


def objective(
    num_steps: int,
    episode_length: int,
    trial: optuna.trial.Trial,
    policy_bounds: List[float],
    baseline_bounds: List[float],
    random_seed: int,
):
    alpha_policy = trial.suggest_loguniform(
        "alpha_policy", policy_bounds[0], policy_bounds[1]
    )
    alpha_baseline = trial.suggest_loguniform(
        "alpha_baseline", baseline_bounds[0], baseline_bounds[1]
    )
    print(
        f"Initialising trial {trial.number}, Alpha policy = {alpha_policy}, Alpha baseline = {alpha_baseline}\n"
    )
    return train_policy(
        alpha_baseline=alpha_baseline,
        alpha_policy=alpha_policy,
        num_steps=num_steps,
        trial=trial,
        alpha_decay=0.9999,
        discount_factor=0.999,
        episode_length=episode_length,
        random_seed=random_seed,
    )


def optimise_alpha(
    num_steps: int,
    policy_bounds: List[float],
    baseline_bounds: List[float],
    n_trials: int,
    percentile_kept: float = 66,
    random_seed: int = 0,
    episode_length: int = 10000,
):
    study = optuna.create_study(
        pruner=optuna.pruners.PercentilePruner(
            percentile=percentile_kept, n_warmup_steps=50
        ),
        direction="maximize",
    )
    study.optimize(
        lambda trial: objective(
            num_steps=num_steps,
            trial=trial,
            policy_bounds=policy_bounds,
            baseline_bounds=baseline_bounds,
            random_seed=random_seed,
            episode_length=episode_length,
        ),
        n_trials=n_trials,
        n_jobs=1,
    )


def main():
    # optimise_alpha(
    #     num_steps=250, policy_bounds=[1e-13, 1e-3], baseline_bounds=[1e-13, 1e-3], n_trials=40, percentile_kept=66
    # )
    train_policy(
        alpha_baseline=5e-6,
        alpha_policy=5e-7,
        num_steps=100000,
        alpha_decay=0.99999,
        discount_factor=0.9999,
        ref_num=2001,
        episode_length=10000,
        policy_load="REINFORCE_actions/weights/2000/policy_weights_2000.npy",
        baseline_load="REINFORCE_actions/weights/2000/baseline_weights_2000.npy",
    )

    # policy = Policy(baseline_save="REINFORCE/baseline_weights_p2.npy", policy_save="REINFORCE/policy_weights_p2.npy", alpha_baseline=1, alpha_policy=1)
    # test_solution(policy.choose_action)
    # sanity_check("REINFORCE/baseline_weights2.npy", "REINFORCE/policy_weights2.npy")


if __name__ == "__main__":
    main()
