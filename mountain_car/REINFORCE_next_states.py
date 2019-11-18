import os
import numpy as np
import gym
from mountain_car_runner import DISC_CONSTS, CONSTS, test_solution
from buffer import ExperienceBuffer
from typing import Optional, List
import optuna
import plotly.graph_objects as go

FEATURE_POLYNOMIAL_ORDER = 2


def convert_to_feature(state: np.array) -> np.array:
    # print(state)
    p, v = state.T
    return np.array(
        [
            (p ** n1) * (v ** n2)
            for n1 in range(FEATURE_POLYNOMIAL_ORDER + 1)
            for n2 in range(FEATURE_POLYNOMIAL_ORDER + 1)
        ]
    ).T


def get_next_states(states: np.array, actions: np.array) -> np.array:
    if len(states.shape) > 1:
        positions, velocities = (
            np.zeros((states.shape[0], actions.shape[0])),
            np.zeros((states.shape[0], actions.shape[0])),
        )
        positions += states.T[0].reshape((-1, 1))
        velocities += states.T[1].reshape((-1, 1))
    else:
        positions, velocities = (
            np.zeros((actions.shape[0])),
            np.zeros((actions.shape[0])),
        )
        positions += states[0]
        velocities += states[1]
    # print(f"positions: {positions}\t velocities: {velocities}")

    velocities += (actions - 1) * CONSTS.FORCE + np.cos(3 * positions) * (
        -CONSTS.GRAVITY
    )
    velocities = np.clip(velocities, -CONSTS.MAX_SPEED, CONSTS.MAX_SPEED)
    positions += velocities
    positions = np.clip(positions, CONSTS.MIN_POSITION, CONSTS.MAX_POSITION)
    # print(f"positions: {positions}\t velocities: {velocities}")
    return np.array([positions.T, velocities.T]).T  # .reshape((-1, 3, 2))


def get_action_probs(policy_weights, feature_vectors):
    action_exponents = np.matmul(feature_vectors, policy_weights)
    unnormalised_probs = np.exp(action_exponents - np.mean(action_exponents))
    return unnormalised_probs / np.sum(unnormalised_probs)


class Policy:
    action_space = DISC_CONSTS.ACTION_SPACE

    def __init__(
        self,
        alpha_baseline: float,
        alpha_policy: float,
        trial: Optional = None,
        baseline_save: Optional[str] = None,
        policy_save: Optional[str] = None,
        alpha_decay: Optional[float] = None,
        discount_factor: Optional[float] = None,
        ref_num: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        self.ALPHA_DECAY = alpha_decay if alpha_decay else 1
        assert self.ALPHA_DECAY <= 1
        self.GAMMA = discount_factor if discount_factor else 1
        assert self.GAMMA <= 1
        self.trial = trial
        weights_path = "REINFORCE_states/weights"
        if trial:
            self.id = self.trial.number if self.trial else -1
            os.makedirs(f"{weights_path}/{self.id}", exist_ok=True)
            self.baseline_save = f"{weights_path}/{self.id}/baseline_weights_optuna.npy"
            self.policy_save = f"{weights_path}/{self.id}/policy_weights_optuna.npy"
        else:
            self.id = ref_num if ref_num else 0
            self.baseline_save = (
                baseline_save
                if baseline_save
                else f"{weights_path}/baseline_weights_p{FEATURE_POLYNOMIAL_ORDER}.npy"
            )
            self.policy_save = (
                policy_save
                if policy_save
                else f"{weights_path}/policy_weights_p{FEATURE_POLYNOMIAL_ORDER}.npy"
            )
        self.feature_size = (FEATURE_POLYNOMIAL_ORDER + 1) ** 2

        if random_seed:
            np.random.seed(random_seed)
        self.baseline_weights = (
            np.load(self.baseline_save)
            if baseline_save
            else 10 * np.random.normal(size=self.feature_size)
        )
        self.policy_weights = (
            np.load(self.policy_save)
            if policy_save
            else 10 * np.random.normal(size=self.feature_size)
        )
        self.memory_buffer = ExperienceBuffer()
        self.ALPHA_BASELINE = alpha_baseline
        self.ALPHA_POLICY = alpha_policy
        self.policy_plot = self.policy_weights
        self.baseline_plot = self.baseline_weights

    def choose_action(self, state) -> List[int]:
        next_states = get_next_states(state, self.action_space)
        next_feature_vectors = convert_to_feature(next_states)
        action_probs = get_action_probs(self.policy_weights, next_feature_vectors)
        return [np.random.choice(self.action_space, p=action_probs)]

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
            # print("returns: ", returns[-1])
        # print(returns)
        return np.array(returns)

    def gather_experience(self, env: gym.Env, time_limit: int) -> float:
        state = env.reset()
        done = False
        total_reward, reward, timesteps = 0, 0, 0

        while not done:
            next_states = get_next_states(state, self.action_space)
            # print("Next states:", next_states)
            next_feature_vectors = convert_to_feature(next_states)
            action_probs = get_action_probs(self.policy_weights, next_feature_vectors)
            # print(action_probs)
            action_chosen = np.random.choice(self.action_space, p=action_probs)

            self.memory_buffer.update(state, action_chosen, action_probs, reward)

            state, reward, done, info = env.step(action_chosen)
            total_reward += reward
            timesteps += 1

            if timesteps >= time_limit:
                break
        if not done:
            self.memory_buffer.rewards[-1] = -999
        env.close()
        # print("Episode of experience over, total reward = ", total_reward)
        return total_reward

    def update_weights(
        self, returns: np.array, save_data: bool = False, step: Optional[int] = None
    ) -> None:
        """"""
        # print(f"Updating weights. Current policy value: {self.policy_weights}\n")
        states, actions, action_probs = self.memory_buffer.recall_memory()
        basic_features = convert_to_feature(states)
        values = np.matmul(self.baseline_weights, basic_features.T)
        # print(f"Values: {values}\nValues shape: {values.shape}\n")
        deltas = returns - values
        # print(f"Deltas: {deltas}\n")
        delta_baseline = self.ALPHA_BASELINE * np.matmul(deltas, basic_features)
        # print(f"delta_baseline: {delta_baseline}\n")
        self.baseline_weights += delta_baseline
        next_states = get_next_states(states, self.action_space)
        # print(f"Next states: {next_states}\n")
        next_feature_vectors = convert_to_feature(next_states)
        # print(f"Next feature vectors: {next_feature_vectors}\n")
        steps = np.array(range(next_feature_vectors.shape[0]))
        chosen_features = next_feature_vectors[steps, actions]
        # print(f"Chosen features: {chosen_features}")
        grad_ln_policy = chosen_features - np.sum(
            action_probs.reshape((-1, DISC_CONSTS.ACTION_SPACE.shape[0], 1))
            * next_feature_vectors,
            axis=1,
        )
        # print(f"Grad ln policy: {grad_ln_policy}")
        self.policy_weights += self.ALPHA_POLICY * np.matmul(deltas, grad_ln_policy)

        if save_data:
            self.save_run_data(values, deltas, returns, step)
        self.baseline_plot = np.append(self.baseline_plot, self.baseline_weights)
        self.policy_plot = np.append(self.policy_plot, self.policy_weights)
        self.memory_buffer.clear()
        self.ALPHA_BASELINE *= self.ALPHA_DECAY
        self.ALPHA_POLICY *= self.ALPHA_DECAY

    def save_run_data(self, values, deltas, returns, step):
        states, actions, action_probs = self.memory_buffer.recall_memory()
        os.makedirs(f"REINFORCE_states/plots/{self.id}/{step}", exist_ok=True)
        np.save(f"REINFORCE_states/plots/{self.id}/{step}/values.npy", values)
        np.save(f"REINFORCE_states/plots/{self.id}/{step}/deltas.npy", deltas)
        np.save(
            f"REINFORCE_states/plots/{self.id}/{step}/states.npy", states,
        )
        np.save(
            f"REINFORCE_states/plots/{self.id}/{step}/action_probs.npy", action_probs,
        )
        np.save(
            f"REINFORCE_states/plots/{self.id}/{step}/actions.npy", actions,
        )
        np.save(
            f"REINFORCE_states/plots/{self.id}/{step}/rewards.npy",
            self.memory_buffer.get_rewards(),
        )
        np.save(f"REINFORCE_states/plots/{self.id}/{step}/returns.npy", returns)

    def save_weights(self) -> None:
        np.save(self.baseline_save, self.baseline_weights)
        np.save(self.policy_save, self.policy_weights)

    def save_param_plots(self) -> None:
        if self.trial:
            np.save(
                f"REINFORCE_states/plots/optuna_baseline_plot_{self.id}.npy",
                self.baseline_plot,
            )
            np.save(
                f"REINFORCE_states/plots/optuna_policy_plot_{self.id}.npy",
                self.policy_plot,
            )
        else:
            np.save(
                f"REINFORCE_states/plots/baseline_plot_{self.id}.npy",
                self.baseline_plot,
            )
            np.save(
                f"REINFORCE_states/plots/policy_plot_{self.id}.npy", self.policy_plot,
            )


def sanity_check(policy_load):
    actions = DISC_CONSTS.ACTION_SPACE
    policy_params = np.load(policy_load)
    points = [np.array([-1, 0]), np.array([0, 0.065]), np.array([0.3, -0.02])]

    for count, point in enumerate(points):
        next_states = get_next_states(point, actions)
        next_features = convert_to_feature(next_states)
        print(
            f"Point {count}:\n{point}\n{get_action_probs(policy_params, next_features)}\n"
        )


def train_policy(
    alpha_baseline: float,
    alpha_policy: float,
    num_steps: int,
    discount_factor: float,
    alpha_decay: float,
    policy_save: Optional[str] = None,
    baseline_save: Optional[str] = None,
    trial: Optional = None,
    ref_num: Optional[int] = None,
    random_seed: Optional[int] = None,
):
    ref_num = ref_num if ref_num else trial.number + 1 if trial else 0
    env = gym.make("MountainCar-v0").env
    policy = Policy(
        alpha_baseline=alpha_baseline,
        alpha_policy=alpha_policy,
        trial=trial,
        policy_save=policy_save,
        baseline_save=baseline_save,
        discount_factor=discount_factor,
        alpha_decay=alpha_decay,
        ref_num=ref_num,
        random_seed=random_seed,
    )
    step = 0
    moving_avg = np.array([-10000])
    rewards = np.array([-10000])

    def save_performance_plots():
        if trial:
            np.save(
                f"REINFORCE_states/plots/optuna_moving_avg_{ref_num - 1}.npy",
                moving_avg,
            )
            np.save(f"REINFORCE_states/plots/optuna_returns_{ref_num - 1}.npy", rewards)
        else:
            np.save(f"REINFORCE_states/plots/moving_avg_{ref_num}.npy", moving_avg)
            np.save(f"REINFORCE_states/plots/returns_{ref_num}.npy", rewards)

    try:
        while True:
            total_reward = policy.gather_experience(env, 10000)
            returns = policy.calculate_returns()
            policy.update_weights(returns, save_data=(step % 100 == 0), step=step)

            step += 1
            rewards = np.append(rewards, total_reward)
            moving_avg = np.append(
                moving_avg, 0.01 * total_reward + 0.99 * moving_avg[-1]
            )

            if step % 10 == 0:
                policy.save_weights()
                save_performance_plots()
                policy.save_param_plots()

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
                    f"Problem successfully solved - policy saved at {policy.policy_save}!"
                )
                break

            if step >= num_steps:
                break
    finally:
        policy.save_weights()
        save_performance_plots()
    return moving_avg[-1]


def objective(
    num_steps: int,
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
        random_seed=random_seed,
    )


def optimise_alpha(
    num_steps: int,
    policy_bounds: List[float],
    baseline_bounds: List[float],
    n_trials: int,
    percentile_kept: float,
    random_seed: int,
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
        ),
        n_trials=n_trials,
        n_jobs=1,
    )


def main():
    optimise_alpha(
        num_steps=500,
        baseline_bounds=[1e-12, 0.5e-3],
        policy_bounds=[1e-12, 0.5e-3],
        n_trials=100,
        percentile_kept=66,
        random_seed=0,
    )

    # test_solution(policy.choose_action)

    # train_policy(
    #     alpha_baseline=1e-2,
    #     alpha_policy=1e-2,
    #     num_steps=1000,
    #     alpha_decay=0.9999,
    #     discount_factor=0.999,
    #     ref_num=70,
    # )

    # sanity_check("REINFORCE_states/baseline_weights2.npy", "REINFORCE_states/policy_weights2.npy")


if __name__ == "__main__":
    main()
