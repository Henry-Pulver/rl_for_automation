import os
import numpy as np
import math
from tqdm import tqdm
import gym
from consts import DISC_CONSTS, CONSTS
from algorithms.buffer import ExperienceBuffer
from typing import Optional, List
import optuna

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
        alpha_baseline: Optional[float],
        alpha_policy: float,
        ref_num: int,
        baseline_load: Optional[str] = None,
        policy_load: Optional[str] = None,
        alpha_decay: Optional[float] = None,
        discount_factor: Optional[float] = None,
        random_seed: Optional[int] = None,
    ):
        self.ALPHA_DECAY = alpha_decay if alpha_decay else 1
        assert self.ALPHA_DECAY <= 1
        self.GAMMA = discount_factor if discount_factor else 1
        assert self.GAMMA <= 1

        folder_path = "REINFORCE_states"
        self.id = ref_num
        self.plots_save = f"{folder_path}/plots/{self.id}"
        self.weights_save = f"{folder_path}/weights/{self.id}"
        os.makedirs(self.plots_save, exist_ok=True)
        os.makedirs(self.weights_save, exist_ok=True)

        self.feature_size = (FEATURE_POLYNOMIAL_ORDER + 1) ** 2
        if random_seed is not None:
            np.random.seed(random_seed)
        self.policy_weights = (  # np.array([8.81451949, 9.16454747, -0.051268, 6.15630185, 7.01648719, -17.27751088, 8.85939796, -3.22674909, -3.97002752])
            np.load(policy_load)
            if policy_load
            else 10 * np.random.normal(size=self.feature_size)
        )
        self.baseline_weights = (
            (
                np.load(baseline_load)
                if baseline_load
                else 10 * np.random.normal(size=self.feature_size)
            )
            if alpha_baseline is not None
            else None
        )

        # print("Policy weights: ", self.policy_weights)
        # print("BL weights:", self.baseline_weights)
        self.memory_buffer = ExperienceBuffer(
            action_space_size=3, state_dimension=CONSTS.STATE_SPACE_SIZE
        )
        self.ALPHA_BASELINE = alpha_baseline
        self.ALPHA_POLICY = alpha_policy
        self.policy_plot = self.policy_weights
        self.baseline_plot = self.baseline_weights
        self.avg_delta_plot = np.array([])

    def action_probs(self, state: np.array):
        next_states = get_next_states(state, self.action_space)
        next_feature_vectors = convert_to_feature(next_states)
        action_probs = get_action_probs(self.policy_weights, next_feature_vectors)
        action_probs = np.array(
            [1 if math.isnan(prob) else prob for prob in action_probs]
        )
        return action_probs

    def choose_action(self, state: np.array) -> List[int]:
        """For use with test_solution() function"""
        action_probs = self.action_probs(state)
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

        if self.ALPHA_BASELINE is not None:
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

        # Wait 20 steps for baseline to settle
        if step > 20:
            approx_value = deltas if self.ALPHA_BASELINE is not None else returns
            self.policy_weights += self.ALPHA_POLICY * np.matmul(
                approx_value, grad_ln_policy
            )

        # if save_data:
        #     self.save_run_data(values, deltas, returns, step)

        self.policy_plot = np.append(self.policy_plot, self.policy_weights)
        if self.ALPHA_BASELINE is not None:
            self.baseline_plot = np.append(self.baseline_plot, self.baseline_weights)
            self.avg_delta_plot = np.append(self.avg_delta_plot, np.mean(deltas))
            self.ALPHA_BASELINE *= self.ALPHA_DECAY
        self.memory_buffer.clear()
        self.ALPHA_POLICY *= self.ALPHA_DECAY

    # def save_run_data(self, values, deltas, returns, step):
    #     states, actions, action_probs = self.memory_buffer.recall_memory()
    #     os.makedirs(f"REINFORCE_states/plots/{self.id}/{step}", exist_ok=True)
    #     np.save(f"REINFORCE_states/plots/{self.id}/{step}/values.npy", values)
    #     np.save(f"REINFORCE_states/plots/{self.id}/{step}/deltas.npy", deltas)
    #     np.save(
    #         f"REINFORCE_states/plots/{self.id}/{step}/states.npy", states,
    #     )
    #     np.save(
    #         f"REINFORCE_states/plots/{self.id}/{step}/action_probs.npy", action_probs,
    #     )
    #     np.save(
    #         f"REINFORCE_states/plots/{self.id}/{step}/actions.npy", actions,
    #     )
    #     np.save(
    #         f"REINFORCE_states/plots/{self.id}/{step}/rewards.npy",
    #         self.memory_buffer.get_rewards(),
    #     )
    #     np.save(f"REINFORCE_states/plots/{self.id}/{step}/returns.npy", returns)

    def save(self) -> None:
        np.save(
            f"{self.weights_save}/policy_weights_{self.id}.npy", self.policy_weights
        )
        np.save(f"{self.plots_save}/policy_plot_{self.id}.npy", self.policy_plot)
        if self.ALPHA_BASELINE is not None:
            np.save(
                f"{self.plots_save}/avg_delta_plot_{self.id}.npy", self.avg_delta_plot
            )
            np.save(
                f"{self.plots_save}/baseline_plot_{self.id}.npy", self.baseline_plot
            )
            np.save(
                f"{self.weights_save}/baseline_weights_{self.id}.npy",
                self.baseline_weights,
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
    alpha_policy: float,
    num_steps: int,
    episode_length: int,
    discount_factor: float,
    alpha_decay: float,
    alpha_baseline: Optional[float] = None,
    policy_load: Optional[str] = None,
    baseline_load: Optional[str] = None,
    trial: Optional = None,
    ref_num: Optional[int] = None,
    random_seed: Optional[int] = None,
):
    ref_num = ref_num if ref_num else trial.number if trial else 0
    env = gym.make("MountainCar-v0").env
    save_path = f"REINFORCE_states/plots/{ref_num}"

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
            total_reward = policy.gather_experience(env, 10000)
            returns = policy.calculate_returns()
            policy.update_weights(returns, save_data=(step % 100 == 0), step=step)

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
                    f"Problem successfully solved - policy saved at {policy.weights_save}!"
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
    percentile_kept: float,
    random_seed: int,
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
    #     num_steps=500,
    #     baseline_bounds=[1e-12, 0.5e-3],
    #     policy_bounds=[1e-12, 0.5e-3],
    #     n_trials=100,
    #     percentile_kept=66,
    #     random_seed=0,
    # )

    # test_solution(policy.choose_action)

    # load_ref_num = 41
    # load_path = f"REINFORCE_states/weights/{load_ref_num}"
    train_policy(
        alpha_baseline=None,
        alpha_policy=5,
        num_steps=1000000,
        episode_length=10000,
        alpha_decay=1,
        discount_factor=0.999,
        ref_num=50 + 1,
        # policy_load=f"{load_path}/policy_weights_{load_ref_num}.npy",
        # baseline_load=f"{load_path}/baseline_weights_{load_ref_num}.npy",
    )

    # sanity_check("REINFORCE_states/baseline_weights2.npy", "REINFORCE_states/policy_weights2.npy")


if __name__ == "__main__":
    main()
