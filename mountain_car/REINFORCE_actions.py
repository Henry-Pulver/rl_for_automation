import numpy as np
from typing import Optional, List
import plotly.graph_objects as go
from mountain_car_runner import DISC_CONSTS, test_solution
from buffer import ExperienceBuffer
import optuna
import gym

FEATURE_POLYNOMIAL_ORDER = 2


def convert_to_basic_feature(observation: np.array) -> np.array:
    assert observation.shape == (2,)
    p, v = observation
    return np.array(
        [
            (p ** n1) * (v ** n2)
            for n1 in range(FEATURE_POLYNOMIAL_ORDER + 1)
            for n2 in range(FEATURE_POLYNOMIAL_ORDER + 1)
        ]
    )


def feature_to_action_feature(actions: np.array, feature: np.array) -> np.array:
    all_action_features = np.zeros((len(actions), len(actions) * len(feature)))
    for action in actions:
        all_action_features[action][
            action * len(feature) : len(feature) + action * len(feature)
        ] = feature
    return all_action_features


def get_action_probs(policy_weights, action_feature_vectors) -> np.array:
    action_exponents = np.exp(np.matmul(action_feature_vectors, policy_weights))
    return action_exponents / np.sum(action_exponents)


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
    ) -> None:
        self.ALPHA_DECAY = alpha_decay if alpha_decay else 1
        assert self.ALPHA_DECAY <= 1
        self.GAMMA = discount_factor if discount_factor else 1
        assert self.GAMMA <= 1
        self.trial = trial
        if trial:
            self.baseline_save = f"REINFORCE/baseline_weights_optuna{trial.number}.npy"
            self.policy_save = f"REINFORCE/policy_weights_optuna{trial.number}.npy"
        else:
            self.baseline_save = (
                baseline_save
                if baseline_save
                else f"REINFORCE/baseline_weights_p{FEATURE_POLYNOMIAL_ORDER}.npy"
            )
            self.policy_save = (
                policy_save
                if policy_save
                else f"REINFORCE/policy_weights_p{FEATURE_POLYNOMIAL_ORDER}.npy"
            )
        self.feature_size = (FEATURE_POLYNOMIAL_ORDER + 1) ** 2
        self.baseline_weights = (
            np.load(self.baseline_save)
            if baseline_save
            else np.random.normal(size=self.feature_size)
        )
        self.policy_weights = (
            np.load(self.policy_save)
            if policy_save
            else np.random.normal(size=len(self.action_space) * self.feature_size)
        )
        self.memory_buffer = ExperienceBuffer()
        self.ALPHA_BASELINE = alpha_baseline
        self.ALPHA_POLICY = alpha_policy
        self.policy_plot = self.policy_weights
        self.baseline_plot = self.baseline_weights

    def choose_action(self, state: np.array) -> List:
        feature_vector = convert_to_basic_feature(state)
        action_feature_vectors = feature_to_action_feature(
            self.action_space, feature_vector
        )
        action_probs = get_action_probs(self.policy_weights, action_feature_vectors)
        return [np.random.choice(self.action_space, p=action_probs)]

    def calculate_returns(self) -> np.array:
        episode_len = self.memory_buffer.get_length()

        gammas = np.logspace(
            0, np.log10(self.GAMMA ** (episode_len - 1)), num=episode_len
        )
        returns = np.array([])
        for timestep in range(episode_len):
            future_rewards = self.memory_buffer.rewards[timestep:]
            returns = np.append(
                returns, np.sum(np.dot(future_rewards, gammas)))
            gammas = gammas[:-1]
        return returns

    def gather_experience(self, env: gym.Env, time_limit: int) -> float:
        state = env.reset()
        done = False
        total_reward, reward, timesteps = 0, 0, 0

        while not done:
            feature_vector = convert_to_basic_feature(state)
            action_feature_vectors = feature_to_action_feature(
                self.action_space, feature_vector
            )
            action_probs = get_action_probs(self.policy_weights, action_feature_vectors)
            action_chosen = np.random.choice(self.action_space, p=action_probs)

            self.memory_buffer.update(state, action_chosen, action_probs, reward)

            state, reward, done, info = env.step(action_chosen)
            total_reward += reward
            timesteps += 1

            if timesteps >= time_limit:
                break
        env.close()
        # print("Episode of experience over, total reward = ", total_reward)
        return total_reward

    def update_weights(self, returns: np.array) -> None:
        """"""
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
        self.ALPHA_BASELINE *= self.ALPHA_DECAY
        self.ALPHA_POLICY *= self.ALPHA_DECAY
        self.baseline_plot = np.append(self.baseline_plot, self.baseline_weights)
        self.policy_plot = np.append(self.policy_plot, self.policy_weights)
        self.save_plots()
        self.memory_buffer.clear()

    def save_weights(self) -> None:
        np.save(self.baseline_save, self.baseline_weights)
        np.save(self.policy_save, self.policy_weights)

    def save_plots(self) -> None:
        baseline_plot = self.baseline_plot.reshape(self.feature_size, -1).T
        policy_plot = self.policy_plot.reshape(
            self.feature_size * len(self.action_space), -1
        ).T
        if self.trial:
            np.save(
                f"REINFORCE_plots/optuna_baseline_plot_{self.trial.number}.npy",
                baseline_plot,
            )
            np.save(
                f"REINFORCE_plots/optuna_policy_plot_{self.trial.number}.npy",
                policy_plot,
            )
        else:
            np.save(
                f"REINFORCE_plots/baseline_plot_p{FEATURE_POLYNOMIAL_ORDER}.npy",
                baseline_plot,
            )
            np.save(
                f"REINFORCE_plots/policy_plot_p{FEATURE_POLYNOMIAL_ORDER}.npy",
                policy_plot,
            )


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
    discount_factor: float,
    alpha_decay: float,
    policy_save: Optional[str] = None,
    baseline_save: Optional[str] = None,
    trial: Optional = None,
    ref_num: Optional = None,
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
    )
    step = 0
    # moving_avg = np.load(f"REINFORCE_plots/moving_avg_2.npy")
    # rewards = np.load(f"REINFORCE_plots/returns_2.npy")
    moving_avg = np.array([-10000])
    rewards = np.array([-10000])

    def save_plots():
        if trial:
            np.save(f"REINFORCE_plots/optuna_moving_avg_{ref_num}.npy", moving_avg)
            np.save(f"REINFORCE_plots/optuna_returns_{ref_num}.npy", rewards)
        else:
            np.save(f"REINFORCE_plots/moving_avg_{ref_num}.npy", moving_avg)
            np.save(f"REINFORCE_plots/returns_{ref_num}.npy", rewards)

    try:
        while True:
            total_reward = policy.gather_experience(env, 10000)
            returns = policy.calculate_returns()
            policy.update_weights(returns)

            step += 1
            rewards = np.append(rewards, total_reward)
            moving_avg = np.append(
                moving_avg, 0.01 * total_reward + 0.99 * moving_avg[-1]
            )

            if step % 10 == 0:
                policy.save_weights()
                save_plots()

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
        save_plots()
    return moving_avg[-1]


def objective(
    num_steps: int,
    trial: optuna.trial.Trial,
    policy_bounds: List[float],
    baseline_bounds: List[float],
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
        discount_factor=1,
    )


def optimise_alpha(
    num_steps: int,
    policy_bounds: List[float],
    baseline_bounds: List[float],
    n_trials: int,
    percentile_kept: float,
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
        ),
        n_trials=n_trials,
        n_jobs=-1,
    )


def plot_conv_theta(filename: str) -> None:
    fig = go.Figure()
    y = np.load(f"REINFORCE_plots/baseline_plot_p3.npy").T
    x = np.linspace(0, y.shape[1], y.shape[1] + 1)
    for theta in y:
        fig.add_trace(go.Scatter(x=x, y=theta))
    fig.write_html(f"{filename}_baseline_plot_p3.html", auto_open=True)

    fig = go.Figure()
    y = np.load(f"REINFORCE_plots/policy_plot_p3.npy").T
    x = np.linspace(0, y.shape[1], y.shape[1] + 1)
    for theta in y:
        fig.add_trace(go.Scatter(x=x, y=theta))
    fig.write_html(f"{filename}_policy_plot_p3.html", auto_open=True)


def plot_conv_performance(filename: str, ref_num: int) -> None:
    fig = go.Figure()
    y = np.load(f"REINFORCE_plots/moving_avg_{ref_num}.npy")
    x = np.linspace(0, len(y), len(y) + 1)
    fig.add_trace(go.Scatter(x=x, y=y))
    fig.write_html(f"{filename}_moving_avg_{ref_num}.html", auto_open=True)

    fig = go.Figure()
    y = np.load(f"REINFORCE_plots/returns_{ref_num}.npy")
    x = np.linspace(0, len(y), len(y) + 1)
    fig.add_trace(go.Scatter(x=x, y=y))
    fig.write_html(f"{filename}_returns_{ref_num}.html", auto_open=True)


def plot_hyp_opt_performance(filename: str) -> None:
    runs_finished = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
    x = np.linspace(0, 200, 21)
    fig = go.Figure()
    for run_num in runs_finished:
        y = np.load(f"REINFORCE_plots/optuna_moving_avg_{run_num}.npy")
        fig.add_trace(go.Scatter(x=x, y=y, name=f"{run_num}"))
    fig.write_html(f"{filename}_moving_avg.html", auto_open=True)
    x = np.linspace(0, 200, 201)
    fig = go.Figure()
    for run_num in runs_finished:
        y = np.load(f"REINFORCE_plots/optuna_returns_{run_num}.npy")
        fig.add_trace(go.Scatter(x=x, y=y, name=f"{run_num}"))
    fig.write_html(f"{filename}_returns.html", auto_open=True)


def main():
    # optimise_alpha(
    #     num_steps=250, policy_bounds=[1e-13, 1e-3], baseline_bounds=[1e-13, 1e-3], n_trials=40, percentile_kept=66
    # )
    train_policy(
        alpha_baseline=1.1035831392989129e-12,
        alpha_policy=3.6786104643296704e-12,
        num_steps=100000,
        alpha_decay=0.9999,
        discount_factor=0.995,
        ref_num=0,
        policy_save="REINFORCE/policy_weights_p2.npy",
        baseline_save="REINFORCE/baseline_weights_p2.npy",
    )
    # plot_conv_theta("theta_68400")
    # plot_conv_performance("performance_68400", 3)

    # plot_conv_theta("filename")

    # plot_conv_performance("convergence_performance")
    # policy = Policy(baseline_save="REINFORCE/baseline_weights_p2.npy", policy_save="REINFORCE/policy_weights_p2.npy", alpha_baseline=1, alpha_policy=1)
    # test_solution(policy.choose_action)
    # sanity_check("REINFORCE/baseline_weights2.npy", "REINFORCE/policy_weights2.npy")


if __name__ == "__main__":
    main()
