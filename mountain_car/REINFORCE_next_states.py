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
    p, v = state.T
    return np.array(
        [
            (p ** n1) * (v ** n2)
            for n1 in range(FEATURE_POLYNOMIAL_ORDER + 1)
            for n2 in range(FEATURE_POLYNOMIAL_ORDER + 1)
        ]
    ).T


def get_next_states(state: np.array, actions: np.array) -> np.array:
    positions, velocities = np.zeros(actions.shape), np.zeros(actions.shape)
    positions += state[0]
    velocities += state[1]

    velocities += (actions - 1) * CONSTS.FORCE + np.cos(3 * positions) * (
        -CONSTS.GRAVITY
    )
    velocities = np.clip(velocities, -CONSTS.MAX_SPEED, CONSTS.MAX_SPEED)
    positions += velocities
    positions = np.clip(positions, CONSTS.MIN_POSITION, CONSTS.MAX_POSITION)
    return np.array([positions, velocities]).T


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
    ):
        self.ALPHA_DECAY = alpha_decay if alpha_decay else 1
        assert self.ALPHA_DECAY <= 1
        self.GAMMA = discount_factor if discount_factor else 1
        assert self.GAMMA <= 1
        self.trial = trial
        if trial:
            self.id = self.trial.number if self.trial else -1
            os.makedirs(f"REINFORCE_states/weights/{self.id}", exist_ok=True)
            self.baseline_save = (
                f"REINFORCE_states/weights/{self.id}/baseline_weights_optuna.npy"
            )
            self.policy_save = (
                f"REINFORCE_states/weights/{self.id}/policy_weights_optuna.npy"
            )
        else:
            self.id = ref_num if ref_num else 0
            self.baseline_save = (
                baseline_save
                if baseline_save
                else f"REINFORCE_states/baseline_weights_p{FEATURE_POLYNOMIAL_ORDER}.npy"
            )
            self.policy_save = (
                policy_save
                if policy_save
                else f"REINFORCE_states/policy_weights_p{FEATURE_POLYNOMIAL_ORDER}.npy"
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
            else np.random.normal(size=self.feature_size)
        )
        self.memory_buffer = ExperienceBuffer()
        self.ALPHA_BASELINE = alpha_baseline
        self.ALPHA_POLICY = alpha_policy
        self.policy_plot = self.policy_weights
        self.baseline_plot = self.baseline_weights

    def choose_action(self, state):
        next_states = get_next_states(state, self.action_space)
        next_feature_vectors = convert_to_feature(next_states)
        action_probs = get_action_probs(self.policy_weights, next_feature_vectors)
        return [np.random.choice(self.action_space, p=action_probs)]

    def calculate_returns(self):
        episode_len = self.memory_buffer.get_length()
        gammas = np.logspace(
            0, np.log10(self.GAMMA ** (episode_len - 1)), num=episode_len
        )
        returns = np.array([])
        for timestep in range(episode_len):
            future_rewards = self.memory_buffer.rewards[timestep:]
            # print("future rewards: ", future_rewards)
            returns = np.append(returns, np.sum(np.dot(future_rewards, gammas)))
            gammas = gammas[:-1]
            # print("returns: ", returns[-1])
        # print(returns)
        return returns

    def gather_experience(self, env: gym.Env, time_limit: int) -> float:
        state = env.reset()
        done = False
        total_reward, reward, timesteps = 0, 0, 0

        while not done:
            next_states = get_next_states(state, self.action_space)
            next_feature_vectors = convert_to_feature(next_states)
            action_probs = get_action_probs(self.policy_weights, next_feature_vectors)
            action_chosen = np.random.choice(self.action_space, p=action_probs)

            self.memory_buffer.update(state, action_chosen, action_probs, reward)

            state, reward, done, info = env.step(action_chosen)
            total_reward += reward
            timesteps += 1

            if timesteps >= time_limit:
                break
        # if not done:
        #     self.memory_buffer.rewards[-1] = -200
        env.close()
        # print("Episode of experience over, total reward = ", total_reward)
        return total_reward

    def update_weights(
        self, returns: np.array, save_data: bool = False, step: Optional[int] = None
    ) -> None:
        """"""
        # print(f"Updating weights. Current policy value: {self.policy_weights}\n")
        if save_data:
            values = np.array([])
            deltas = np.array([])
            baseline_weights = np.array([])
            policy_weights = np.array([])

        for timestep, state in enumerate(self.memory_buffer.states):
            basic_feature = convert_to_feature(state)
            value = np.dot(self.baseline_weights, basic_feature)
            delta = returns[timestep] - value
            self.baseline_weights += self.ALPHA_BASELINE * delta * basic_feature
            action_taken = self.memory_buffer.actions[timestep]
            action_probs = self.memory_buffer.action_probs[timestep]
            next_states = get_next_states(state, self.action_space)
            next_feature_vectors = convert_to_feature(next_states)
            grad_ln_policy = next_feature_vectors[action_taken] - np.sum(
                next_feature_vectors * action_probs.reshape((3, 1)), axis=0
            )
            # print(grad_ln_policy)
            self.policy_weights += self.ALPHA_POLICY * delta * grad_ln_policy
            if save_data:
                values = np.append(values, value)
                deltas = np.append(deltas, delta)
                baseline_weights = np.append(baseline_weights, self.baseline_weights)
                policy_weights = np.append(policy_weights, self.policy_weights)

        # print(f"Current policy value: {self.policy_weights}\n")
        if save_data:
            self.save_run_data(
                values, deltas, baseline_weights, policy_weights, returns, step
            )
        self.baseline_plot = np.append(self.baseline_plot, self.baseline_weights)
        self.policy_plot = np.append(self.policy_plot, self.policy_weights)
        self.memory_buffer.clear()
        self.ALPHA_BASELINE *= self.ALPHA_DECAY
        self.ALPHA_POLICY *= self.ALPHA_DECAY

    def save_run_data(
        self, values, deltas, baseline_weights, policy_weights, returns, step
    ):
        os.makedirs(f"REINFORCE_states/plots/{self.id}/{step}", exist_ok=True)
        np.save(f"REINFORCE_states/plots/{self.id}/{step}/values.npy", values)
        np.save(f"REINFORCE_states/plots/{self.id}/{step}/deltas.npy", deltas)
        np.save(
            f"REINFORCE_states/plots/{self.id}/{step}/baseline_weights.npy",
            baseline_weights,
        )
        np.save(
            f"REINFORCE_states/plots/{self.id}/{step}/policy_weights.npy",
            policy_weights,
        )
        np.save(
            f"REINFORCE_states/plots/{self.id}/{step}/states.npy",
            self.memory_buffer.states,
        )
        np.save(
            f"REINFORCE_states/plots/{self.id}/{step}/action_probs.npy",
            self.memory_buffer.action_probs,
        )
        np.save(
            f"REINFORCE_states/plots/{self.id}/{step}/actions.npy",
            self.memory_buffer.actions,
        )
        np.save(
            f"REINFORCE_states/plots/{self.id}/{step}/rewards.npy",
            self.memory_buffer.rewards,
        )
        np.save(f"REINFORCE_states/plots/{self.id}/{step}/returns.npy", returns)

    def save_weights(self) -> None:
        # print(f"Saving weights. Current policy value: {self.policy_weights}\n")
        np.save(self.baseline_save, self.baseline_weights)
        np.save(self.policy_save, self.policy_weights)
        # print(f"Current policy value: {self.policy_weights}\n")

    def save_param_plots(self) -> None:
        # print(self.baseline_plot[0])
        # print(self.policy_plot)
        baseline_plot = self.baseline_plot.reshape(self.feature_size, -1).T
        # print(baseline_plot)
        policy_plot = self.policy_plot.reshape(self.feature_size, -1).T
        # print(policy_plot, "\n")
        if self.trial:
            np.save(
                f"REINFORCE_states/plots/optuna_baseline_plot_{self.trial.number}.npy",
                baseline_plot,
            )
            np.save(
                f"REINFORCE_states/plots/optuna_policy_plot_{self.trial.number}.npy",
                policy_plot,
            )
        else:
            np.save(
                f"REINFORCE_states/plots/baseline_plot_p{FEATURE_POLYNOMIAL_ORDER}.npy",
                baseline_plot,
            )
            np.save(
                f"REINFORCE_states/plots/policy_plot_p{FEATURE_POLYNOMIAL_ORDER}.npy",
                policy_plot,
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
        ref_num=ref_num,
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
        discount_factor=0.995,
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
        n_jobs=1,
    )


def plot_weights_and_performance(filename: str) -> None:
    fig = go.Figure()
    y = np.load(f"REINFORCE_states/plots/optuna_baseline_plot_4.npy").T
    x = np.linspace(0, y.shape[1], y.shape[1] + 1)
    for theta in y:
        fig.add_trace(go.Scatter(x=x, y=theta))
    fig.write_html(
        f"REINFORCE_states/plots/{filename}_states_baseline_plot.html", auto_open=True
    )

    fig = go.Figure()
    y = np.load(f"REINFORCE_states/plots/optuna_policy_plot_4.npy").T
    x = np.linspace(0, y.shape[1], y.shape[1] + 1)
    for theta in y:
        fig.add_trace(go.Scatter(x=x, y=theta))
    fig.write_html(
        f"REINFORCE_states/plots/{filename}_states_policy_plot.html", auto_open=True
    )

    fig = go.Figure()
    y = np.load(f"REINFORCE_states/plots/optuna_moving_avg_4.npy").T
    x = np.linspace(0, y.shape[0], y.shape[0] + 1)
    fig.add_trace(go.Scatter(x=x, y=y))
    fig.write_html(
        f"REINFORCE_states/plots/{filename}_states_moving_avg.html", auto_open=True
    )

    fig = go.Figure()
    y = np.load(f"REINFORCE_states/plots/optuna_returns_4.npy").T
    x = np.linspace(0, y.shape[0], y.shape[0] + 1)
    fig.add_trace(go.Scatter(x=x, y=y))
    fig.write_html(
        f"REINFORCE_states/plots/{filename}_states_returns.html", auto_open=True
    )


def plot_run() -> None:
    files = os.listdir("REINFORCE_states/plots/0")
    for file in files:
        print(file)
        if not file.endswith(".npy"):
            files.remove(file)
        else:
            fig = go.Figure()
            y = np.load(f"REINFORCE_states/plots/0/{file}", allow_pickle=True)
            if len(y.shape) > 1:
                y = y.T
                print(y.shape)
                x = np.linspace(0, y.shape[1], y.shape[1] + 1)
                for theta in y:
                    fig.add_trace(go.Scatter(x=x, y=theta,))
            else:
                x = np.linspace(0, y.shape[0], y.shape[0] + 1)
                fig.add_trace(go.Scatter(x=x, y=y))
            file = os.path.splitext(file)[0]
            fig.write_html(f"REINFORCE_states/plots/0/{file}.html", auto_open=True)


def main():
    # optimise_alpha(
    #     num_steps=500,
    #     baseline_bounds=[1e-6, 1e-2],
    #     policy_bounds=[1e-6, 1e-2],
    #     n_trials=40,
    #     percentile_kept=66,
    # )

    train_policy(
        alpha_baseline=1.73e-3,
        alpha_policy=2.24e-3,
        num_steps=100000,
        alpha_decay=0.9975,
        discount_factor=0.995,
        ref_num=0
    )

    # plot_weights_and_performance("16-11")
    # test_solution(policy.choose_action)
    # sanity_check("REINFORCE_states/baseline_weights2.npy", "REINFORCE_states/policy_weights2.npy")
    # plot_run()
    # print(np.random.choice([0, 1, 2], p=[0, 0.5, 5]))


if __name__ == "__main__":
    main()
