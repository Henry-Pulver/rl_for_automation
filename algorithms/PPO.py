import numpy as np
import torch
import torch.nn as nn
import datetime

from torch.functional import F
import gym
from collections import namedtuple

from typing import Tuple, List, Optional
from pathlib import Path

from algorithms.action_chooser import ActionChooser
from algorithms.actor_critic import ActorCritic, ActorCriticParams
from algorithms.advantage_estimation import get_td_error, get_gae
from algorithms.buffer import PPOExperienceBuffer
from algorithms.utils import generate_save_location, generate_ppo_hyp_str

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


hyp_names = (
    "gamma",
    "T",
    "num_epochs",
    "learning_rate",
    "c1",
    "c2",
    "lamda",
    "epsilon",
    "beta",
    "d_targ",
)
try:
    HyperparametersPPO = namedtuple(
        "HyperparametersPPO", hyp_names, defaults=(None,) * len(hyp_names),
    )
except TypeError:
    HyperparametersPPO = namedtuple("HyperparametersPPO", hyp_names)
    HyperparametersPPO.__new__.__defaults__ = (None,) * len(hyp_names)
"""
    gamma: Discount factor for time delay in return.
    T: Time horizon.
    num_epochs: Number of epochs of learning carried out on each T timesteps.
    learning_rate: Learning rate of Adam optimizer on Actor and Critic.
    c1: Value function loss weighting factor.
    c2: (Optional) Entropy bonus loss term weighting factor.
    lamda: (Optional) GAE weighting factor.
    epsilon: (Optional) PPO clipping parameter.
    beta: (Optional) KL penalty parameter.
    d_targ: (Optional) Adaptive KL target.
"""

hyperparams_ppo_atari = HyperparametersPPO(
    gamma=0.99,
    lamda=0.95,
    learning_rate=2e-3,
    T=1024,
    epsilon=0.2,
    c2=0.01,
    num_epochs=3,
)


class PPO:
    PPO_TYPES = ["clip", "adaptive_KL", "fixed_KL", "unclipped"]
    ADVANTAGE_TYPES = ["monte_carlo", "monte_carlo_baseline", "gae"]

    def __init__(
        self,
        state_dimension: Tuple,
        action_space: int,
        save_path: Path,
        hyperparameters: HyperparametersPPO,
        actor_critic_params: ActorCriticParams,
        param_plot_num: int,
        entropy: bool = True,
        ppo_type: str = "clip",
        advantage_type: str = "monte_carlo_baseline",
        policy_burn_in: int = 0,
    ):
        assert ppo_type in self.PPO_TYPES
        assert advantage_type in self.ADVANTAGE_TYPES
        self.ppo_type = ppo_type
        self.adv_type = advantage_type
        self.hyp = hyperparameters
        if self.ppo_type[-2:] == "KL":
            self.beta = self.hyp.beta
        if self.adv_type == "gae":
            assert self.hyp.lamda is not None
        self.param_sharing = actor_critic_params.num_shared_layers is not None

        self.policy_burn_in = policy_burn_in
        self.lr = self.hyp.learning_rate
        self.gamma = self.hyp.gamma
        self.eps_clip = self.hyp.epsilon
        self.K_epochs = self.hyp.num_epochs
        self.entropy = entropy

        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.policy = ActorCritic(
            state_dimension, action_space, actor_critic_params,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr,)
        self.policy_old = ActorCritic(
            state_dimension, action_space, actor_critic_params,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.determine_plotted_params(
            param_plot_num,
            actor_critic_params.actor_layers,
            actor_critic_params.critic_layers,
            state_dimension,
        )
        self.shared_plot = []
        self.actor_plot = []
        self.critic_plot = []
        self.loss_plots = {"entropy_loss": [], "clipped_loss": [], "value_loss": []}

        self.MseLoss = nn.MSELoss()

    def update(self, buffer: PPOExperienceBuffer, ep_num: int):
        # Monte Carlo estimate of state rewards:
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(buffer.rewards), reversed(buffer.is_terminal)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        # Normalizing the rewards:
        returns = torch.tensor(returns).to(device)
        returns_mean = returns.mean()
        returns_std_dev = returns.std() + 1e-5
        norm_returns = (returns - returns_mean) / returns_std_dev

        # convert list to tensor
        old_states = torch.stack(buffer.states).to(device).detach()
        old_actions = torch.stack(buffer.actions).to(device).detach()
        old_logprobs = torch.stack(buffer.log_probs).to(device).detach()
        old_probs = torch.stack(buffer.action_probs).to(device).detach()

        # Optimize policy for K epochs:
        for k in range(self.K_epochs):

            loss = self.calculate_loss(
                old_states,
                old_actions,
                old_logprobs,
                old_probs,
                norm_returns,
                buffer,
                ep_num >= self.policy_burn_in,
                returns_mean,
                returns_std_dev,
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.record_policy_params()
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def calculate_loss(
        self,
        states,
        actions,
        old_logprobs,
        old_probs,
        norm_returns,
        buffer,
        update,
        returns_mean,
        returns_std_dev,
    ):
        # Evaluating old actions and values :
        logprobs, state_values, dist_entropy, action_probs = self.policy.evaluate(
            states, actions
        )

        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(logprobs - old_logprobs)

        # Finding Loss:
        loss = 0
        advantages = self.calculate_advantages(
            norm_returns, state_values.detach(), buffer, returns_mean, returns_std_dev
        )
        surr1 = ratios * advantages
        if self.ppo_type == "clip":
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            clipped_loss = -torch.min(surr1, surr2)
            main_loss = clipped_loss
        elif self.ppo_type == "fixed_KL":
            d = F.kl_div(old_probs, action_probs)
            main_loss = -surr1 + self.hyp.beta * d
        elif self.ppo_type == "adaptive_KL":
            d = F.kl_div(old_probs, action_probs)
            if d < self.hyp.d_targ / 1.5:
                self.beta /= 2
            elif d > self.hyp.d_targ * 1.5:
                self.beta *= 2
            main_loss = -surr1 + self.beta * d
        elif self.ppo_type == "unclipped":
            main_loss = -surr1
        if update:
            loss += main_loss
        value_loss = self.hyp.c1 * self.MseLoss(state_values, norm_returns)
        loss += value_loss

        if self.entropy:
            loss -= self.hyp.c2 * dist_entropy.mean()

        self.record_losses(
            main_loss.mean().detach().numpy(),
            -0.01 * dist_entropy.mean().detach().numpy(),
            value_loss.detach().numpy(),
        )

        return loss

    def calculate_advantages(
        self,
        norm_returns,
        state_values: torch.tensor,
        buffer: PPOExperienceBuffer,
        returns_mean: torch.Tensor,
        returns_std_dev: torch.Tensor,
    ):
        if self.adv_type == "monte_carlo":
            advantages = norm_returns
        elif self.adv_type == "monte_carlo_baseline":
            advantages = norm_returns - state_values
        elif self.adv_type == "gae":
            scaled_state_values = (state_values * returns_std_dev) + returns_mean
            td_errors = get_td_error(
                scaled_state_values.numpy(), buffer, self.hyp.gamma
            )
            unnormalised_advs = torch.from_numpy(
                get_gae(td_errors, buffer.is_terminal, self.hyp.gamma, self.hyp.lamda)
            ).to(device)
            advantages = (unnormalised_advs - unnormalised_advs.mean()) / (
                unnormalised_advs.std() + 1e-5
            )
        else:
            raise ValueError("Invalid advantage type used!")
        return advantages

    def sample_nn_params(self):
        """Gets randomly sampled actor NN parameters from 1st layer."""
        shared_params = None
        actor_params = self.policy.state_dict()["actor_layers.0.weight"].numpy()[
            self.actor_params_x, self.actor_params_y
        ]
        critic_params = self.policy.state_dict()["critic_layers.0.weight"].numpy()[
            self.critic_params_x, self.critic_params_y
        ]
        if self.param_sharing:
            shared_params = self.policy.state_dict()["shared_layers.0.weight"].numpy()[
                self.shared_params_x, self.shared_params_y
            ]
        return actor_params, critic_params, shared_params

    def record_policy_params(self):
        actor_params, critic_params, shared_params = self.sample_nn_params()
        self.actor_plot.append(actor_params)
        self.critic_plot.append(critic_params)
        if self.param_sharing:
            self.shared_plot.append(shared_params)

    def record_losses(self, main_loss, entropy_loss, value_loss):
        self.loss_plots["clipped_loss"].append(main_loss)
        self.loss_plots["value_loss"].append(value_loss)
        if self.entropy:
            self.loss_plots["entropy_loss"].append(entropy_loss)

    def save(self):
        np.save(
            f"{self.save_path}/policy_params.npy", np.array(self.actor_plot),
        )
        np.save(
            f"{self.save_path}/critic_params.npy", np.array(self.critic_plot),
        )
        if self.param_sharing:
            np.save(
                f"{self.save_path}/shared_params.npy", np.array(self.shared_plot),
            )
        # LEGACY - main loss saved as "clipped loss" even tho not necessarily clipped
        np.save(
            f"{self.save_path}/mean_clipped_loss.npy",
            np.array(self.loss_plots["clipped_loss"]),
        )
        np.save(
            f"{self.save_path}/mean_value_loss.npy",
            np.array(self.loss_plots["value_loss"]),
        )
        if self.entropy:
            np.save(
                f"{self.save_path}/mean_entropy_loss.npy",
                np.array(self.loss_plots["entropy_loss"]),
            )

    def determine_plotted_params(
        self,
        param_plot_num: int,
        actor_layers: Tuple,
        critic_layers,
        state_dimension: Tuple,
    ):
        if self.param_sharing:
            self.shared_params_x = np.random.randint(
                low=0, high=actor_layers[0], size=param_plot_num
            )
            self.shared_params_y = np.random.randint(
                low=0, high=state_dimension[0], size=param_plot_num
            )
            actor_x_max = actor_layers[-1]
            actor_y_max = actor_layers[-2]
            critic_x_max = critic_layers[-1]
            critic_y_max = critic_layers[-2]
        else:
            actor_x_max = actor_layers[0]
            actor_y_max = state_dimension[0]
            critic_x_max = critic_layers[0]
            critic_y_max = state_dimension[0]
        self.actor_params_x = np.random.randint(
            low=0, high=actor_x_max, size=param_plot_num
        )
        self.actor_params_y = np.random.randint(
            low=0, high=actor_y_max, size=param_plot_num
        )
        self.critic_params_x = np.random.randint(
            low=0, high=critic_x_max, size=param_plot_num
        )
        self.critic_params_y = np.random.randint(
            low=0, high=critic_y_max, size=param_plot_num
        )


def plot_episode_reward(save_path: Path, running_rewards: List):
    np.save(f"{save_path}/rewards.npy", np.array(running_rewards))


def train_ppo(
    env_name: str,
    max_episodes: int,
    update_timestep: int,
    log_interval: int,
    hyp: HyperparametersPPO,
    actor_critic_params: ActorCriticParams,
    solved_reward: float,
    random_seeds: List,
    load_path: Optional[str] = None,
    max_timesteps: Optional[int] = None,
    render: bool = False,
    verbose: bool = False,
    ppo_type: str = "clip",
    advantage_type: str = "monte_carlo_baseline",
    date: Optional[str] = None,
    param_plot_num: int = 2,
    policy_burn_in: int = 0,
    chooser_params: Tuple = (None, None, None),
):
    env = gym.make(env_name).env
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    episode_numbers = []
    for random_seed in random_seeds:
        if random_seed is not None:
            torch.manual_seed(random_seed)
            env.seed(random_seed)
            print(f"Set random seed to: {random_seed}")

        buffer = PPOExperienceBuffer(state_dim, action_dim)

        hyp_str = generate_ppo_hyp_str(ppo_type, hyp)
        save_path = generate_save_location(
            Path("data"),
            actor_critic_params.actor_layers,
            f"PPO-{ppo_type}",
            env_name,
            random_seed,
            hyp_str,
            date,
        )
        save_path.mkdir(parents=True, exist_ok=True)

        ppo = PPO(
            state_dimension=state_dim,
            action_space=action_dim,
            actor_critic_params=actor_critic_params,
            hyperparameters=hyp,
            save_path=save_path,
            entropy=True,
            ppo_type=ppo_type,
            advantage_type=advantage_type,
            param_plot_num=param_plot_num,
            policy_burn_in=policy_burn_in,
        )
        if load_path is not None:
            ppo.policy.load_state_dict(torch.load(load_path))
            ppo.policy.eval()

        # logging variables
        running_rewards = []
        avg_length = 0
        timestep = 0
        action_chooser = ActionChooser(*chooser_params)

        # training loop
        total_steps = 0
        for ep_num in range(1, max_episodes + 1):
            state = env.reset()
            ep_total_reward = 0
            t = 0
            action_list = []
            action_chooser.reset()
            keep_running = True if max_timesteps is None else t < max_timesteps
            while keep_running:
                timestep += 1
                t += 1
                keep_running = True if max_timesteps is None else t < max_timesteps

                # Running policy_old:
                action = ppo.policy_old.act(state, buffer)
                action = action_chooser.step(action)
                action_list.append(action)
                state, reward, done, _ = env.step(action)

                # Saving reward and is_terminal:
                buffer.rewards.append(reward)
                buffer.is_terminal.append(done)

                # update if its time
                if timestep % update_timestep == 0:
                    ppo.update(buffer, ep_num)
                    buffer.clear()
                    timestep = 0

                ep_total_reward += reward
                if render:
                    env.render()
                if done:
                    break

            avg_length += t / log_interval
            total_steps += t
            running_rewards.append(ep_total_reward)

            ep_str = ("{0:0" + f"{len(str(max_episodes))}" + "d}").format(ep_num)
            if verbose:
                print(
                    f"Episode {ep_str} of {max_episodes}. \t Total reward = {ep_total_reward}"
                )
                print(f"action list: {action_list}")

            # logging
            if ep_num % log_interval == 0:
                running_reward = np.mean(running_rewards[-log_interval:])
                print(
                    f"Episode {ep_str} of {max_episodes}. \t Avg length: {int(avg_length)} \t Reward: {running_reward}"
                )
                plot_episode_reward(save_path, running_rewards)
                ppo.save()

                # stop training if avg_reward > solved_reward
                if running_reward > solved_reward:
                    print("########## Solved! ##########")
                    break
                avg_length = 0
        episode_numbers.append((ep_num, total_steps))
        torch.save(ppo.policy.state_dict(), f"{save_path}/PPO_actor_critic.pth")
    print(f"episode_numbers: {episode_numbers}")
    return episode_numbers


def main():
    #### ATARI ####
    log_interval = 20  # print avg reward in the interval
    max_episodes = 100000  # max training episodes
    max_timesteps = 10000  # max timesteps in one episode
    update_timestep = 1024
    random_seeds = list(range(0, 5))
    ppo_types = ["clip"]

    #### OTHER STUFF ####
    env_names = ["Acrobot-v1", "CartPole-v1", "MountainCar-v0"]
    solved_rewards = [-80, 195, -135]  # stop training if avg_reward > solved_reward
    actor_critic_params = ActorCriticParams(
        actor_layers=(32, 32),
        actor_activation="tanh",
        critic_layers=(32, 32),
        critic_activation="tanh",
        num_shared_layers=1,
    )
    adv_types = ["gae"]
    chooser_params = (100, 1, 100)

    date = datetime.date.today().strftime("%d-%m-%Y")

    outcomes = []

    try:
        for env_name, solved_reward in zip(env_names, solved_rewards):
            outcomes.append(env_name)
            for adv_type in adv_types:
                outcomes.append(adv_type)
                # for param_sharing in param_sharings:
                hyp = HyperparametersPPO(
                    gamma=0.99,  # discount factor
                    lamda=0.95,  # GAE weighting factor
                    learning_rate=3.5e-3,
                    T=1024,  # update policy every n timesteps
                    epsilon=0.2,  # clip parameter for PPO
                    c1=0.5,  # value function hyperparam
                    c2=0.01,  # entropy hyperparam
                    num_epochs=3,  # update policy for K epochs
                    # d_targ=d_targ,          # adaptive KL param
                    # beta=beta,              # fixed KL param
                )

                outcomes.append(
                    train_ppo(
                        env_name=env_name,
                        solved_reward=solved_reward,
                        hyp=hyp,
                        actor_critic_params=actor_critic_params,
                        random_seeds=random_seeds,
                        update_timestep=update_timestep,
                        log_interval=log_interval,
                        max_episodes=max_episodes,
                        max_timesteps=max_timesteps,
                        ppo_type=ppo_types[0],
                        advantage_type=adv_type,
                        date=date,
                        # render=True,
                        param_plot_num=10,
                        # policy_burn_in=5,
                        # chooser_params=chooser_params,
                    )
                )
    finally:
        print(f"outcomes:")
        for outcome in outcomes:
            print(outcome)


if __name__ == "__main__":
    main()
