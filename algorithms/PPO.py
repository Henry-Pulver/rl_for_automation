import numpy as np
import gym
import torch
import torch.nn as nn
from torch.functional import F
from shutil import rmtree
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
from typing import Tuple, List, Optional

from action_chooser import ActionChooser
from actor_critic import ActorCritic, ActorCriticParams
from discrete_policy import DiscretePolicy
from discriminator import DiscrimParams
from advantage_estimation import get_td_error, get_gae
from buffer import PPOExperienceBuffer
from plotter import Plotter
from utils import generate_save_location, generate_ppo_hyp_str

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
    c1: (Optional) Value function loss weighting factor.
    c2: Entropy bonus loss term weighting factor. Set to 0 for no entropy component.
    lamda: (Optional) GAE weighting factor.
    epsilon: (Optional) PPO clipping parameter.
    beta: (Optional) KL penalty parameter.
    d_targ: (Optional) Adaptive KL target.
"""


class PPO:
    PPO_TYPES = ["clip", "adaptive_KL", "fixed_KL", "unclipped"]
    ADVANTAGE_TYPES = ["monte_carlo", "monte_carlo_baseline", "gae"]

    def __init__(
        self,
        state_dimension: Tuple,
        action_space: int,
        save_path: Path,
        hyperparameters: HyperparametersPPO,
        policy_params: namedtuple,
        param_plot_num: int,
        ppo_type: str = "clip",
        advantage_type: str = "monte_carlo_baseline",
        policy_burn_in: int = 0,
        neural_net_save: str = "PPO_actor_critic",
        max_plot_size: int = 10000,
        discrim_params: Optional[DiscrimParams] = None,
        verbose: bool = False,
        additional_plots: Optional[List] = None,
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
        self.verbose = verbose

        self.policy_burn_in = policy_burn_in
        self.using_value = type(policy_params) == ActorCriticParams

        additional_plots = [] if additional_plots is None else additional_plots
        plots = additional_plots + [
            ("mean_clipped_loss", np.ndarray),
            ("rewards", float),
        ]
        plots = plots + [("mean_entropy_loss", np.float64)] if self.hyp.c2 != 0 else plots
        plots = plots + [("mean_value_loss", np.ndarray)] if self.using_value else plots
        counts = [("num_steps_taken", int), ("episode_num", int)]
        self.plotter = Plotter(
            policy_params,
            save_path,
            plots,
            counts,
            max_plot_size,
            param_plot_num,
            state_dimension,
            action_space,
            discrim_params,
            verbose,
        )

        self.neural_net_save = save_path / f"{neural_net_save}.pth"

        Policy = ActorCritic if self.using_value else DiscretePolicy

        self.policy = Policy(state_dimension, action_space, policy_params,).to(device)
        if self.neural_net_save.exists():
            self._load_network()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.hyp.learning_rate
        )
        self.policy_old = Policy(state_dimension, action_space, policy_params,).to(
            device
        )
        self.policy_old.load_state_dict(self.policy.state_dict())

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
            discounted_reward = reward + (self.hyp.gamma * discounted_reward)
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

        # Optimize policy for the number of epochs hyperparam:
        for k in range(self.hyp.num_epochs):
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
            self.record_nn_params()
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
        plot_data = {}
        # Evaluating old actions and values :
        logprobs, state_values, dist_entropy, action_probs = self.policy.evaluate(
            states, actions
        )
        if update:
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Loss:
            advantages = torch.squeeze(
                self.calculate_advantages(
                    norm_returns, state_values, buffer, returns_mean, returns_std_dev
                )
            )
            assert ratios.size() == advantages.size()
            surr1 = ratios * advantages
            if self.ppo_type == "clip":
                surr2 = (
                    torch.clamp(ratios, 1 - self.hyp.epsilon, 1 + self.hyp.epsilon)
                    * advantages
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
            else:
                raise ValueError("Invalid PPO type used!")
            plot_data["mean_clipped_loss"] = deepcopy(main_loss.mean().detach().cpu().numpy())

            loss = main_loss
            if self.hyp.c2 != 0:
                entropy_loss = -self.hyp.c2 * dist_entropy.mean()
                loss += entropy_loss
                plot_data["mean_entropy_loss"] = deepcopy(np.squeeze(
                    entropy_loss.mean().detach().cpu().numpy()
                ))
        else:
            loss = torch.tensor([0], requires_grad=True, dtype=torch.float32).clone()

        if self.using_value:
            value_loss = self.hyp.c1 * self.MseLoss(state_values, norm_returns)
            loss += value_loss
            plot_data["mean_value_loss"] = deepcopy(value_loss.detach().cpu().numpy())

        self.plotter.record_data(plot_data)
        return loss

    def calculate_advantages(
        self,
        norm_returns: torch.tensor,
        state_values: Optional[torch.tensor] = None,
        buffer: Optional[PPOExperienceBuffer] = None,
        returns_mean: Optional[torch.Tensor] = None,
        returns_std_dev: Optional[torch.Tensor] = None,
    ):
        if self.adv_type == "monte_carlo":
            advantages = norm_returns
        elif self.adv_type == "monte_carlo_baseline":
            advantages = norm_returns - state_values.detach()
        elif self.adv_type == "gae":
            scaled_state_values = (
                state_values.detach() * returns_std_dev
            ) + returns_mean
            td_errors = get_td_error(
                scaled_state_values.cpu().numpy(), buffer, self.hyp.gamma
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

    def record_nn_params(self):
        """Gets randomly sampled actor NN parameters from 1st layer."""
        names, x_params, y_params = self.plotter.get_param_plot_nums()
        sampled_params = {}
        for name, x_param, y_param in zip(names, x_params, y_params):
            sampled_params[name] = (
                self.policy.state_dict()[name].cpu().numpy()[x_param, y_param]
            )
        self.plotter.record_data(sampled_params)

    def save(self):
        self._save_network()
        self.plotter.save_plots()

    def _save_network(self):
        torch.save(self.policy.state_dict(), f"{self.neural_net_save}")

    def _load_network(self):
        print(f"Loading neural network saved at: {self.neural_net_save}")
        net = torch.load(self.neural_net_save, map_location=device)
        self.policy.load_state_dict(net)


def train_ppo(
    env_name: str,
    max_episodes: int,
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
    restart: bool = False,
    action_space: Optional[List] = None,
):
    try:
        env = gym.make(env_name).env
        state_dim = env.observation_space.shape
        action_dim = env.action_space.n if action_space is None else len(action_space)

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
            if restart:
                if save_path.exists():
                    print("Old data removed!")
                    rmtree(save_path)

            ppo = PPO(
                state_dimension=state_dim,
                action_space=action_dim,
                policy_params=actor_critic_params,
                hyperparameters=hyp,
                save_path=save_path,
                ppo_type=ppo_type,
                advantage_type=advantage_type,
                param_plot_num=param_plot_num,
                policy_burn_in=policy_burn_in,
                verbose=verbose,
            )
            if load_path is not None:
                ppo.policy.load_state_dict(torch.load(load_path))

            # logging variables
            avg_length = 0
            timestep = 0  # Determines when to update the network
            running_reward = 0
            action_chooser = ActionChooser(*chooser_params, action_space)
            ep_num_start = ppo.plotter.get_count("episode_num")

            # training loop
            print(f"Starting running from episode number {ep_num_start + 1}\n")
            for ep_num in range(ep_num_start + 1, max_episodes + 1):  # Run episodes
                state = env.reset()
                ep_total_reward = 0
                t = 0
                action_chooser.reset()
                keep_running = True if max_timesteps is None else t < max_timesteps
                while keep_running:  # Run 1 episode
                    timestep += 1
                    t += 1
                    keep_running = True if max_timesteps is None else t < max_timesteps

                    # Running policy_old:
                    action = ppo.policy_old.act(state, buffer)
                    action = action_chooser.step(action)
                    state, reward, done, _ = env.step(action)

                    # Saving reward and is_terminal:
                    buffer.rewards.append(reward)
                    buffer.is_terminal.append(done)

                    # update if its time
                    if timestep % hyp.T == 0:
                        ppo.update(buffer, ep_num)
                        buffer.clear()
                        timestep = 0

                    ep_total_reward += reward
                    if render:
                        env.render()
                    if done:
                        break

                avg_length += t / log_interval
                running_reward += ep_total_reward / log_interval
                ppo.plotter.record_data(
                    {"rewards": ep_total_reward, "num_steps_taken": t, "episode_num": 1}
                )

                ep_str = ("{0:0" + f"{len(str(max_episodes))}" + "d}").format(ep_num)
                if verbose:
                    print(f"{ep_total_reward},")
                    # print(
                    #     f"Episode {ep_str} of {max_episodes}. \t Total reward = {ep_total_reward}"
                    # )

                if ep_num % log_interval == 0:
                    print(
                        f"Episode {ep_str} of {max_episodes}. \t Avg length: {int(avg_length)} \t Reward: {np.round(running_reward, 1)}"
                    )
                    ppo.save()

                    # stop training if avg_reward > solved_reward
                    if running_reward > solved_reward:
                        print("########## Solved! ##########")
                        break
                    running_reward = 0
                    avg_length = 0
            episode_numbers.append(ep_num)
        print(f"episode_numbers: {episode_numbers}")
        return episode_numbers
    except KeyboardInterrupt as interrupt:
        ppo.save()
        raise interrupt
