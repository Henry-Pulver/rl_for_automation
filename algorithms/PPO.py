import numpy as np
import torch
import torch.nn as nn
import datetime

from torch.functional import F
import gym
from collections import namedtuple

from typing import Tuple, List, Optional
from pathlib import Path

from action_chooser import ActionChooser
from actor_critic import ActorCritic, ActorCriticParams
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
    c1: Value function loss weighting factor.
    c2: (Optional) Entropy bonus loss term weighting factor.
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
        actor_critic_params: ActorCriticParams,
        param_plot_num: int,
        ppo_type: str = "clip",
        advantage_type: str = "monte_carlo_baseline",
        policy_burn_in: int = 0,
        neural_net_save: str = "PPO_actor_critic.pth",
        max_plot_size: int = 10000,
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

        plots = [
            ("mean_entropy_loss", np.float64),
            ("mean_clipped_loss", np.ndarray),
            ("mean_value_loss", np.ndarray),
            ("rewards", float),
        ]
        counts = [("num_steps_taken", int), ("episode_num", int)]
        self.plotter = Plotter(
            actor_critic_params,
            save_path,
            plots,
            counts,
            max_plot_size,
            param_plot_num,
            state_dimension,
        )

        self.neural_net_save = save_path / neural_net_save

        self.policy = ActorCritic(
            state_dimension, action_space, actor_critic_params,
        ).to(device)
        if self.neural_net_save.exists():
            self._load_network()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr,)
        self.policy_old = ActorCritic(
            state_dimension, action_space, actor_critic_params,
        ).to(device)
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
        else:
            raise ValueError("Invalid PPO type used!")
        if update:
            loss += main_loss
        value_loss = self.hyp.c1 * self.MseLoss(state_values, norm_returns)
        loss += value_loss

        loss -= self.hyp.c2 * dist_entropy.mean()

        self.plotter.record_data(
            {
                "mean_entropy_loss": -0.01 * dist_entropy.mean().detach().cpu().numpy(),
                "mean_clipped_loss": main_loss.mean().detach().cpu().numpy(),
                "mean_value_loss": value_loss.detach().cpu().numpy(),
            }
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
            sampled_params[name] = self.policy.state_dict()[name].cpu().numpy()[x_param, y_param]
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
):
    try:
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

            ppo = PPO(
                state_dimension=state_dim,
                action_space=action_dim,
                actor_critic_params=actor_critic_params,
                hyperparameters=hyp,
                save_path=save_path,
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
            timestep = 0  # Determines when to update the network
            action_chooser = ActionChooser(*chooser_params)

            # training loop
            total_steps = 0
            for ep_num in range(1, max_episodes + 1):  # Run episodes
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
                total_steps += t
                running_rewards.append(ep_total_reward)
                ppo.plotter.record_data({"rewards": ep_total_reward, "num_steps_taken": t, "episode_num": 1})

                ep_str = ("{0:0" + f"{len(str(max_episodes))}" + "d}").format(ep_num)
                if verbose:
                    print(
                        f"Episode {ep_str} of {max_episodes}. \t Total reward = {ep_total_reward}"
                    )

                if ep_num % log_interval == 0:
                    running_reward = np.mean(running_rewards[-log_interval:])
                    print(
                        f"Episode {ep_str} of {max_episodes}. \t Avg length: {int(avg_length)} \t Reward: {running_reward}"
                    )
                    ppo.save()

                    # stop training if avg_reward > solved_reward
                    if running_reward > solved_reward:
                        print("########## Solved! ##########")
                        break
                    avg_length = 0
            episode_numbers.append((ep_num, total_steps))
        print(f"episode_numbers: {episode_numbers}")
        return episode_numbers
    except KeyboardInterrupt as interrupt:
        ppo.save()
        raise interrupt


def main():
    print("\n\nTHIS MUST BE RUN IN TERMINAL OTHERWISE IT WON'T SAVE!\n\n")
    #### ATARI ####
    log_interval = 20  # print avg reward in the interval
    max_episodes = 100000  # max training episodes
    max_timesteps = 10000  # max timesteps in one episode
    random_seeds = list(range(0, 5))
    ppo_types = ["clip"]
    adv_types = ["gae"]
    chooser_params = (100, 1, 100)
    actor_critic_params = ActorCriticParams(
        actor_layers=(32, 32),
        actor_activation="tanh",
        critic_layers=(32, 32),
        critic_activation="tanh",
        num_shared_layers=1,
    )
    hyp = HyperparametersPPO(
        gamma=0.99,  # discount factor
        lamda=0.95,  # GAE weighting factor
        learning_rate=2e-3,
        T=1024,  # update policy every n timesteps
        epsilon=0.2,  # clip parameter for PPO
        c1=0.5,  # value function hyperparam
        c2=0.01,  # entropy hyperparam
        num_epochs=3,  # update policy for K epochs
        # d_targ=d_targ,          # adaptive KL param
        # beta=beta,              # fixed KL param
    )

    #### OTHER STUFF ####
    env_names = ["Acrobot-v1", "CartPole-v1", "MountainCar-v0"]
    solved_rewards = [-80, 195, -135]  # stop training if avg_reward > solved_reward

    date = datetime.date.today().strftime("%d-%m-%Y")

    outcomes = []
    try:
        for env_name, solved_reward in zip(env_names, solved_rewards):
            outcomes.append(env_name)
            for adv_type in adv_types:
                outcomes.append(adv_type)
                outcomes.append(
                    train_ppo(
                        env_name=env_name,
                        solved_reward=solved_reward,
                        hyp=hyp,
                        actor_critic_params=actor_critic_params,
                        random_seeds=random_seeds,
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
