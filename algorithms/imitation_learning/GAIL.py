import numpy as np
import torch
import gym
from pathlib import Path
from shutil import rmtree
from collections import namedtuple
from typing import Tuple, List, Optional

from buffer import DemonstrationBuffer, GAILExperienceBuffer
from discriminator import Discriminator, DiscrimParams
from discrete_policy import DiscretePolicyParams
from PPO import PPO
from utils import generate_save_location, generate_gail_str

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyp_names = (
    "gamma",
    "batch_size",
    "num_epochs",
    "num_discrim_epochs",
    "learning_rate",
    "discrim_lr",
    "entropy",
    "num_demos",
    "success_margin",
    "fraction_expert",
    "c1",
    "c2",
    "lamda",
    "epsilon",
    "beta",
    "d_targ",
)


try:
    HyperparametersGAIL = namedtuple(
        "HyperparametersGAIL", hyp_names, defaults=(None,) * len(hyp_names),
    )
except TypeError:
    HyperparametersGAIL = namedtuple("HyperparametersGAIL", hyp_names)
    HyperparametersGAIL.__new__.__defaults__ = (None,) * len(hyp_names)
"""
    gamma: Discount factor for time delay in return.
    batch_size: Number of state-action pairs in update.
    num_epochs: Number of epochs of learning carried out on each T timesteps.
    num_discrim_epochs: Number of epochs of learning carried out by the discriminator. 
    learning_rate: Learning rate of Adam optimizer on Actor and Critic.
    discrim_lr: Discriminator learning rate for Adam optimizer.
    entropy: Bool determining whether or not to use entropy.
    num_demos: Number of demonstrations used.
    success_margin: Percentage of avg expert return within which success is achieved. 
    fraction_expert: Fraction of the discriminator update data that is from the expert.
    c1: (Optional) Value function loss weighting factor.
    c2: (Optional) Entropy bonus loss term weighting factor.
    lamda: (Optional) GAE weighting factor.
    epsilon: (Optional) PPO clipping parameter.
    beta: (Optional) KL penalty parameter.
    d_targ: (Optional) Adaptive KL target.
"""


def pick_demo(possible_demos: List) -> int:
    chosen_ref = np.random.randint(0, len(possible_demos))
    return possible_demos[chosen_ref]


def sample_from_buffers(
    demo_buffer: DemonstrationBuffer,
    gail_buffer: GAILExperienceBuffer,
    fraction_expert: int,
    action_space: int,
    possible_demos: List,
):
    num_learner_samples = gail_buffer.get_length()
    learner_states = torch.stack(gail_buffer.states).to(device).detach()
    learner_actions = torch.stack(gail_buffer.actions).to(device)
    learner_action_vectors = torch.zeros((learner_actions.shape[0], action_space))
    learner_action_vectors[
        torch.arange(learner_actions.shape[0]), learner_actions
    ] = 256
    learner_tuples = torch.cat((learner_states, learner_action_vectors), dim=1)

    num_expert_samples = int(
        num_learner_samples * (1 - fraction_expert) / fraction_expert
    )
    # Sort the is_terminals
    prev_demo_length = 0
    while demo_buffer.get_length() < num_expert_samples:
        gail_buffer.is_terminal += [0] * (
            demo_buffer.get_length() - prev_demo_length - 1
        ) + [1]
        demo_buffer.load_demo(pick_demo(possible_demos))
        prev_demo_length = demo_buffer.get_length()
    gail_buffer.is_terminal += [0] * (
        num_learner_samples + num_expert_samples - len(gail_buffer.is_terminal)
    )

    expert_states, expert_actions = demo_buffer.recall_expert_data(num_expert_samples)
    expert_states = torch.from_numpy(expert_states).type(dtype=torch.float32).to(device)
    expert_actions = torch.from_numpy(expert_actions).type(dtype=torch.int64).to(device)
    expert_action_vectors = torch.zeros((expert_actions.shape[0], action_space))
    expert_action_vectors[torch.arange(expert_actions.shape[0]), expert_actions] = 256
    expert_tuples = torch.cat((expert_states, expert_action_vectors), dim=1)

    data_set = torch.cat((learner_tuples, expert_tuples), dim=0)
    gail_buffer.state_actions = data_set
    labels = torch.cat(
        (torch.zeros(num_learner_samples), torch.ones(num_expert_samples))
    )
    gail_buffer.discrim_labels = labels

    return gail_buffer


class GAILTrainer(PPO):
    def __init__(
        self,
        state_dimension: Tuple,
        action_space: int,
        save_path: Path,
        hyp: HyperparametersGAIL,
        policy_params: DiscretePolicyParams,
        discriminator_params: DiscrimParams,
        param_plot_num: int,
        ppo_type: str = "clip",
        adv_type: str = "monte_carlo",
        max_plot_size: int = 10000,
        policy_burn_in: int = 0,
        verbose: bool = False,
    ):
        self.discrim_net_save = save_path / "GAIL_discrim.pth"

        self.discriminator = Discriminator(
            state_dimension, action_space, discriminator_params,
        ).to(device)

        self.discrim_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=hyp.discrim_lr
        )
        gail_plots = [("discrim_loss", np.float64)]

        super(GAILTrainer, self).__init__(
            state_dimension,
            action_space,
            save_path,
            hyp,
            policy_params,
            param_plot_num,
            ppo_type,
            advantage_type=adv_type,
            neural_net_save=f"GAIL-{adv_type}",
            max_plot_size=max_plot_size,
            discrim_params=discriminator_params,
            policy_burn_in=policy_burn_in,
            verbose=verbose,
            additional_plots=gail_plots,
        )
        self.num_learner_samples = None

        self.KL_loss = torch.nn.KLDivLoss()

    def update(self, buffer: GAILExperienceBuffer, ep_num: int):
        # Update discriminator
        state_actions = buffer.state_actions.to(device)
        if self.num_learner_samples is None:
            self.num_learner_samples = buffer.get_length()
        for epoch in range(self.hyp.num_discrim_epochs):
            discrim_probs = self.discriminator(state_actions).to(device)
            # learner_loss = torch.log(torch.clamp(discrim_probs[:self.num_learner_samples], 0.01, 1)).mean()
            # print(buffer.discrim_labels[:self.num_learner_samples])
            # print(buffer.discrim_labels[self.num_learner_samples:])
            discrim_vector = torch.cat(
                (
                    torch.unsqueeze(1 - discrim_probs, dim=1),
                    torch.unsqueeze(discrim_probs, dim=1),
                ),
                dim=-1,
            )
            # print(f"discrim labels: {buffer.discrim_labels}")
            labels = torch.eye(2)[buffer.discrim_labels.type(torch.long)]
            learner_loss = self.KL_loss(
                input=discrim_vector[: self.num_learner_samples],
                target=labels[: self.num_learner_samples],
            )
            expert_loss = self.KL_loss(
                input=discrim_vector[self.num_learner_samples :],
                target=labels[self.num_learner_samples :],
            )
            # expert_loss = torch.log(torch.clamp(1 - discrim_probs[self.num_learner_samples:], 0.01, 1)).mean()
            loss = expert_loss + learner_loss
            print(f"Learner loss: \t{learner_loss}")
            print(f"Expert loss: \t{expert_loss}")

            # loss = self.KL_loss(discrim_probs, buffer.discrim_labels)
            print(
                f"Learner labels 0: \t{discrim_probs[:self.num_learner_samples].mean()}"
            )
            print(
                f"Expert labels 1: \t{discrim_probs[self.num_learner_samples:].mean()}"
            )
            # print(f"Learner data -\t label: {0}\t discrim prob: {discrim_probs[:1024].mean()}")
            # print(f"Learner data -\t label: {0}\t discrim prob: {discrim_probs[:1024].size()}")
            # print(f"Learner data -\t label: {0}\t discrim prob: {discrim_probs[:1024]}")
            # print(f"Expert data -\t label: {1}\t discrim prob: {discrim_probs[1024:].mean()}")
            # print(f"Expert data -\t label: {1}\t discrim prob: {discrim_probs[1024:].size()}")
            # print(f"Expert data -\t label: {1}\t discrim prob: {discrim_probs[1024:]}")
            plotted_loss = loss.detach().cpu().numpy()
            self.plotter.record_data({"discrim_loss": plotted_loss})
            # print(f"Plotted loss = {plotted_loss}")
            self.discrim_optim.zero_grad()
            loss.mean().backward()
            self.discrim_optim.step()

        # Update policy
        num_learner_steps = buffer.get_length()
        buffer.rewards = list(
            np.squeeze(
                torch.log(
                    self.discriminator(state_actions[:num_learner_steps])
                )  # Take log of discrim
                .detach()
                .cpu()
                .numpy()
            )
        )
        # print(f"Expert data -\t label: {1}\t discrim prob: {np.mean(buffer.rewards)}")
        print(
            "------------------------------------------------------------------------"
        )
        super(GAILTrainer, self).update(buffer, ep_num)

    def record_nn_params(self):
        """Gets randomly sampled actor NN parameters from 1st layer."""
        names, x_params, y_params = self.plotter.get_param_plot_nums()
        sampled_params = {}

        for name, x_param, y_param in zip(names, x_params, y_params):
            network_to_sample = (
                self.discriminator if name[:7] == "discrim" else self.policy
            )
            sampled_params[name] = (
                network_to_sample.state_dict()[name].cpu().numpy()[x_param, y_param]
            )
        self.plotter.record_data(sampled_params)

    def _save_network(self):
        super(GAILTrainer, self)._save_network()
        torch.save(self.discriminator.state_dict(), f"{self.discrim_net_save}")

    def _load_network(self):
        super(GAILTrainer, self)._load_network()
        print(f"Loading discriminator network saved at: {self.discrim_net_save}")
        net = torch.load(self.discrim_net_save, map_location=device)
        self.discriminator.load_state_dict(net)


def train_gail(
    demo_path: Path,
    env_name: str,
    policy_params: DiscretePolicyParams,
    discrim_params: DiscrimParams,
    max_episodes: int,
    log_interval: int,
    hyp: HyperparametersGAIL,
    random_seeds: List,
    max_timesteps: Optional[int] = None,
    render: bool = False,
    verbose: bool = False,
    ppo_type: str = "clip",
    adv_type: str = "monte_carlo",
    date: Optional[str] = None,
    param_plot_num: int = 10,
    policy_burn_in: int = 0,
    restart: bool = False,
):
    try:
        env = gym.make(env_name).env
        state_dim = env.observation_space.shape
        action_dim = env.action_space.n

        returns = []
        for expert_trajectory_path in demo_path.iterdir():
            rewards = np.load(
                f"{expert_trajectory_path}/rewards.npy", allow_pickle=True
            )
            returns.append(np.sum(rewards))
        exp_returns = np.mean(returns)
        success_frac = hyp.success_margin / 100  # From % to fraction
        solved_reward = (
            (1 + success_frac) * exp_returns
            if exp_returns < 0
            else (1 - success_frac) * exp_returns
        )

        episode_numbers = []
        for random_seed in random_seeds:
            if random_seed is not None:
                torch.manual_seed(random_seed)
                env.seed(random_seed)
                print(f"Set random seed to: {random_seed}")

            buffer = GAILExperienceBuffer(state_dim, action_dim)
            demo_buffer = DemonstrationBuffer(demo_path, state_dim, action_dim)

            hyp_str = generate_gail_str(ppo_type, hyp)
            save_path = generate_save_location(
                Path("data"),
                policy_params.actor_layers,
                f"GAIL-{ppo_type}",
                env_name,
                random_seed,
                hyp_str,
                date,
            )
            if restart:
                if save_path.exists():
                    print("Old data removed!")
                    rmtree(save_path)

            gail = GAILTrainer(
                state_dimension=state_dim,
                action_space=action_dim,
                hyp=hyp,
                save_path=save_path,
                ppo_type=ppo_type,
                adv_type=adv_type,
                policy_params=policy_params,
                discriminator_params=discrim_params,
                param_plot_num=param_plot_num,
                policy_burn_in=policy_burn_in,
                verbose=verbose,
            )

            # logging variables
            running_reward = 0
            avg_length = 0
            timestep = 0
            possible_demos = gail.plotter.determine_demo_nums(demo_path, hyp.num_demos)
            ep_num_start = gail.plotter.get_count("episode_num")

            # training loop
            print(f"Starting running from episode number {ep_num_start + 1}\n")
            for ep_num in range(ep_num_start + 1, max_episodes + 1):
                state = env.reset()
                ep_total_reward = 0
                for t in range(max_timesteps):
                    timestep += 1

                    # Running policy_old:
                    action = gail.policy_old.act(state, buffer)
                    state, reward, done, _ = env.step(action)

                    # Saving reward and is_terminal:
                    buffer.rewards.append(reward)
                    buffer.is_terminal.append(done)

                    # Update if its time
                    if timestep % hyp.batch_size == 0:
                        buffer = sample_from_buffers(
                            demo_buffer,
                            buffer,
                            hyp.fraction_expert,
                            action_dim,
                            possible_demos,
                        )

                        gail.update(buffer, ep_num)
                        buffer.clear()
                        timestep = 0

                    ep_total_reward += reward
                    if render:
                        env.render()
                    if done:
                        break

                avg_length += t / log_interval
                running_reward += ep_total_reward / log_interval
                gail.plotter.record_data(
                    {"rewards": ep_total_reward, "num_steps_taken": t, "episode_num": 1}
                )

                ep_str = ("{0:0" + f"{len(str(max_episodes))}" + "d}").format(ep_num)
                if verbose:
                    print(
                        f"Episode {ep_str} of {max_episodes}. \t Total reward = {ep_total_reward}"
                    )

                # logging
                if ep_num % log_interval == 0:
                    print(
                        f"Episode {ep_str} of {max_episodes}. \t Avg length: {int(avg_length)} \t Reward: {np.round(running_reward, 1)}"
                    )
                    gail.save()

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
        gail.save()
        raise interrupt


# num_samples = 4
# n = np.append(np.zeros(num_samples), np.ones(num_samples))
# t1 = torch.tensor(n, dtype=torch.long)
# print(t1)
# y = torch.eye(num_samples)
# print(y)
# print(y[t1])

# t2 = 1 - t1
# print(torch.unsqueeze(t1, dim=1))
# print(t2)
# t3 = torch.cat((torch.unsqueeze(t1, dim=1), torch.unsqueeze(t2, dim=1)), dim=1)
# print(t3)


# prob_correct = 0.9
# t2 = torch.tensor([1 - prob_correct] * num_samples + [prob_correct] * num_samples)
# t3 = torch.tensor([prob_correct] * num_samples * 2)
# # print(t1)
# # print(t2)
# def print_mean(t1, t2):
#     print(t1 - t2)
#     # loss = torch.log(torch.abs(t1 - t2))
#     learner_loss = torch.log(t2[:num_samples])
#     expert_loss = torch.log(1 - t2[num_samples:])
#     loss = learner_loss.mean() + expert_loss.mean()
#     print(learner_loss)
#     print(expert_loss)
#     print(loss)
# print_mean(t1, t2)
# print_mean(t1, t3)
