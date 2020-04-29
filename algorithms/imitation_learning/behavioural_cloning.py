import os
import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
import gym
from shutil import rmtree
from typing import Tuple, List, Optional

from algorithms.discrete_policy import DiscretePolicy, DiscretePolicyParams
from algorithms.buffer import DemonstrationBuffer
from algorithms.plotter import Plotter
from algorithms.utils import generate_save_location

from envs.atari.checking import get_average_score


device = "cuda" if torch.cuda.is_available() else "cpu"


class BCTrainer:
    def __init__(
        self,
        demo_path: Path,
        save_path: Path,
        learning_rate: float,
        state_space_size: Tuple,
        action_space_size: int,
        discrete_policy_params: DiscretePolicyParams,
        param_plot_num: int,
        max_plot_size: int = 10000,
    ):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.policy = DiscretePolicy(
            state_dimension=state_space_size,
            action_space=action_space_size,
            params=discrete_policy_params,
        ).float()

        self.demo_buffer = DemonstrationBuffer(
            demo_path, state_space_size, action_space_size
        )
        self.epochs_trained = 0

        plots = [
            ("minibatch_loss", np.float32),
            ("epoch_loss", np.float32),
            ("avg_score", np.float64),
        ]
        counts = [("num_steps", int)]
        self.plotter = Plotter(
            network_params=discrete_policy_params,
            save_path=save_path,
            counts=counts,
            plots=plots,
            max_plot_size=max_plot_size,
            param_plot_num=param_plot_num,
            state_dim=state_space_size,
            action_space=action_space_size,
        )
        self.save_path = save_path

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

        # zero the parameter gradients
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()

    def _load_demos(self, demo_list):
        self.demo_buffer.from_numpy()
        np.random.shuffle(demo_list)
        for demo_num in demo_list:
            self.demo_buffer.load_demo(demo_num)
        self.demo_buffer.to_numpy()

    def train_network(
        self,
        num_epochs: int,
        demo_list: List,
        max_minibatch_size: int,
        steps_per_epoch: int,
    ):
        """
        Trains network for `num_epochs` epochs.

        Args:
            num_epochs: Number of epochs to run.
            demo_list: List of demo numbers which can be used.
            max_minibatch_size: Maximum size of minibatch.
            steps_per_epoch: Number of steps of experience per epoch.
        """
        # avg_loss_plot = []
        for epoch in range(1, num_epochs + 1):
            sum_loss, num_steps = 0, 0

            while num_steps < steps_per_epoch:
                while self.demo_buffer.get_length() < max_minibatch_size:
                    self._load_demos(demo_list)

                minibatch_size = min(steps_per_epoch - num_steps, max_minibatch_size)

                sampled_states, sampled_actions, _ = self.demo_buffer.random_sample(
                    minibatch_size
                )
                states = torch.from_numpy(sampled_states).to(device)
                actions = torch.from_numpy(sampled_actions).to(device)
                # forward + backward + optimize
                action_probs = self.policy(states.float())
                action_probs = action_probs.float()
                actions = actions.type(torch.long)

                loss = self.loss_fn(action_probs, actions)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.plotter.record_data(
                    {"minibatch_loss": loss.cpu().detach().numpy()}
                )
                num_steps += minibatch_size
                sum_loss += loss.cpu().detach().numpy()
                self._record_nn_params()

            avg_loss = np.round(
                sum_loss / np.ceil(steps_per_epoch / max_minibatch_size), 5
            )
            self.plotter.record_data({"epoch_loss": avg_loss})
            self.plotter.save_plots()
            self.epochs_trained += 1
            print(f"Epoch number: {self.epochs_trained}\tAvg loss: {avg_loss}")
            # avg_loss_plot.append(avg_loss)
            # if len(avg_loss_plot) > 10:
            #     if (
            #         abs(avg_loss - avg_loss_plot[0]) < 1e-9
            #         and abs(avg_loss - avg_loss_plot[-3]) < 1e-9
            #         and abs(avg_loss - avg_loss_plot[-1]) < 1e-9
            #     ):
            #         print(
            #             f"\n\nAvg loss hasn't changed for 10 iterations.\n Skipping to next seed.\n"
            #         )
            #         break
        self._save_network()

    def _record_nn_params(self):
        """Gets randomly sampled actor NN parameters from 1st layer."""
        names, x_params, y_params = self.plotter.get_param_plot_nums()
        sampled_params = {}
        for name, x_param, y_param in zip(names, x_params, y_params):
            sampled_params[name] = (
                self.policy.state_dict()[name].cpu().numpy()[x_param, y_param]
            )
        self.plotter.record_data(sampled_params)

    def _save_network(self):
        neural_net_save = self.save_path / f"BC-{self.epochs_trained}-epochs.pth"
        torch.save(self.policy.state_dict(), f"{neural_net_save}")


def train_network(
    env_name: str,
    num_demos: int,
    minibatch_size: int,
    epoch_nums: List,
    steps_per_epoch: int,
    demo_path: Path,
    learning_rate: float,
    random_seeds: List,
    discrete_policy_params: DiscretePolicyParams,
    param_plot_num: int,
    num_test_trials: int,
    restart: bool,
    date: Optional[str] = None,
):
    env = gym.make(env_name).env
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    demo_list = np.random.choice(os.listdir(f"{demo_path}"), num_demos, replace=False)

    for random_seed in random_seeds:

        if random_seed is not None:
            torch.manual_seed(random_seed)
            env.seed(random_seed)
            print(f"Set random seed to: {random_seed}")

        hyp_str = f"demos-{num_demos}"
        save_location = generate_save_location(
            Path("data"),
            discrete_policy_params.actor_layers,
            f"BC",
            env_name,
            random_seed,
            hyp_str,
            date,
        )
        if restart:
            if save_location.exists():
                print("Old data removed!")
                rmtree(save_location)

        trainer = BCTrainer(
            demo_path=demo_path,
            save_path=save_location,
            learning_rate=learning_rate,
            action_space_size=action_dim,
            state_space_size=state_dim,
            discrete_policy_params=discrete_policy_params,
            param_plot_num=param_plot_num,
        )

        prev_num_epochs = 0
        for count, epoch_num in enumerate(epoch_nums):
            if (count + 1) % 2 == 0 or env_name != "MountainCar-v0":
                num_epochs_to_train = epoch_num - prev_num_epochs
                trainer.train_network(
                    num_epochs=num_epochs_to_train,
                    demo_list=demo_list,
                    max_minibatch_size=minibatch_size,
                    steps_per_epoch=steps_per_epoch,
                )
                mean_score = get_average_score(
                    network_load=save_location / f"BC-{epoch_num}-epochs.pth",
                    env=env,
                    episode_timeout=1998,
                    show_solution=False,
                    num_trials=num_test_trials,
                    params=discrete_policy_params,
                )
                trainer.plotter.record_data({"avg_score": mean_score})
                prev_num_epochs += num_epochs_to_train
