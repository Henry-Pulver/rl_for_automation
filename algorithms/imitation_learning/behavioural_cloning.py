import os
import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
import gym
from collections import namedtuple
from shutil import rmtree
from typing import Tuple, List, Optional

from algorithms.actor_critic import ActorCritic, ActorCriticParams
from algorithms.discrete_policy import DiscretePolicy
from algorithms.buffer import DemonstrationBuffer
from algorithms.plotter import Plotter
from algorithms.trainer import Trainer
from algorithms.utils import generate_save_location

from envs.atari.checking import get_average_score


device = "cuda" if torch.cuda.is_available() else "cpu"


class BC:
    def __init__(
        self,
        demo_path: Path,
        save_path: Path,
        learning_rate: float,
        state_space_size: Tuple,
        action_space_size: int,
        params: namedtuple,
        param_plot_num: int,
        max_plot_size: int = 10000,
    ):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        net_type = ActorCritic if type(params) == ActorCriticParams else DiscretePolicy
        self.policy = (
            net_type(
                state_dimension=state_space_size,
                action_space=action_space_size,
                params=params,
            )
            .float()
            .to(device)
        )

        self.demo_buffer = DemonstrationBuffer(
            demo_path, state_space_size, action_space_size
        )
        plots = [
            ("minibatch_loss", np.float32),
            ("epoch_loss", np.float32),
            ("avg_score", np.float64),
            ("std_dev", np.float64),
        ]
        counts = [("num_steps", int)]
        self.plotter = Plotter(
            network_params=params,
            save_path=save_path,
            counts=counts,
            plots=plots,
            max_plot_size=max_plot_size,
            param_plot_num=param_plot_num,
            state_dim=state_space_size,
            action_space=action_space_size,
        )
        self.save_path = save_path

        self.loss_fn = nn.NLLLoss(reduction="mean")

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
        demo_list: List,
        max_minibatch_size: int,
        steps_per_epoch: int,
        epoch_number: int,
    ):
        """
        Trains network for `num_epochs` epochs.

        Args:
            demo_list: List of demo numbers which can be used.
            max_minibatch_size: Maximum size of minibatch.
            steps_per_epoch: Number of steps of experience per epoch.
        """
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
            action_logprobs = self.policy.logprobs(states.float()).float().to(device)
            actions = actions.type(torch.long)

            loss = self.loss_fn(action_logprobs, actions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.plotter.record_data({"minibatch_loss": loss.cpu().detach().numpy()})
            num_steps += minibatch_size
            sum_loss += loss.cpu().detach().numpy()
            self._record_nn_params()

        avg_loss = np.round(sum_loss / np.ceil(steps_per_epoch / max_minibatch_size), 5)
        self.plotter.record_data({"epoch_loss": avg_loss})
        self.plotter.save_plots()
        print(f"Epoch number: {epoch_number}\tAvg loss: {avg_loss}")
        self._save_network(epoch_number)

    def _record_nn_params(self):
        """Gets randomly sampled actor NN parameters from 1st layer."""
        names, x_params, y_params = self.plotter.get_param_plot_nums()
        sampled_params = {}
        for name, x_param, y_param in zip(names, x_params, y_params):
            sampled_params[name] = (
                self.policy.state_dict()[name].cpu().numpy()[x_param, y_param]
            )
        self.plotter.record_data(sampled_params)

    def _save_network(self, epoch_number: int):
        neural_net_save = self.save_path / f"BC-{epoch_number}-epochs.pth"
        torch.save(self.policy.state_dict(), f"{neural_net_save}")

    def network_exists(self, epoch_number: int):
        return (self.save_path / f"BC-{epoch_number}-epochs.pth").exists()

    def load_network(self, epoch_number: int):
        load_path = f"{self.save_path}/BC-{epoch_number}-epochs.pth"
        print(f"Loading network number: {epoch_number}\tfrom location: {load_path}")
        self.policy.load_state_dict(torch.load(load_path))


class BCTrainer(Trainer):
    def train(
        self,
        num_demos: int,
        minibatch_size: int,
        epoch_nums: int,
        steps_per_epoch: int,
        demo_path: Path,
        learning_rate: float,
        random_seeds: List,
        params: namedtuple,
        param_plot_num: int,
        num_test_trials: int,
        restart: bool,
        test_demo_timeout: Optional[int] = None,
        chooser_params: Tuple = (None, None, None),
    ):
        random_seeds = random_seeds if random_seeds is not None else [0]
        for random_seed in random_seeds:
            self.set_seed(random_seed)

            save_location = generate_save_location(
                self.save_base_path,
                params.actor_layers,
                f"BC",
                self.env_name,
                random_seed,
                f"demos-{num_demos}",
                self.date,
            )
            self.restart(save_location, restart)

            bc = BC(
                demo_path=demo_path,
                save_path=save_location,
                learning_rate=learning_rate,
                action_space_size=self.action_dim,
                state_space_size=self.state_dim,
                params=params,
                param_plot_num=param_plot_num,
            )

            demo_list = list(bc.plotter.determine_demo_nums(demo_path, num_demos))

            for epoch_num in range(1, epoch_nums + 1):
                if not bc.network_exists(epoch_num):
                    bc.train_network(
                        epoch_number=epoch_num,
                        demo_list=demo_list,
                        max_minibatch_size=minibatch_size,
                        steps_per_epoch=steps_per_epoch,
                    )
                    mean_score, std_dev = get_average_score(
                        network_load=save_location / f"BC-{epoch_num}-epochs.pth",
                        env=self.env,
                        episode_timeout=test_demo_timeout,
                        num_trials=num_test_trials,
                        params=params,
                        chooser_params=chooser_params,
                    )
                    bc.plotter.record_data(
                        {"avg_score": mean_score, "std_dev": std_dev}
                    )
                else:
                    bc.load_network(epoch_num)
