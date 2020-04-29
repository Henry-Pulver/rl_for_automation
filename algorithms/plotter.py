import os
import numpy as np
from collections import namedtuple
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from actor_critic import ActorCriticParams
from discriminator import DiscrimParams


class Count:
    def __init__(self, name: str, datatype: type, save_path: Path):
        self.name = name
        self.datatype = datatype
        self.count_save_path = save_path / f"{name}.npy"
        if self.count_save_path.exists():
            print(f"Loading count: {name}")
            try:
                self.value = np.load(f"{self.count_save_path}")
            except ValueError:
                print(self.count_save_path)
                self.value = np.load(f"{self.count_save_path}", allow_pickle=True)
        else:
            self.value = 0  # Count value

    def record_count(self, counted_data):
        """
        Adds `counted_data` to current count and saves the current count value.

        Args:
            counted_data: Data to be added to the current count.
        """
        assert type(counted_data) == self.datatype
        self.value += counted_data

    def save_count(self, verbose: bool):
        if verbose:
            print(f"Saving count: {self.name} at {self.count_save_path}")
        np.save(f"{self.count_save_path}", self.value)


class Plot:
    def __init__(self, name: str, datatype: type, save_path: Path, max_plot_size: int):
        self.name = name
        self.datatype = datatype
        self.plot_save_path = save_path / self.name
        self.max_plot_size = max_plot_size
        if (self.plot_save_path / self._get_filename(1)).exists():
            print(f"Loading plot: {name}")
            self.file_num = self._find_newest_plot()
            self.plot = list(
                np.load(
                    f"{self.plot_save_path}/{self._get_filename(self.file_num)}",
                    allow_pickle=True,
                )
            )
            if len(self.plot) == self.max_plot_size:
                del self.plot[:]
                self.file_num += 1
        else:
            self.plot_save_path.mkdir(parents=True, exist_ok=True)
            self.plot = []
            self.file_num = 1

    def _get_filename(self, number: int):
        return f"{self.name}_{number}.npy"

    def _find_newest_plot(self):
        newest_plot_number = 1
        while (self.plot_save_path / self._get_filename(newest_plot_number)).exists():
            newest_plot_number += 1
        return newest_plot_number - 1

    def record_plot_data(self, data_entry, verbose: bool):
        if not type(data_entry) == self.datatype:
            data_entry = data_entry.astype(self.datatype)
            if type(data_entry) == np.ndarray:
                assert data_entry.dtype == self.datatype
            else:
                # print(self.name)
                # print(self.datatype)
                # print(type(data_entry))
                # print(data_entry)
                assert type(data_entry) == self.datatype
        self.plot.append(data_entry)
        if len(self.plot) >= self.max_plot_size:
            self.save_plot(verbose)
            del self.plot[:]
            self.file_num += 1
            return self.name
        else:
            return None

    def save_plot(self, verbose: bool):
        if verbose:
            print(
                f"Saving plot: {self.name} at {self.plot_save_path}/{self._get_filename(self.file_num)}"
            )
        np.save(
            f"{self.plot_save_path}/{self._get_filename(self.file_num)}",
            np.array(self.plot),
        )


class Plotter:
    """
    Object that handles Plots and Counts (all data that is saved from training runs
    except the neural network params) and neural network params to plot over time.
    """

    def __init__(
        self,
        network_params: namedtuple,
        save_path: Path,
        plots: List[Tuple],
        counts: List[Tuple],
        max_plot_size: int,
        param_plot_num: int,
        state_dim: Tuple,
        action_space: Optional[int] = None,
        discrim_params: Optional[DiscrimParams] = None,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.using_value = type(network_params) == ActorCriticParams
        self.using_discrim = type(discrim_params) == DiscrimParams
        self.save_path = save_path
        self.param_save = self.save_path / "params"
        self.param_names_array = []
        self.param_x_array = []
        self.param_y_array = []
        if self.param_save.exists():
            self._load_params()
        else:
            self._determine_plotted_params(
                network_params, param_plot_num, state_dim, action_space, discrim_params
            )
            self._save_params()

        plots += [(name, np.ndarray) for name in self.param_names_array]
        self.plots = [
            Plot(name, datatype, save_path, max_plot_size) for name, datatype in plots
        ]
        self.counts = [Count(name, datatype, save_path) for name, datatype in counts]

    def _load_params(self):
        self.param_names_array = np.load(f"{self.param_save}/param_names_array.npy")
        self.param_x_array = np.load(f"{self.param_save}/param_x_array.npy")
        self.param_y_array = np.load(f"{self.param_save}/param_y_array.npy")

    def _save_params(self):
        """
        Save params (NOT PLOTS) so param plots can pick up where they left off when
        restarting an interrupted training run.
        """
        self.param_save.mkdir(parents=True)
        np.save(f"{self.param_save}/param_names_array.npy", self.param_names_array)
        np.save(f"{self.param_save}/param_x_array.npy", self.param_x_array)
        np.save(f"{self.param_save}/param_y_array.npy", self.param_y_array)

    def _determine_plotted_params(
        self,
        network_params: namedtuple,
        param_plot_num: int,
        state_dim: Tuple,
        action_space: Optional[int],
        discrim_params: Optional[DiscrimParams],
    ):
        """
        Randomly chooses neural network parameters to be plotted. The number from each
        layer which are plotted is `param_plot_num`.

        Args:
            network_params: Network params namedtuple. Specifies number of layers etc.
            param_plot_num: Number of params to plot per layer.
            state_dim: Tuple specifying the dimension of the state space.
        """
        num_shared = network_params.num_shared_layers if (self.using_value and network_params.num_shared_layers is not None) else 0
        prev_layer_size = np.prod(state_dim)

        for count, layer in enumerate(network_params.actor_layers):
            layer_type = "shared" if count < num_shared else "actor"
            layer_num = (
                int(2 * (count))
                if count < num_shared
                else int(2 * (count - num_shared))
            )
            layer_name = f"{layer_type}_layers.{layer_num}.weight"
            self._sample_params(layer_name, layer, prev_layer_size, param_plot_num)
            prev_layer_size = layer

        if self.using_value:
            prev_layer_size = (
                np.prod(state_dim)
                if num_shared == 0
                else network_params.critic_layers[num_shared - 1]
            )
            for count, layer in enumerate(network_params.critic_layers[num_shared:]):
                layer_name = f"critic_layers.{int(2 * count)}.weight"
                self._sample_params(layer_name, layer, prev_layer_size, param_plot_num)
                prev_layer_size = layer
        elif self.using_discrim:
            prev_layer_size = np.prod(state_dim) + action_space
            for count, layer in enumerate(discrim_params.hidden_layers):
                layer_name = f"discrim_layers.{int(2 * count)}.weight"
                self._sample_params(layer_name, layer, prev_layer_size, param_plot_num)
                prev_layer_size = layer

        self.param_names_array = np.array(self.param_names_array)
        self.param_x_array = np.array(self.param_x_array)
        self.param_y_array = np.array(self.param_y_array)

    def determine_demo_nums(self, demo_path: Path, num_demos) -> np.ndarray:
        """
        Only used for imitation learning to pick demo numbers to use.

        Args:
            demo_path: The path to the demo files.
            num_demos: The number of demos to use.
        """
        assert self.using_discrim
        demo_nums_save = self.param_save / "demo_nums.npy"
        if demo_nums_save.exists():
            demo_nums = np.load(f"{demo_nums_save}")
        else:
            demo_nums = np.random.choice(
                os.listdir(f"{demo_path}"), num_demos, replace=False
            )
            np.save(f"{demo_nums_save}", demo_nums)
        return demo_nums

    def _sample_params(self, layer_name, layer, prev_layer_size, param_plot_num):
        self.param_names_array.append(layer_name)
        self.param_x_array.append(
            np.random.randint(low=0, high=layer, size=param_plot_num)
        )
        self.param_y_array.append(
            np.random.randint(low=0, high=prev_layer_size, size=param_plot_num)
        )

    def get_param_plot_nums(self):
        return self.param_names_array, self.param_x_array, self.param_y_array

    def record_data(self, data_dict: Dict):
        """
        Records data - either Plots or Counts.

        Args:
            data_dict: dictionary of name: data for each plot/count to be updated.
        """
        saved_plots = []
        for name, data in data_dict.items():
            recorded = False
            for plot in self.plots:
                if plot.name == name:
                    saved_plot = plot.record_plot_data(data, self.verbose)
                    recorded = True
                    if saved_plot is not None:
                        saved_plots.append(saved_plot)
            if not recorded:
                for count in self.counts:
                    if count.name == name:
                        count.record_count(data)
                        recorded = True
            if not recorded:
                raise ValueError("Name doesn't match any registered plots or counts")
        if not saved_plots == []:
            print(f"Saving plots: {saved_plots}")

    def save_plots(self):
        """Saves data - both plots and counts."""
        for plot in self.plots:
            plot.save_plot(self.verbose)
        for count in self.counts:
            count.save_count(self.verbose)

    def get_count(self, count_name: str):
        """
        Returns value of requested count.

        Args:
            count_name: Name of the count to be retrieved.

        Returns:
            The value of the count with `count_name`.
        """
        for count in self.counts:
            if count.name == count_name:
                return count.value
        raise FileNotFoundError("Count with name `count_name` has not been found.")
