import numpy as np
from collections import namedtuple
from pathlib import Path
from typing import Tuple, List, Dict


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
            counted_data:
        """
        assert type(counted_data) == self.datatype
        self.value += counted_data
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
                np.load(f"{self.plot_save_path}/{self._get_filename(self.file_num)}", allow_pickle=True)
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

    def record_data(self, data_entry):
        if not type(data_entry) == self.datatype:
            print(self.name)
            print(type(data_entry))
        assert type(data_entry) == self.datatype
        self.plot.append(data_entry)
        if len(self.plot) >= self.max_plot_size:
            self.save_plot()
            self.file_num += 1

    def save_plot(self):
        print(f"Saving plot: {self.name} at {self.plot_save_path}/{self._get_filename(self.file_num)}")
        np.save(
            f"{self.plot_save_path}/{self._get_filename(self.file_num)}",
            np.array(self.plot),
        )
        del self.plot[:]


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
    ):
        self.save_path = save_path
        self.param_save = self.save_path / "params"
        self.param_names_array = []
        self.param_x_array = []
        self.param_y_array = []
        if self.param_save.exists():
            self._load_params()
        else:
            self._determine_plotted_params(network_params, param_plot_num, state_dim)
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
        self, network_params: namedtuple, param_plot_num: int, state_dim: Tuple
    ):
        """
        Randomly chooses neural network parameters to be plotted. The number from each
        layer which are plotted is `param_plot_num`.

        Args:
            network_params: Network params namedtuple. Specifies number of layers etc.
            param_plot_num: Number of params to plot per layer.
            state_dim: Tuple specifying the dimension of the state space.
        """
        num_shared = network_params.num_shared_layers
        prev_layer_size = np.prod(state_dim)

        for count, layer in enumerate(network_params.actor_layers):
            layer_type = 'shared' if count < num_shared else 'actor'
            layer_num = int(2 * (count)) if count < num_shared else int(2 * (count - num_shared))
            layer_name = f"{layer_type}_layers.{layer_num}.weight"
            self.param_names_array.append(layer_name)
            self.param_x_array.append(
                np.random.randint(low=0, high=layer, size=param_plot_num)
            )
            self.param_y_array.append(
                np.random.randint(low=0, high=prev_layer_size, size=param_plot_num)
            )
            prev_layer_size = layer

        prev_layer_size = (
            np.prod(state_dim)
            if num_shared == 0
            else network_params.critic_layers[num_shared - 1]
        )
        for count, layer in enumerate(network_params.critic_layers[num_shared:]):
            layer_name = f"critic_layers.{int(2 * count)}.weight"
            self.param_names_array.append(layer_name)
            self.param_x_array.append(
                np.random.randint(low=0, high=layer, size=param_plot_num)
            )
            self.param_y_array.append(
                np.random.randint(low=0, high=prev_layer_size, size=param_plot_num)
            )
            prev_layer_size = layer

        self.param_names_array = np.array(self.param_names_array)
        self.param_x_array = np.array(self.param_x_array)
        self.param_y_array = np.array(self.param_y_array)

    def get_param_plot_nums(self):
        return self.param_names_array, self.param_x_array, self.param_y_array

    def record_data(self, data_dict: Dict):
        """
        Records data - either Plots or Counts.

        Args:
            data_dict: dictionary of name: data for each plot/count to be updated.
        """
        for name, data in data_dict.items():
            recorded = False
            for plot in self.plots:
                if plot.name == name:
                    plot.record_data(data)
                    recorded = True
            if not recorded:
                for count in self.counts:
                    if count.name == name:
                        count.record_count(data)
                        recorded = True
            if not recorded:
                raise ValueError("Name doesn't match any registered plots or counts")

    def save_plots(self):
        """
        Saves data - Counts automagically save counts, so only Plots need be saved when
        training is interrupted.
        """
        for plot in self.plots:
            plot.save_plot()
