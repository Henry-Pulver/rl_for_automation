import tempfile
from pathlib import Path
import unittest
import numpy as np

from algorithms.actor_critic import ActorCriticParams
from algorithms.plotter import Plotter


class PlotterTests(unittest.TestCase):
    PARAM_FILES = ["param_names_array", "param_x_array", "param_y_array"]

    def test_plotter(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Test plots
            # tmpdirname = "save_test"
            test_plot_name = "save_test"
            test_count_name = "save_count"
            plots = [(test_plot_name, int)]
            counts = [(test_count_name, int)]
            max_plot_size = 4
            layers = (8, 8)
            num_shared = 1
            params = ActorCriticParams(layers, layers, "str", "str", num_shared)
            temp_path = Path(tmpdirname)
            plotter = Plotter(params, temp_path, plots, counts, max_plot_size, 2, (2,))
            data = [1, 3, 2, 8, 9]
            for item in data:
                plotter.record_data({test_plot_name: item, test_count_name: item})
            plotter.save_plots()

            # Checking plot has been created and data has been added as expected
            for i in range(2):
                plot = temp_path / test_plot_name / f"{test_plot_name}_{i+1}.npy"
                assert plot.exists()
                assert (
                    list(np.load(f"{plot}"))
                    == data[max_plot_size * i : max_plot_size * (i + 1)]
                )

            # Checking param files are created as expected
            for param_filename in self.PARAM_FILES:
                assert (temp_path / "params" / f"{param_filename}.npy").exists()

            # Checking NN param plots are created as expected
            num_layers_total = len(layers) * 2 - num_shared
            for layer_num in range(num_layers_total):
                layer_type = "shared" if layer_num < num_shared else "actor"
                layer_type = layer_type if layer_num < len(layers) else "critic"
                layer_num = (
                    int(2 * layer_num)
                    if layer_num < num_shared
                    else int(2 * (layer_num - num_shared))
                )
                layer_num = (
                    layer_num
                    if layer_num < len(layers)
                    else int(2 * (layer_num - len(layers)))
                )
                layer_name = f"{layer_type}_layers.{layer_num}.weight"
                assert (temp_path / layer_name / f"{layer_name}_1.npy").exists()

            # Check count worked as expected
            assert np.load(f"{temp_path}/{test_count_name}.npy") == np.sum(data)
