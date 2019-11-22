import os
import numpy as np

for file in os.listdir("REINFORCE_states/plots/42"):
    if file.endswith(".npy"):
        loaded_file = np.load(f"REINFORCE_states/plots/42/{file}")
        if len(loaded_file.shape) > 1:
            for count, plot in enumerate(loaded_file):
                loaded_file[count] = plot[0::10]
        else:
            loaded_file = loaded_file[0::10]

        np.save(f"REINFORCE_states/plots/42/{file}", loaded_file)
