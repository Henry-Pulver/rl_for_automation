import os
import numpy as np


def concatenate_plots():
    ref_numbers = [40, 41]
    filenames = [
        "avg_delta_plot",
        "baseline_plot",
        "moving_avg",
        "policy_plot",
        "returns",
    ]
    os.makedirs("REINFORCE_states/plots/concatenated", exist_ok=True)

    for filename in filenames:
        concatenated_file = np.array([])
        for ref_num in ref_numbers:
            file_path = f"REINFORCE_states/plots/{ref_num}/{filename}_{ref_num}.npy"
            concatenated_file = np.append(concatenated_file, np.load(file_path))
        np.save(
            f"REINFORCE_states/plots/concatenated/{filename}_{ref_numbers[0]}-{ref_numbers[1]}.npy",
            concatenated_file,
        )


def main():
    policy = np.load("REINFORCE_states/plots/50/policy_plot_50.npy")
    print(policy[0:9])
    # list_of_sizes = []
    # for file in os.listdir("REINFORCE_states/plots/40"):
    #     if file.endswith(".npy"):
    #         loaded_file = np.load(f"REINFORCE_states/plots/40/{file}")
    #         list_of_sizes.append(loaded_file.shape)
    #
    # list_of_sizes.sort()
    # print(list_of_sizes)
    #
    # for file in os.listdir("REINFORCE_states/plots/40"):
    #     if file.endswith(".npy"):
    #         loaded_file = np.load(f"REINFORCE_states/plots/40/{file}")
    #         print(file)
    #         if loaded_file.shape[0] > 11982:
    #             loaded_file = loaded_file.reshape((9, -1))
    #             new_file = np.zeros((9, 1199))
    #             for count, plot in enumerate(loaded_file):
    #                 new_file[count] = plot[0::10]
    #             new_file = new_file.flatten()
    #         else:
    #             new_file = loaded_file[0::10]
    #         print(new_file.shape)
    # concatenate_plots()
    # np.save(f"REINFORCE_states/plots/40/{file}", loaded_file)


if __name__ == "__main__":
    main()

# new_file = np.zeros((9, 23229))
# line = np.ones(23229)
# new_file[0] = line
# print(new_file[1].shape)
# print(new_file[2].shape)
