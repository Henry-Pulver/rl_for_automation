import numpy as np
from pathlib import Path
import gym
from envs.atari.get_avg_score import get_average_score
from algorithms.discrete_policy import DiscretePolicyParams
from algorithms.utils import generate_save_location


def main():
    hidden_layers = (32, 32)
    date = "09-05-2020"

    discrete_policy_params = DiscretePolicyParams(
        actor_layers=hidden_layers, actor_activation="tanh"
    )
    save_base_path = Path("data")
    env_names = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    num_epochs = [50, 30, 30]
    num_seeds = 10
    demo_nums = [1, 3, 10, 30, 100]
    overall_list = []
    for env_name, epoch in zip(env_names, num_epochs):
        overall_list.append(env_name)
        env_list = []
        for demo_num in demo_nums:
            means, std_devs =[], []
            for seed in range(num_seeds):
                save_location = generate_save_location(
                    save_base_path,
                    discrete_policy_params.actor_layers,
                    f"BC",
                    env_name,
                    seed,
                    f"demos-{demo_num}",
                    date,
                )
                env = gym.make(env_name).env
                mean_score, std_dev = get_average_score(
                    network_load=save_location / f"BC-{epoch}-epochs.pth",
                    env=env,
                    episode_timeout=10000,
                    num_trials=25,
                    params=discrete_policy_params,
                    chooser_params=(None, None, None),
                )
                means.append(mean_score)
                std_devs.append(std_dev)
            demo_std_dev = np.sqrt(np.mean(np.array(std_devs) ** 2))
            env_list.append(f"Num demos: {demo_num}\t {np.round(np.mean(means), 1)} \pm {np.round(demo_std_dev, 1)}")
        overall_list.append(env_list)
    for entry in overall_list:
        print(entry)


if __name__ == "__main__":
    main()
