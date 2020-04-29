from pathlib import Path
import datetime

from algorithms.discrete_policy import DiscretePolicyParams
from algorithms.imitation_learning.behavioural_cloning import train_network


def main():
    env_names = [
        "CartPole-v1",
        "MountainCar-v0",
        "Acrobot-v1",
    ]
    date = datetime.date.today().strftime("%d-%m-%Y")
    num_demos = [1, 2, 5, 10, 20, 50, 100]  # , 200, 500, 1000]
    # num_demos = [100]  # , 200, 500, 1000]
    num_epochs = list(range(1, 50 + 1))
    minibatch_size = 64
    random_seeds = list(range(8))
    steps_per_epoch_list = [4000, 10000, 2000]

    # NN Architecture - hidden layer sizes
    hidden_layers = (32, 32)
    architecture_str = ""
    for layer in hidden_layers:
        architecture_str += str(layer) + "-"

    discrete_policy_params = DiscretePolicyParams(
        actor_layers=hidden_layers, actor_activation="relu"
    )

    for env_name, steps_per_epoch in zip(env_names, steps_per_epoch_list):
        print(f"\nNew Env: {env_name}")
        demo_path = Path(f"expert_demos/{env_name}/")
        for demo_num in num_demos:
            print(f"Num demos: {demo_num}")
            train_network(
                num_demos=demo_num,
                env_name=env_name,
                epoch_nums=num_epochs,
                minibatch_size=minibatch_size,
                demo_path=demo_path,
                discrete_policy_params=discrete_policy_params,
                learning_rate=1e-4,
                date=date,
                param_plot_num=5,
                random_seeds=random_seeds,
                steps_per_epoch=steps_per_epoch,
                num_test_trials=40,
                restart=False,
            )


if __name__ == "__main__":
    main()
