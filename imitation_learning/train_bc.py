from pathlib import Path
import datetime

from algorithms.discrete_policy import DiscretePolicyParams
from algorithms.imitation_learning.behavioural_cloning import train_behavioural_cloning


def main():
    env_names = [
        "CartPole-v1",
        # "MountainCar-v0",
        # "Acrobot-v1",
    ]
    # date = datetime.date.today().strftime("%d-%m-%Y")
    date = "1235457"
    num_demos = [1, 3, 10, 30, 100]
    num_epochs = 50
    minibatch_size = 64
    random_seeds = list(range(10))
    steps_per_epoch_list = [8000, 25000, 2000]
    test_demo_timeout = 10000
    # chooser_params = (None, None, None)

    # NN Architecture - hidden layer sizes
    hidden_layers = (32, 32)

    discrete_policy_params = DiscretePolicyParams(
        actor_layers=hidden_layers, actor_activation="tanh"
    )

    for env_name, steps_per_epoch in zip(env_names, steps_per_epoch_list):
        print(f"\nRunning BC on env: {env_name}")
        demo_path = Path(f"expert_demos/{env_name}/")
        for demo_num in num_demos:
            print(f"Num demos: {demo_num}")
            train_behavioural_cloning(
                num_demos=demo_num,
                env_name=env_name,
                epoch_nums=num_epochs,
                minibatch_size=minibatch_size,
                demo_path=demo_path,
                params=discrete_policy_params,
                learning_rate=1e-4,
                date=date,
                param_plot_num=5,
                random_seeds=random_seeds,
                steps_per_epoch=steps_per_epoch,
                num_test_trials=20,
                restart=False,
                test_demo_timeout=test_demo_timeout,
                # chooser_params=chooser_params,
            )


if __name__ == "__main__":
    main()
