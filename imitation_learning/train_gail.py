import datetime
from pathlib import Path

from discrete_policy import DiscretePolicyParams
from actor_critic import ActorCriticParams
from discriminator import DiscrimParams
from imitation_learning.GAIL import HyperparametersGAIL, train_gail


def main():
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 10000  # max timesteps in one episode
    random_seeds = list(range(2, 8))
    ppo_types = "clip"
    chooser_params = (100, 1, 100)
    discrim_params = DiscrimParams(hidden_layers=(32, 32, 32), activation="tanh",)
    policy_params = ActorCriticParams(
        actor_layers=(32, 32),
        actor_activation="tanh",
        critic_layers=(32, 32),
        critic_activation="tanh",
        num_shared_layers=1,
    )
    # policy_params = DiscretePolicyParams(
    #     actor_layers=(32, 32), actor_activation="tanh",
    # )

    #### OTHER STUFF ####
    env_names = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    # env_names = ["MountainCar-v0"]
    adv_type = "gae"
    # solved_rewards = [-80, 195, -135]  # stop training if avg_reward > solved_reward
    # env_names = ["Acrobot-v1"]
    # solved_rewards = [-80]  # stop training if avg_reward > solved_reward
    # env_names = ["MountainCar-v0"]
    # solved_rewards = [-135]  # stop training if avg_reward > solved_reward
    demo_path = Path("expert_demos")
    # num_demos = reversed([1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 30, 40, 50, 75, 100])
    num_demos = [100]
    restart = True

    date = datetime.date.today().strftime("%d-%m-%Y")

    outcomes = []
    try:
        for env_name in env_names:
            print(f"env name: {env_name}")
            outcomes.append(env_name)
            for demo_num in num_demos:
                hyp = HyperparametersGAIL(
                    gamma=0.99,  # discount factor
                    lamda=0.95,  # GAE factor
                    learning_rate=4e-3,
                    discrim_lr=4e-3,
                    num_demos=demo_num,
                    batch_size=1024,  # update policy every n timesteps
                    fraction_expert=0.5,  # fraction of discrim data from expert
                    epsilon=0.2,  # clip parameter for PPO
                    c1=0.5,  # value hyperparam
                    c2=0.01,  # entropy hyperparam
                    num_epochs=3,  # update policy for K epochs
                    num_discrim_epochs=1,
                    success_margin=10,  # % less than expert avg that is success
                    entropy=True,
                )
                outcomes.append(num_demos)
                outcomes.append(
                    train_gail(
                        demo_path=demo_path / env_name,
                        env_name=env_name,
                        hyp=hyp,
                        policy_params=policy_params,
                        discrim_params=discrim_params,
                        random_seeds=random_seeds,
                        log_interval=log_interval,
                        max_episodes=max_episodes,
                        max_timesteps=max_timesteps,
                        ppo_type=ppo_types,
                        adv_type=adv_type,
                        date=date,
                        param_plot_num=10,
                        restart=restart,
                        policy_burn_in=2000,
                        verbose=False,
                    )
                )
    finally:
        print(f"outcomes:")
        for outcome in outcomes:
            print(outcome)


if __name__ == "__main__":
    main()
