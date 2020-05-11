import datetime
from pathlib import Path

from discrete_policy import DiscretePolicyParams
from actor_critic import ActorCriticParams
from discriminator import DiscrimParams
from imitation_learning.GAIL import HyperparametersGAIL, GAILTrainer


def main():
    env_names = [
        "CartPole-v1",
        "Acrobot-v1",
        "MountainCar-v0",
    ]
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 10000  # max timesteps in one episode
    random_seeds = list(range(10))
    ppo_types = "clip"
    nn_arch = (32, 32)
    activation = "tanh"
    discrim_params = DiscrimParams(hidden_layers=nn_arch, activation=activation,)
    policy_params = ActorCriticParams(
        actor_layers=nn_arch,
        actor_activation=activation,
        critic_layers=nn_arch,
        critic_activation=activation,
        num_shared_layers=1,
    )
    adv_type = "gae"
    demo_path = Path("expert_demos")
    num_demos = [100, 30, 10]
    restart = True

    date = datetime.date.today().strftime("%d-%m-%Y")
    # date = "26-01-1998"

    outer_outcomes = []
    try:
        for env_name in env_names:
            outcomes = []
            print(f"env name: {env_name}")
            trainer = GAILTrainer(
                env_name=env_name, save_base_path=Path("data"), date=date,
            )
            outcomes.append(env_name)
            for demo_num in num_demos:
                hyp = HyperparametersGAIL(
                    gamma=0.99,  # discount factor
                    lamda=0.95,  # GAE factor
                    learning_rate=4e-3,
                    discrim_lr=2e-2,
                    num_demos=demo_num,
                    batch_size=1024,  # update policy every n timesteps
                    fraction_expert=0.5,  # fraction of discrim data from expert
                    epsilon=0.1,  # clip parameter for PPO
                    c1=0.5,  # value hyperparam
                    c2=0.01,  # entropy hyperparam
                    num_epochs=3,  # update policy for K epochs
                    num_discrim_epochs=4,  # update discriminator for K epochs
                    success_margin=10,  # % less than expert avg that is success
                )
                outcomes.append(demo_num)
                outcomes.append(
                    trainer.train(
                        demo_path=demo_path / env_name,
                        policy_params=policy_params,
                        hyp=hyp,
                        discrim_params=discrim_params,
                        random_seeds=random_seeds,
                        log_interval=log_interval,
                        max_episodes=max_episodes,
                        max_timesteps=max_timesteps,
                        ppo_type=ppo_types,
                        adv_type=adv_type,
                        param_plot_num=10,
                        restart=restart,
                        verbose=False,
                    )
                )
            outer_outcomes.append(outcomes)
    finally:
        print(f"outcomes:")
        for outcome in outer_outcomes:
            print(outcome)


if __name__ == "__main__":
    main()
