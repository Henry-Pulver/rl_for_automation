from PPO import HyperparametersPPO, train_ppo
from actor_critic import ActorCriticParams
import datetime


def main():
    # env_names = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    # env_names = ["MountainCar-v0", "Acrobot-v1"]
    env_names = ["Acrobot-v1"]
    # solved_rewards = [195, -80, -135]  # stop training if avg_reward > solved_reward
    solved_rewards = [-135, -80]  # stop training if avg_reward > solved_reward
    worst_scores = [-10000, -10000]
    # worst_scores = [-10000]
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 10000  # max timesteps in one episode
    random_seeds = list(range(5))
    ppo_type = "fixed_KL"
    adv_type = "monte_carlo_baseline"
    clip_param = 0.2
    beta_values = [1]
    # chooser_params = (100, 1, 100)

    date = datetime.date.today().strftime("%d-%m-%Y")

    outer_outcomes = []
    try:
        for beta in beta_values:
            actor_critic_params = ActorCriticParams(
                actor_layers=(32, 32),
                actor_activation="tanh",
                critic_layers=(32, 32),
                critic_activation="tanh",
                num_shared_layers=0,
            )
            # if num_shared in [1, 0]:
            #     random_seeds = random_seeds[5:]
            # outer_outcomes.append(f"num_shared: {num_shared}")
            hyp = HyperparametersPPO(
                gamma=0.99,  # discount factor
                # lamda=0.95,  # GAE weighting factor
                learning_rate=2e-3,
                T=1024,  # update policy every n timesteps
                # epsilon=clip_param,  # clip parameter for PPO
                c1=0.5,  # value function hyperparam
                c2=0.01,  # entropy hyperparam
                num_epochs=3,  # update policy for K epochs
                # d_targ=d_targ,          # adaptive KL param
                beta=beta,  # fixed KL param
            )
            outcomes = []
            for env_name, solved_reward, worst in zip(
                env_names, solved_rewards, worst_scores
            ):
                outcomes.append(env_name)
                outcomes.append(beta)
                outcomes.append(
                    train_ppo(
                        env_name=env_name,
                        solved_reward=solved_reward,
                        hyp=hyp,
                        actor_critic_params=actor_critic_params,
                        random_seeds=random_seeds,
                        log_interval=log_interval,
                        max_episodes=max_episodes,
                        max_timesteps=max_timesteps,
                        ppo_type=ppo_type,
                        advantage_type=adv_type,
                        date=date,
                        param_plot_num=10,
                        restart=True,
                        worst_performance=worst,
                        # policy_burn_in=0,
                    )
                )
            outer_outcomes.append(outcomes)
    finally:
        print(f"outcomes:")
        for outer_outcome in outer_outcomes:
            print(outer_outcome)


if __name__ == "__main__":
    main()
