from PPO import HyperparametersPPO, train_ppo
from actor_critic import ActorCriticParams
import datetime


def main():
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 10000  # max timesteps in one episode
    random_seeds = list(range(5))
    ppo_types = ["clip"]
    adv_types = ["monte_carlo_baseline"]
    clip_params = [0]
    chooser_params = (100, 1, 100)
    actor_critic_params = ActorCriticParams(
        actor_layers=(32, 32),
        actor_activation="tanh",
        critic_layers=(32, 32),
        critic_activation="tanh",
        # num_shared_layers=0,
    )
    #### OTHER STUFF ####
    env_names = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    solved_rewards = [195, -80, -135]  # stop training if avg_reward > solved_reward

    date = datetime.date.today().strftime("%d-%m-%Y")

    outer_outcomes, outcomes = [], []
    try:
        for clip_param in clip_params:
            outer_outcomes.append(f"clip param: {clip_param}")
            hyp = HyperparametersPPO(
                gamma=0.99,  # discount factor
                lamda=0.95,  # GAE weighting factor
                learning_rate=2e-3,
                entropy=True,
                T=1024,  # update policy every n timesteps
                epsilon=clip_param,  # clip parameter for PPO
                c1=0.5,  # value function hyperparam
                c2=0.01,  # entropy hyperparam
                num_epochs=3,  # update policy for K epochs
                # d_targ=d_targ,          # adaptive KL param
                # beta=0.003,              # fixed KL param
            )
            for env_name, solved_reward in zip(env_names, solved_rewards):
                outcomes.append(env_name)
                for adv_type in adv_types:
                    outcomes.append(adv_type)
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
                            ppo_type=ppo_types[0],
                            advantage_type=adv_type,
                            date=date,
                            param_plot_num=20,
                            restart=True
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
