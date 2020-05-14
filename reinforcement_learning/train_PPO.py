#!/usr/bin/env python3

from pathlib import Path

from algorithms.PPO import HyperparametersPPO, PPOTrainer
from algorithms.actor_critic import ActorCriticParams
from algorithms.parser import get_ppo_parser


def main():
    parser = get_ppo_parser(description="Parser for PPO")
    args = parser.parse_args()

    env_names = (
        [args.env_name]
        if args.env_name is not None
        else ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    )
    solved_rewards = (
        [args.solved_reward] if args.solved_reward is not None else [195, -80, -135]
    )  # stop training if avg_reward > solved_reward
    worst_scores = (
        [args.worst_score] if args.worst_score is not None else [9, -10000, -10000]
    )
    max_episodes = args.max_episodes if args.max_episodes is not None else 1000000
    random_seeds = list(args.random_seeds)
    nn_layers = (args.neurons_per_layer,) * args.num_layers
    learning_rate = args.lr if args.lr is not None else 2e-3
    load_path = Path(args.load_path) if args.load_path is not None else None
    # chooser_params = (100, 1, 100)

    outer_outcomes = []
    try:
        actor_critic_params = ActorCriticParams(
            actor_layers=nn_layers,
            actor_activation=args.activation,
            critic_layers=nn_layers,
            critic_activation=args.activation,
            num_shared_layers=args.num_shared_layers,
        )
        hyp = HyperparametersPPO(
            gamma=0.99,  # discount factor
            lamda=args.lamda,  # GAE weighting factor
            learning_rate=learning_rate,
            T=args.T,  # update policy every n timesteps
            c1=args.value_coeff,  # value function hyperparam
            c2=args.entropy_coeff,  # entropy hyperparam
            num_epochs=args.num_epochs,  # update policy for K epochs
            epsilon=args.epsilon,  # clip parameter for PPO
            d_targ=args.d_targ,  # adaptive KL param
            beta=args.beta,  # fixed KL param
        )
        outcomes = []
        for env_name, solved_reward, worst in zip(
            env_names, solved_rewards, worst_scores
        ):
            outcomes.append(env_name)
            trainer = PPOTrainer(env_name, Path(args.save_base_path), date=args.date)
            outcomes.append(
                trainer.train(
                    solved_reward=solved_reward,
                    hyp=hyp,
                    actor_critic_params=actor_critic_params,
                    random_seeds=random_seeds,
                    load_path=load_path,
                    log_interval=args.log_interval,
                    max_episodes=max_episodes,
                    max_timesteps=args.max_timesteps,
                    ppo_type=args.ppo_type,
                    advantage_type=args.adv_type,
                    param_plot_num=args.param_plot_num,
                    worst_performance=worst,
                    policy_burn_in=args.burn_in_steps,
                    restart=args.r,
                    verbose=args.v,
                )
            )
        outer_outcomes.append(outcomes)
    finally:
        print(f"outcomes:")
        for outer_outcome in outer_outcomes:
            print(outer_outcome)


if __name__ == "__main__":
    main()
