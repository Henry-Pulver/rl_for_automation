#!/usr/bin/env python3

import torch
from typing import Optional, List, Tuple
from pathlib import Path

from algorithms.PPO import HyperparametersPPO, PPO
from algorithms.actor_critic import ActorCriticParams
from algorithms.action_chooser import ActionChooser
from algorithms.buffer import PPOExperienceBuffer
from algorithms.parser import get_actor_critic_parser
from algorithms.trainer import Trainer, RunLogger
from algorithms.utils import generate_save_location


class REINFORCETrainer(Trainer):
    def train(
        self,
        max_episodes: int,
        log_interval: int,
        hyp: HyperparametersPPO,
        actor_critic_params: ActorCriticParams,
        solved_reward: float,
        random_seeds: List,
        load_path: Optional[str] = None,
        max_timesteps: Optional[int] = None,
        render: bool = False,
        advantage_type: str = "monte_carlo_baseline",
        param_plot_num: int = 2,
        policy_burn_in: int = 0,
        chooser_params: Tuple = (None, None, None),
        restart: bool = False,
        log_type: str = "legacy",
        verbose: bool = False,
    ):
        try:
            episode_numbers = []
            for random_seed in random_seeds:
                self.set_seed(random_seed)

                buffer = PPOExperienceBuffer(self.state_dim, self.action_dim)

                save_path = generate_save_location(
                    Path("data"),
                    actor_critic_params.actor_layers,
                    "REINFORCE",
                    self.env_name,
                    random_seed,
                    "REINFORCE",
                    self.date,
                )
                self.restart(save_path, restart)

                reinforce = PPO(
                    state_dimension=self.state_dim,
                    action_space=self.action_dim,
                    policy_params=actor_critic_params,
                    hyperparameters=hyp,
                    save_path=save_path,
                    ppo_type="clip",
                    advantage_type=advantage_type,
                    param_plot_num=param_plot_num,
                    policy_burn_in=policy_burn_in,
                    verbose=verbose,
                )
                if load_path is not None:
                    reinforce.policy.load_state_dict(torch.load(load_path))

                # logging variables
                action_chooser = ActionChooser(*chooser_params)
                ep_num_start = reinforce.plotter.get_count("episode_num")
                run_logger = RunLogger(max_episodes, log_type, 0.99, verbose)

                # training loop
                print(f"Starting running from episode number {ep_num_start + 1}\n")
                for ep_num in range(ep_num_start + 1, max_episodes + 1):  # Run episodes
                    state = self.env.reset()
                    action_chooser.reset()
                    t = 0
                    keep_running = True if max_timesteps is None else t < max_timesteps
                    while keep_running:  # Run 1 episode
                        t += 1
                        keep_running = (
                            True if max_timesteps is None else t < max_timesteps
                        )

                        # Running policy_old:
                        action = reinforce.policy_old.act(state, buffer)
                        action = action_chooser.step(action)
                        state, reward, done, _ = self.env.step(action)

                        # Saving reward and is_terminal:
                        buffer.rewards.append(reward)
                        buffer.is_terminal.append(done)

                        run_logger.update(1, reward)
                        if render:
                            self.env.render()
                        if done:
                            break
                    reinforce.update(buffer, ep_num)
                    buffer.clear()
                    reinforce.plotter.record_data(
                        {
                            "rewards": run_logger.ep_reward,
                            "num_steps_taken": t,
                            "episode_num": 1,
                        }
                    )
                    run_logger.end_episode()

                    if ep_num % log_interval == 0:
                        run_logger.output_logs(ep_num, log_interval)
                        reinforce.save()

                        # stop training if avg_reward > solved_reward
                        if run_logger.avg_reward > solved_reward:
                            print("########## Solved! ##########")
                            break
                episode_numbers.append(ep_num)
            print(f"episode_numbers: {episode_numbers}")
            return episode_numbers
        except KeyboardInterrupt as interrupt:
            reinforce.save()
            raise interrupt


def main():
    parser = get_actor_critic_parser(description="Parser for REINFORCE algorithm")
    args = parser.parse_args()
    max_episodes = args.max_episodes if args.max_episodes is not None else 1000000
    random_seeds = list(args.random_seeds)
    adv_type = args.adv_type
    nn_layers = (args.neurons_per_layer,) * args.num_layers
    actor_critic_params = ActorCriticParams(
        actor_layers=nn_layers,
        actor_activation=args.activation,
        critic_layers=nn_layers,
        critic_activation=args.activation,
        num_shared_layers=args.num_shared_layers,
    )
    learning_rate = args.lr if args.lr is not None else 2e-3
    hyp = HyperparametersPPO(
        gamma=0.99,  # discount factor
        lamda=args.lamda,  # GAE weighting factor
        learning_rate=learning_rate,
        epsilon=10000,  # clip parameter for PPO
        c1=args.value_coeff,  # value function hyperparam
        c2=0,  # entropy hyperparam
        num_epochs=1,  # update policy for K epochs
    )

    env_names = (
        [args.env_name]
        if args.env_name is not None
        else ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    )
    solved_rewards = (
        [args.solved_reward] if args.solved_reward is not None else [195, -80, -135]
    )  # stop training if avg_reward > solved_reward

    load_path = Path(args.load_path) if args.load_path is not None else None

    outcomes = []
    try:
        for env_name, solved_reward in zip(env_names, solved_rewards):
            outcomes.append(env_name)
            trainer = REINFORCETrainer(
                env_name, Path(args.save_base_path), date=args.date
            )
            outcomes.append(adv_type)
            outcomes.append(
                trainer.train(
                    solved_reward=solved_reward,
                    hyp=hyp,
                    actor_critic_params=actor_critic_params,
                    random_seeds=random_seeds,
                    load_path=load_path,
                    log_interval=args.log_interval,  # print avg reward in the interval
                    max_episodes=max_episodes,
                    max_timesteps=args.max_timesteps,  # max timesteps in one episode
                    advantage_type=adv_type,
                    param_plot_num=args.param_plot_num,
                    restart=args.r,
                    verbose=args.v,
                )
            )
    finally:
        print(f"outcomes:")
        for outcome in outcomes:
            print(outcome)


if __name__ == "__main__":
    main()
