import numpy as np
import gym
import torch
import datetime
from typing import Optional, List, Tuple
from shutil import rmtree
from pathlib import Path

from PPO import HyperparametersPPO, PPO
from actor_critic import ActorCriticParams
from algorithms.action_chooser import ActionChooser
from algorithms.buffer import PPOExperienceBuffer
from algorithms.utils import generate_save_location


def train_reinforce(
    env_name: str,
    max_episodes: int,
    log_interval: int,
    hyp: HyperparametersPPO,
    actor_critic_params: ActorCriticParams,
    solved_reward: float,
    random_seeds: List,
    load_path: Optional[str] = None,
    max_timesteps: Optional[int] = None,
    render: bool = False,
    verbose: bool = False,
    advantage_type: str = "monte_carlo_baseline",
    date: Optional[str] = None,
    param_plot_num: int = 2,
    policy_burn_in: int = 0,
    chooser_params: Tuple = (None, None, None),
    restart: bool = False,
):
    try:
        env = gym.make(env_name).env
        state_dim = env.observation_space.shape
        action_dim = env.action_space.n

        episode_numbers = []
        for random_seed in random_seeds:
            if random_seed is not None:
                torch.manual_seed(random_seed)
                env.seed(random_seed)
                print(f"Set random seed to: {random_seed}")

            buffer = PPOExperienceBuffer(state_dim, action_dim)

            save_path = generate_save_location(
                Path("data"),
                actor_critic_params.actor_layers,
                "REINFORCE",
                env_name,
                random_seed,
                "REINFORCE",
                date,
            )
            if restart:
                if save_path.exists():
                    print("Old data removed!")
                    rmtree(save_path)

            reinforce = PPO(
                state_dimension=state_dim,
                action_space=action_dim,
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
            avg_length = 0
            timestep = 0  # Determines when to update the network
            running_reward = 0
            action_chooser = ActionChooser(*chooser_params)
            ep_num_start = reinforce.plotter.get_count("episode_num")

            # training loop
            print(f"Starting running from episode number {ep_num_start + 1}\n")
            for ep_num in range(ep_num_start + 1, max_episodes + 1):  # Run episodes
                state = env.reset()
                ep_total_reward = 0
                t = 0
                action_chooser.reset()
                keep_running = True if max_timesteps is None else t < max_timesteps
                while keep_running:  # Run 1 episode
                    timestep += 1
                    t += 1
                    keep_running = True if max_timesteps is None else t < max_timesteps

                    # Running policy_old:
                    action = reinforce.policy_old.act(state, buffer)
                    action = action_chooser.step(action)
                    state, reward, done, _ = env.step(action)

                    # Saving reward and is_terminal:
                    buffer.rewards.append(reward)
                    buffer.is_terminal.append(done)

                    ep_total_reward += reward
                    if render:
                        env.render()
                    # update if its time
                    if done:
                        reinforce.update(buffer, ep_num)
                        buffer.clear()
                        timestep = 0
                        break

                avg_length += t / log_interval
                running_reward += ep_total_reward / log_interval
                reinforce.plotter.record_data(
                    {"rewards": ep_total_reward, "num_steps_taken": t, "episode_num": 1}
                )

                ep_str = ("{0:0" + f"{len(str(max_episodes))}" + "d}").format(ep_num)
                if verbose:
                    print(
                        f"Episode {ep_str} of {max_episodes}. \t Total reward = {ep_total_reward}"
                    )

                if ep_num % log_interval == 0:
                    print(
                        f"Episode {ep_str} of {max_episodes}. \t Avg length: {int(avg_length)} \t Reward: {np.round(running_reward, 1)}"
                    )
                    reinforce.save()

                    # stop training if avg_reward > solved_reward
                    if running_reward > solved_reward:
                        print("########## Solved! ##########")
                        break
                    running_reward = 0
                    avg_length = 0
            episode_numbers.append(ep_num)
        print(f"episode_numbers: {episode_numbers}")
        return episode_numbers
    except KeyboardInterrupt as interrupt:
        reinforce.save()
        raise interrupt


def main():
    log_interval = 20  # print avg reward in the interval
    max_episodes = 100000  # max training episodes
    max_timesteps = 10000  # max timesteps in one episode
    random_seeds = list(range(0, 5))
    adv_types = ["monte_carlo_baseline"]
    actor_critic_params = ActorCriticParams(
        actor_layers=(32, 32),
        actor_activation="tanh",
        critic_layers=(32, 32),
        critic_activation="tanh",
        num_shared_layers=1,
    )
    hyp = HyperparametersPPO(
        gamma=0.99,  # discount factor
        # lamda=0.95,  # GAE weighting factor
        learning_rate=2e-3,
        entropy=False,
        T=1024,  # update policy every n timesteps
        epsilon=100,  # clip parameter for PPO
        c1=0.5,  # value function hyperparam
        c2=0,  # entropy hyperparam
        num_epochs=1,  # update policy for K epochs
    )

    env_names = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    solved_rewards = [195, -80, -135]  # stop training if avg_reward > solved_reward

    date = datetime.date.today().strftime("%d-%m-%Y")

    outcomes = []
    try:
        for env_name, solved_reward in zip(env_names, solved_rewards):
            outcomes.append(env_name)
            for adv_type in adv_types:
                outcomes.append(adv_type)
                outcomes.append(
                    train_reinforce(
                        env_name=env_name,
                        solved_reward=solved_reward,
                        hyp=hyp,
                        actor_critic_params=actor_critic_params,
                        random_seeds=random_seeds,
                        log_interval=log_interval,
                        max_episodes=max_episodes,
                        max_timesteps=max_timesteps,
                        advantage_type=adv_type,
                        date=date,
                        param_plot_num=10,
                        restart=True,
                    )
                )
    finally:
        print(f"outcomes:")
        for outcome in outcomes:
            print(outcome)


if __name__ == "__main__":
    main()
