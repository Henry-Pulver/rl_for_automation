from pathlib import Path
import gym
import numpy as np
import cv2
import torch
from tqdm import tqdm
from typing import Callable, Optional
from collections import namedtuple

from algorithms.discrete_policy import DiscretePolicy
from algorithms.actor_critic import ActorCritic, ActorCriticParams
from algorithms.buffer import DemonstrationBuffer

from envs.atari.consts import GAME_STRINGS_TEST, GAME_STRINGS_LEARN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_solution(
    choose_action: Callable,
    env: gym.Env,
    record_video: bool = False,
    show_solution: bool = True,
    episode_timeout: int = 200,
    demo_buffer: Optional[DemonstrationBuffer] = None,
    video_fps: int = 60,
    verbose: bool = False,
    video_filename: str = "output",
) -> int:
    render_type = "rgb_array" if record_video else "human"

    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        out = cv2.VideoWriter(f"{video_filename}.mp4", fourcc, video_fps, (600, 400))
    env.seed(np.random.randint(low=0, high=2000))
    state = env.reset()
    done = False
    total_reward, reward, step = 0, 0, 0
    try:
        while not done:
            if show_solution or record_video:
                rgb_array = env.render(render_type)
                if record_video:
                    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                    out.write(bgr_array)
            action_chosen = choose_action(state)
            if demo_buffer is not None:
                demo_buffer.update(state, action_chosen, reward=reward)
            state, reward, done, info = env.step(action_chosen)
            total_reward += reward
            done = step > episode_timeout if not done else done
            step += 1
        if verbose:
            print("final reward = ", total_reward)
    finally:
        if record_video:
            out.release()
        env.close()
    return total_reward


def get_average_score(
    network_load: Path,
    episode_timeout: int,
    show_solution: bool,
    num_trials: int,
    env: gym.Env,
    params: namedtuple,
) -> float:
    # Load in neural network from file
    net_type = ActorCritic if type(params) == ActorCriticParams else DiscretePolicy
    network = net_type(
        action_space=env.action_space.n,
        state_dimension=env.observation_space.shape,
        params=params,
    ).float()
    network.load_state_dict(torch.load(network_load, map_location=device))
    network.eval()

    # Run the env, record the scores
    scores = []
    for trial in tqdm(range(num_trials)):
        video_filename = f"PPO_breakout_{trial}"
        scores.append(
            run_solution(
                choose_action=network.act,
                record_video=False,  # Ignore this - instead use Windows + G and record
                env=env,
                show_solution=show_solution,
                episode_timeout=episode_timeout,
                video_filename=video_filename,
            )
        )
    print(f"\nScores: {scores}")
    print(f"\nMean score: {np.mean(scores)}")
    print(f"\nStd Dev: {np.sqrt(np.var(scores))}")
    return np.mean(scores)


def main():
    game_ref = 3
    test_game_name = GAME_STRINGS_TEST[game_ref]
    game_name = GAME_STRINGS_LEARN[game_ref]

    # for game_ref in range(4):
    print(game_name)
    env = gym.make(game_name).env

    network_load = f"../solved_networks/PPO_{game_name}.pth"
    hidden_layers = (128, 128, 128, 128)
    activation = "relu"
    actor_critic_params = ActorCriticParams(
        actor_layers=hidden_layers,
        critic_layers=hidden_layers,
        actor_activation=activation,
        critic_activation=activation,
        num_shared_layers=3,
    )
    get_average_score(
        network_load=Path(network_load),
        env=env,
        episode_timeout=1000000,
        show_solution=False,
        num_trials=100,
        params=actor_critic_params,
    )


if __name__ == "__main__":
    main()
