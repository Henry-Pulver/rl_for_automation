from pathlib import Path
import gym
import numpy as np
import cv2
import torch
from typing import Callable, Optional, Tuple

from algorithms.actor_critic import ActorCritic
from algorithms.buffer import DemonstrationBuffer
from algorithms.imitation_learning.behavioural_cloning import pick_action

from atari.consts import GAME_STRINGS_TEST

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
) -> int:
    render_type = "rgb_array" if record_video else "human"

    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter("output.avi", fourcc, video_fps, (600, 400))
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
    hidden_layers: Tuple,
    activation: str,
    param_sharing: bool,
) -> float:
    # Load in neural network from file
    network = ActorCritic(
        action_space=env.action_space.n,
        state_dimension=env.observation_space.shape,
        actor_layers=hidden_layers,
        critic_layers=hidden_layers,
        actor_activation=activation,
        critic_activation=activation,
        param_sharing=param_sharing,
    ).float()
    network.load_state_dict(torch.load(network_load, map_location=device))
    network.eval()

    # Run the env, record the scores
    scores = []
    for _ in range(num_trials):
        scores.append(
            run_solution(
                lambda x: pick_action(state=x, network=network),
                record_video=False,
                env=env,
                show_solution=show_solution,
                episode_timeout=episode_timeout,
            )
        )
    mean_score = np.mean(scores)
    print(f"\nScores: {scores}")
    # logging.info(f"\nScores: {scores}")
    print(f"\nMean score: {mean_score}")
    # logging.info(f"\nMean score: {mean_score}")
    return mean_score


game_ref = 0
env = gym.make(GAME_STRINGS_TEST[game_ref]).env

# network_load = "data/BC/31-01-2020/128-128-128-128/demos_50_seed_3.pt"
network_load = "PPO_breakout_24000.pth"
hidden_layers = (128, 128, 128, 128)
get_average_score(
    network_load=Path(network_load),
    env=env,
    episode_timeout=10000,
    show_solution=True,
    num_trials=100,
    hidden_layers=hidden_layers,
    activation="relu",
    param_sharing=True,
                )

# network_load = "data/BC/28-01-2020/best_breakout_nn.pt"
# hidden_layers = (256, 256, 256)
# get_average_score(
#                     network_load=network_load,
#                     env=env,
#                     episode_timeout=10000,
#                     show_solution=True,
#                     num_trials=100,
#                     hidden_layers=hidden_layers,
#                 )
