from pathlib import Path
import gym
import torch
import logging
import numpy as np
import cv2
import torch
from typing import Callable, Optional, Tuple, Any

from algorithms.discrete_policy import DiscretePolicy
from algorithms.buffer import DemonstrationBuffer
from algorithms.imitation_learning.behavioural_cloning import pick_action


def run_solution(
    pick_action: Callable,
    env: gym.Env,
    record_video: bool = False,
    show_solution: bool = True,
    episode_timeout: int = 200,
    demo_buffer: Optional[DemonstrationBuffer] = None,
    video_fps: int = 60,
    no_action_max: int = 30,
    verbose: bool = False,
) -> int:
    render_type = "rgb_array" if record_video else "human"

    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter("output.avi", fourcc, video_fps, (600, 400))

    state = env.reset()
    done = False
    total_reward, reward, step, only_no_action = 0, 0, 0, True
    try:
        while not done:
            if show_solution or record_video:
                rgb_array = env.render(render_type)
                if record_video:
                    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                    out.write(bgr_array)
            action_chosen = pick_action(state)
            only_no_action = action_chosen == 0 if only_no_action else only_no_action
            if only_no_action and step > no_action_max:
                action_chosen = 1
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
    *args: Any,
) -> float:
    # Load in neural network from file
    network = DiscretePolicy(
        action_space=env.action_space.n,
        state_dimension=env.observation_space.shape,
        hidden_layers=hidden_layers,
    ).float()
    network.load_state_dict(torch.load(network_load))
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