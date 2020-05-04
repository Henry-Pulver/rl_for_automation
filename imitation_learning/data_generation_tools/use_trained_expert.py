from pathlib import Path
import gym
import numpy as np
import cv2
import torch
from tqdm import tqdm
from typing import Callable, Optional

from algorithms.actor_critic import ActorCritic, ActorCriticParams
from algorithms.buffer import DemonstrationBuffer

from envs.atari.consts import GAME_STRINGS_TEST, GAME_STRINGS_LEARN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_solution(
    choose_action: Callable,
    env: gym.Env,
    record_video: bool = False,
    show_solution: bool = True,
    episode_timeout: Optional[int] = None,
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
                demo_buffer.update(state, action_chosen)
            state, reward, done, info = env.step(action_chosen)
            demo_buffer.rewards.append(reward)
            total_reward += reward
            if episode_timeout is not None:
                done = step > episode_timeout if not done else done
            step += 1
        if verbose:
            print("Final reward = ", total_reward)
    finally:
        if record_video:
            out.release()
        env.close()
    return total_reward


def record_demonstrations(
    env: gym.Env,
    policy: Callable,
    num_demos: int,
    save_path: Path,
    show_recording: bool,
    episode_timeout: Optional[int] = None,
    verbose: bool = False,
):
    demo_buffer = DemonstrationBuffer(
        save_path=save_path,
        state_dimension=env.observation_space.shape,
        action_space_size=env.action_space.n,
    )
    returns = []
    for demo in tqdm(range(num_demos)):
        returns.append(run_solution(
            env=env,
            choose_action=policy,
            record_video=False,
            demo_buffer=demo_buffer,
            show_solution=show_recording,
            episode_timeout=episode_timeout,
            verbose=verbose,
        ))
        if demo_buffer.get_length() != episode_timeout:
            if verbose:
                print(f"Saving demo number: {demo}")
            demo_buffer.save_demos(demo_number=demo)
        demo_buffer.clear()
    return returns


def main():
    episode_timeout = None
    hidden_layers = (128, 128, 128, 128)
    activation = "relu"
    policy_params = ActorCriticParams(
        actor_layers=hidden_layers,
        critic_layers=hidden_layers,
        actor_activation=activation,
        critic_activation=activation,
        num_shared_layers=3,
    )

    avg_scores = []
    for game_ref in range(4):
        game_ref = 2
        env_name = GAME_STRINGS_TEST[game_ref]
        env = gym.make(env_name).env
        game_name = GAME_STRINGS_LEARN[game_ref]
        avg_scores.append(game_name)
        network_load = f"../../solved_networks/PPO_{game_name}.pth"
        policy = ActorCritic(
            env.observation_space.shape, env.action_space.n, policy_params
        )
        net = torch.load(network_load, map_location=device)
        policy.load_state_dict(net)
        policy.eval()
        print(run_solution(env=env,
                     choose_action=policy.act,
                     record_video=False,
                     show_solution=True,
                     episode_timeout=episode_timeout,
                     verbose=True,))
        # save_path = Path(f"../expert_demos/{game_name}/")
        # num_demos = 100
        # avg_scores.append(np.mean(record_demonstrations(
        #     env=env,
        #     episode_timeout=episode_timeout,
        #     policy=policy.act,
        #     num_demos=num_demos,
        #     save_path=save_path,
        #     show_recording=False,
        #     verbose=False,
        # )))
    print(avg_scores)


if __name__ == "__main__":
    main()
