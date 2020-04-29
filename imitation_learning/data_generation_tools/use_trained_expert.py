from pathlib import Path
import gym
import numpy as np
import cv2
import torch
from typing import Callable, Optional

from algorithms.actor_critic import ActorCritic, ActorCriticParams
from algorithms.buffer import DemonstrationBuffer

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
    episode_timeout: int,
    verbose: bool = False,
):
    demo_buffer = DemonstrationBuffer(
        save_path=save_path,
        state_dimension=env.observation_space.shape,
        action_space_size=env.action_space.n,
    )
    demo = 0
    while demo < num_demos:
        run_solution(
            env=env,
            choose_action=policy,
            record_video=False,
            demo_buffer=demo_buffer,
            show_solution=show_recording,
            episode_timeout=episode_timeout,
            verbose=verbose,
        )
        if demo_buffer.get_length() != episode_timeout:
            print(f"Saving demo number: {demo}")
            demo_buffer.save_demos(demo_number=demo)
            demo += 1
        demo_buffer.clear()


def main():
    ## IF ATARI ##
    # game_ref = 0
    # env_name = GAME_STRINGS_TEST[game_ref]

    ## ELSE NOT ATARI ##
    env_names = ["Acrobot-v1"]  # , "CartPole-v1"]
    episode_timeout = 10000

    hidden_layers = (32, 32)
    activation = "tanh"
    policy_params = ActorCriticParams(
        actor_layers=hidden_layers,
        critic_layers=hidden_layers,
        actor_activation=activation,
        critic_activation=activation,
        # num_shared_layers=0,
    )

    for env_name in env_names:
        env = gym.make(env_name).env
        network_load = f"../../solved_networks/PPO_{env_name}.pth"
        policy = ActorCritic(
            env.observation_space.shape, env.action_space.n, policy_params
        )
        net = torch.load(network_load, map_location=device)
        policy.load_state_dict(net)
        policy.eval()
        save_path = Path(f"../expert_demos/{env_name}/")
        num_demos = 9
        record_demonstrations(
            env=env,
            episode_timeout=episode_timeout,
            policy=policy.act,
            num_demos=num_demos,
            save_path=save_path,
            show_recording=False,
            verbose=True,
        )


if __name__ == "__main__":
    main()
