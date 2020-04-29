import random
import gym
import cv2
from typing import Callable, Any, Optional

from algorithms.buffer import DemonstrationBuffer


def test_solution(
    pick_action: Callable,
    record_video: bool = False,
    show_solution: bool = True,
    episode_timeout: int = 200,
    demo_buffer: Optional[DemonstrationBuffer] = None,
    video_fps: int = 60,
    verbose: bool = False,
    env_name: str = "MountainCar-v0",
    *args: Any,
) -> int:
    env = gym.make(env_name).env
    render_type = "rgb_array" if record_video else "human"

    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter("output.avi", fourcc, video_fps, (600, 400))

    state = env.reset()
    done = False
    total_reward, reward = 0, 0
    try:
        while not done:
            if show_solution or record_video:
                rgb_array = env.render(render_type)
                if record_video:
                    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                    out.write(bgr_array)
            best_actions = pick_action(state, *args)
            if len(best_actions) > 1:
                if verbose:
                    print(f"Random choice required: {best_actions}")
            action_chosen = random.choice(
                best_actions
            )  # if best_actions else best_actions
            if demo_buffer is not None:
                demo_buffer.update(state, action_chosen, reward=reward)
            state, reward, done, info = env.step(action_chosen)
            total_reward += reward
            done = total_reward <= -episode_timeout if not done else done
        if verbose:
            print("final reward = ", total_reward)
    finally:
        if record_video:
            out.release()
        env.close()
    return total_reward
