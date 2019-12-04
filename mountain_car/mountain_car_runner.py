import numpy as np
import random
import gym
import cv2
from typing import Callable, Any, Optional

from buffer import DemonstrationBuffer


class CONSTS:
    GRAVITY = 0.0025
    FORCE = 0.001
    MIN_POSITION = -1.2
    MAX_POSITION = 0.6
    MAX_SPEED = 0.07
    GOAL_POSITION = 0.5
    STATE_SPACE_SIZE = 2


POSITION_VALUES = np.linspace(
    CONSTS.MIN_POSITION, CONSTS.MAX_POSITION, 200
)  # 200 discrete positions
VELOCITY_VALUES = np.linspace(
    -CONSTS.MAX_SPEED, CONSTS.MAX_SPEED, 140
)  # 100 discrete velocities


class DISC_CONSTS:
    ACTION_SPACE = np.array([x for x in range(3)])
    POSITION_SPACING = (POSITION_VALUES[1] - POSITION_VALUES[0]) / 2
    VELOCITY_SPACING = (VELOCITY_VALUES[1] - VELOCITY_VALUES[0]) / 2
    STATE_SPACE = np.array(
        [np.array([pos, vel]) for vel in VELOCITY_VALUES for pos in POSITION_VALUES]
    )


def test_solution(pick_action: Callable, record_video: bool, demo_buffer: Optional[DemonstrationBuffer] = None, *args: Any) -> None:
    env = gym.make("MountainCar-v0").env
    render_type = "rgb_array" if record_video else "human"

    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_fps = 60
        out = cv2.VideoWriter('output.avi', fourcc, video_fps, (600, 400))

    state = env.reset()
    done = False
    total_reward = 0
    try:
        while not done:
            rgb_array = env.render(render_type)
            if record_video:
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                out.write(bgr_array)
            best_actions = pick_action(state, *args)
            action_chosen = random.choice(
                best_actions
            )  # if best_actions else best_actions
            if demo_buffer is not None:
                demo_buffer.update(state, action_chosen)
            state, reward, done, info = env.step(action_chosen)
            total_reward += reward
        print("final reward = ", total_reward)
        print("final state = ", state)
    finally:
        if record_video:
            out.release()
        env.close()
