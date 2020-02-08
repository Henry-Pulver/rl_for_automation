from gym.utils.play import play
import gym
from algorithms.buffer import PlayBuffer
from pathlib import Path
from atari.consts import GAME_STRINGS_PLAY, GAME_NAMES


def generate_expert_data(game_ref: int, fps: int, num_demos: int, demo_start: int):
    save_path = Path(f"expert_demos/{GAME_NAMES[game_ref]}")
    env = gym.make(GAME_STRINGS_PLAY[game_ref])
    play_buffer = PlayBuffer(
        save_path,
        state_dimension=env.observation_space.shape,
        action_space_size=env.action_space.n,
    )
    for demo in range(num_demos)[demo_start:]:
        play(env, callback=play_buffer.update_play, zoom=5, fps=fps)
        play_buffer.save_demos(demo_number=demo)
        play_buffer.clear()


def main():
    game_ref = 2
    num_demos = 50
    fps = 60
    demo_start = 1
    generate_expert_data(game_ref, fps, num_demos, demo_start)


if __name__ == "__main__":
    main()
