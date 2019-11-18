from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT

env = gym_tetris.make("TetrisA-v0")
env = JoypadSpace(env, MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    # print(env.action_space)
    # print(env._action_map)
    # print(env._action_meanings)
    state, reward, done, info = env.step(env.action_space.sample())
    # print(state.shape)
    print(info)
    env.render()

env.close()
