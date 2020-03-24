GAME_NAMES = [
    "breakout",
    "ms_pacman",
    "pong",
    "space_invaders",
    "beam_rider",
]
# v0 has 'repeat_action_probability' of 0.25 which corresponds to p(prev action is used)
# v4 has 'repeat_action_probability' of 0
# Deterministic has a fixed frameskip of 4, compared to being sampled from (2, 5)
# NoFrameskip has no frameskip at all
GAME_STRINGS_PLAY = [
    "Breakout-ramNoFrameskip-v4",
    "MsPacman-ramNoFrameskip-v4",
    "Pong-ramNoFrameskip-v4",
    "SpaceInvaders-ramNoFrameskip-v4",
    "BeamRider-ramNoFrameskip-v4",
]
GAME_STRINGS_LEARN = [  # So for learning we're reduced the number of frames by
    "Breakout-ram-v4",
    "MsPacman-ram-v4",
    "Pong-ram-v4",
    "SpaceInvaders-ram-v4",
    "BeamRider-ram-v4",
]
GAME_STRINGS_TEST = [
    "Breakout-ramNoFrameskip-v4",
    "MsPacman-ramNoFrameskip-v4",
    "Pong-ramNoFrameskip-v4",
    "SpaceInvaders-ramNoFrameskip-v4",
    "BeamRider-ramNoFrameskip-v4",
]
SOLVED_SCORES = [
    300,
    4000,
    20,
    1500,
    4000,
]
