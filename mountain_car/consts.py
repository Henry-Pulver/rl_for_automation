import numpy as np


class CONSTS:
    GRAVITY = 0.0025
    FORCE = 0.001
    MIN_POSITION = -1.2
    MAX_POSITION = 0.6
    MAX_SPEED = 0.07
    GOAL_POSITION = 0.5
    STATE_SPACE_SIZE = 2


NUM_POSITIONS = 200
NUM_VELOCITIES = 150
POSITION_VALUES = np.linspace(
    CONSTS.MIN_POSITION, CONSTS.MAX_POSITION, NUM_POSITIONS
)  # 200 discrete positions
VELOCITY_VALUES = np.linspace(
    -CONSTS.MAX_SPEED, CONSTS.MAX_SPEED, NUM_VELOCITIES
)  # 100 discrete velocities


class DISC_CONSTS:
    ACTION_SPACE = np.array([x for x in range(3)])
    POSITION_SPACING = (POSITION_VALUES[1] - POSITION_VALUES[0]) / 2
    VELOCITY_SPACING = (VELOCITY_VALUES[1] - VELOCITY_VALUES[0]) / 2
    STATE_SPACE = np.array(
        [np.array([pos, vel]) for vel in VELOCITY_VALUES for pos in POSITION_VALUES]
    )


print(
    np.mean(
        [
            (180,),
            (96,),
            (176,),
            (153,),
            (184,),
            (156,),
            (181,),
            (155,),
            (204,),
            (118,),
            (119,),
            (158,),
            (149,),
            (173,),
            (95,),
        ]
    )
)
