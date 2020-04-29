import numpy as np
from typing import Tuple


class Grid:
    """
    Grid object. Defines possible moves from any position and rewards at each step.
    """

    WALL_POSITIONS = [
        (9, 3),
        (8, 3),
        (7, 3),
        (6, 3),
        (5, 3),
        (4, 3),
        (3, 3),
        (2, 3),
        (7, 6),
        (6, 6),
        (5, 6),
        (4, 6),
        (3, 6),
        (2, 6),
        (1, 6),
        (0, 6),
    ]

    ALLOWED_MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __init__(self):
        """Initialise agent start, finish, grid object

        """
        self._grid = np.array([np.zeros(10)] * 10)
        self._reward_grid = -np.array([np.ones(10)] * 10)
        self._reward_grid[0, 9] = 0

    def find_possible_actions(self, position: Tuple[int, int]) -> np.ndarray:
        possible_moves = []
        for move in self.ALLOWED_MOVES:
            new_position = get_new_state(position, move)
            if check_on_grid(new_position) and new_position not in self.WALL_POSITIONS:
                possible_moves.append(move)
        return np.array(possible_moves)

    def get_reward(self, new_position: Tuple[int, int]) -> int:
        return self._reward_grid[new_position]


class PlayableGrid(Grid):
    """Playable version of Grid object."""

    def __init__(self):
        super(PlayableGrid, self).__init__()
        self.agent_state = (9, 0)
        self.agent_reward = 0
        self.done = False

    def take_action(self, action: Tuple[int, int]):
        assert action in self.find_possible_actions(self.agent_state)
        self.agent_reward += self.get_reward(self.agent_state)
        self.agent_state = get_new_state(self.agent_state, action)
        self.done = self.agent_state == (0, 9)


def check_on_grid(position: Tuple[int, int]) -> bool:
    """Checks if a given position is on the grid."""
    return all([-1 < coord < 10 for coord in position])


def get_new_state(state: Tuple[int, int], action: Tuple[int, int]) -> Tuple[int, int]:
    return tuple([sum(x) for x in zip(state, action)])
