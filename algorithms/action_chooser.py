from typing import Optional, List


class ActionChooser:
    def __init__(
        self,
        action_change_freq: Optional[int],
        action_choice: Optional[int],
        increase_freq: int,
        action_space: Optional[List] = None,
    ):
        self.timestep = 0
        self.reset_count = 0
        self.action_change_freq = action_change_freq
        self.action_choice = action_choice
        self.increase_freq = increase_freq
        self.action_space = action_space

    def step(self, action):
        action = self.action_space[action] if self.action_space is not None else action
        if self.action_change_freq is not None:
            if action == self.action_choice:
                self.timestep = 0
            elif self.timestep >= self.action_change_freq:
                self.timestep = 0
                action = self.action_choice
            else:
                self.timestep += 1
        return action

    def reset(self):
        if self.action_change_freq is not None:
            self.timestep = 0
            self.reset_count += 1
            if self.reset_count >= self.increase_freq:
                self.reset_count = 0
                self.action_change_freq += 1
