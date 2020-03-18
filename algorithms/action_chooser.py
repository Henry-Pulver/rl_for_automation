from typing import Optional


class ActionChooser:
    def __init__(
        self,
        action_change_freq: Optional[int],
        action_choice: Optional[int],
        increase_freq: int,
    ):
        self.timestep = 0
        self.reset_count = 0
        self.action_change_freq = action_change_freq
        self.action_choice = action_choice
        self.increase_freq = increase_freq

    def step(self, action):
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
