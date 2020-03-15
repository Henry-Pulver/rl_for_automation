from typing import Optional


class ActionChooser:
    def __init__(self, action_change_freq: Optional[int], action_choice: Optional[int]):
        self.timestep = 0
        self.action_change_freq = action_change_freq
        self.action_choice = action_choice

    def step(self, action):
        if self.action_change_freq is not None:
            if action == self.action_choice:
                self.timestep = 0
            elif self.timestep == self.action_change_freq:
                self.timestep = 0
                action = self.action_choice
            else:
                self.timestep += 1
        return action

    def reset(self):
        self.timestep = 0
