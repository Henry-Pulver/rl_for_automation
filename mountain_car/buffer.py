import numpy as np
from typing import Tuple


class ExperienceBuffer:
    """
    Stores memory of past experience for model-free methods
    """

    def __init__(self):
        self.clear()

    def update(self, state, action, action_probs, reward: float) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.action_probs.append(action_probs)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []

    def get_length(self):
        return len(self.states)

    def recall_memory(self) -> Tuple[np.array]:
        states = np.array(self.states).reshape((-1, 2))
        actions = np.array(self.actions)
        action_probs = np.array(self.action_probs).reshape((-1, 3))
        return states, actions, action_probs

    def get_rewards(self) -> np.array:
        return np.array(self.rewards)
