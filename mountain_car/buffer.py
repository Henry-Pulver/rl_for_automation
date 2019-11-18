import numpy as np


class ExperienceBuffer:
    """
    Stores memory of past experience for model-free methods
    """

    def __init__(self):
        self.clear()

    def update(self, state, action, action_probs, reward: float) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards = np.append(self.rewards, reward)
        self.action_probs.append(action_probs)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = np.array([])
        self.action_probs = []

    def get_length(self):
        return len(self.states)
