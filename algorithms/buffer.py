import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class ExperienceBuffer:
    """
    Stores memory of past experience for model-free methods
    """

    def __init__(self, state_dimension: Tuple[int], action_space_size: int):
        self.clear()
        self.state_dimension = state_dimension
        self.action_state_size = action_space_size

    def update(
        self, state, action, action_probs=None, reward: Optional[float] = None
    ) -> None:
        """
        Updates buffer with most recently observed states, actions,
        probabilities and rewards
        """
        self.states.append(state)
        self.actions.append(action)
        if reward is not None:
            self.rewards.append(reward)
        if action_probs is not None:
            self.action_probs.append(action_probs)

    def clear(self):
        """Empties buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []

    def get_length(self):
        """Gives number of timesteps of episode of memory."""
        return len(self.states)

    def recall_memory(self) -> Tuple:
        """Returns stored memory."""
        states = np.array(self.states).reshape((-1, self.state_dimension))
        actions = np.array(self.actions)
        action_probs = (
            np.array(self.action_probs).reshape((-1, self.action_state_size))
            if self.action_probs
            else None
        )
        return states, actions, action_probs

    def get_rewards(self) -> np.array:
        """Returns rewards from memory."""
        return np.array(self.rewards)


class DemonstrationBuffer(ExperienceBuffer):
    def __init__(self, save_path: Path, state_dimension: int, action_space_size: int):
        super(DemonstrationBuffer, self).__init__(state_dimension, action_space_size)
        self.save_path: Path = save_path

    def save_demos(self, demo_number: int):
        """Saves data during expert demonstrations."""
        demo_path = self.save_path / f"{demo_number}"
        demo_path.mkdir(parents=True, exist_ok=True)
        np.save(f"{demo_path}/actions.npy", np.array(self.actions))
        np.save(
            f"{demo_path}/states.npy",
            np.array(self.states).reshape((-1, *self.state_dimension)),
        )
        if self.rewards:
            np.save(f"{demo_path}/rewards.npy", np.array(self.rewards))

    def load_demos(self, demo_number: int):
        """Loads expert demonstrations data for training."""
        self.actions = np.load(f"{self.save_path}/{demo_number}/actions.npy")
        self.states = np.load(f"{self.save_path}/{demo_number}/states.npy")
        try:
            self.rewards = np.load(f"{self.save_path}/{demo_number}/rewards.npy")
        except Exception:
            pass


class PlayBuffer(DemonstrationBuffer):
    def __init__(self, save_path: Path, state_dimension: int, action_space_size: int):
        super(PlayBuffer, self).__init__(save_path, state_dimension, action_space_size)

    def update_play(self, prev_obs, obs, action, rew, env_done, info,) -> None:
        """
        Updates buffer with most recently observed states, actions,
        probabilities and rewards
        """
        self.states.append(prev_obs)
        self.actions.append(action)
        if rew is not None:
            self.rewards.append(rew)
