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
        self.rewards = None if reward is None else self.rewards + reward
        self.action_probs = (
            None if action_probs is None else self.action_probs + action_probs
        )

    def clear(self):
        """Empties buffer and sets to lists"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []

    def get_length(self):
        """Gives number of timesteps of episode of memory."""
        if type(self.states) == np.ndarray:
            return self.states.shape[0]
        else:
            print(f"len: {len(self.states)}")
            return len(self.states)

    def recall_memory(self) -> Tuple:
        """Returns stored memory."""
        states = np.array(self.states).reshape((-1, *self.state_dimension))
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

    def to_numpy(self):
        self.states = (
            np.array(self.states).reshape((-1, *self.state_dimension)).astype(np.uint8)
        )
        self.actions = np.array(self.actions).astype(np.uint8)
        self.rewards = (
            np.array(self.rewards).astype(np.uint8)
            if self.rewards is not None
            else None
        )
        self.action_probs = (
            np.array(self.action_probs) if self.action_probs is not None else None
        )

    def from_numpy(self):
        self.states = list(self.states)
        self.actions = list(self.actions)
        self.rewards = list(self.rewards) if self.rewards is not None else None
        self.action_probs = (
            list(self.action_probs) if self.action_probs is not None else None
        )


class DemonstrationBuffer(ExperienceBuffer):
    def __init__(
        self, save_path: Path, state_dimension: Tuple[int], action_space_size: int
    ):
        super(DemonstrationBuffer, self).__init__(state_dimension, action_space_size)
        self.save_path: Path = save_path

    def save_demos(self, demo_number: int):
        """Saves data during expert demonstrations."""
        demo_path = self.save_path / f"{demo_number}"
        demo_path.mkdir(parents=True, exist_ok=True)
        # Convert lists to numpy arrays
        self.to_numpy()
        np.save(f"{demo_path}/actions.npy", self.actions)
        np.save(f"{demo_path}/states.npy", self.states)
        if self.rewards is not None:
            np.save(f"{demo_path}/rewards.npy", self.rewards)
        # Back to lists for other purposes
        self.from_numpy()

    def load_demos(self, demo_number: int):
        """Loads expert demonstrations data for training."""
        self.actions += list(np.load(f"{self.save_path}/{demo_number}/actions.npy"))
        self.states += list(np.load(f"{self.save_path}/{demo_number}/states.npy"))
        try:
            self.rewards += list(np.load(f"{self.save_path}/{demo_number}/rewards.npy"))
        except Exception:
            pass

    def sample(self, batch_size: int) -> Tuple:
        minibatch_size = np.min([batch_size, self.get_length()])
        sample_refs = np.random.randint(
            low=0, high=self.get_length(), size=minibatch_size
        )
        sampled_actions = self.actions[sample_refs]
        self.actions = np.delete(self.actions, sample_refs, axis=0)
        sampled_states = self.states[sample_refs]
        self.states = np.delete(self.states, sample_refs, axis=0)
        sampled_rewards = (
            self.rewards[sample_refs] if not self.rewards is None else None
        )
        self.rewards = (
            np.delete(self.rewards, sample_refs, axis=0)
            if not self.rewards is None
            else None
        )
        return sampled_states, sampled_actions, sampled_rewards


class PlayBuffer(DemonstrationBuffer):
    def __init__(
        self, save_path: Path, state_dimension: Tuple[int], action_space_size: int
    ):
        super(PlayBuffer, self).__init__(save_path, state_dimension, action_space_size)
        # self.frame = 0
        # self.reward_count = 0

    def update_play(self, prev_obs, obs, action, rew, env_done, info,) -> None:
        """
        Updates buffer with most recently observed states, actions,
        probabilities and rewards
        """
        # if (self.frame % 4) == 0:
        self.states.append(prev_obs)
        self.actions.append(action)
        self.rewards.append(rew)
        # if rew is not None:
        # self.rewards.append(self.reward_count)
        # self.reward_count = 0
        # else:
        #     if rew is not None:
        #         self.reward_count += rew
        # self.frame += 1

    def clear(self):
        super(PlayBuffer, self).clear()
        # self.frame = 0
        # self.reward_count = 0