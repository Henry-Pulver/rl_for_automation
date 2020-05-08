import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class ExperienceBuffer:
    """
    Stores memory of past experience for model-free methods
    """

    def __init__(
        self,
        state_dimension: Tuple[int],
        action_space_size: int,
        max_memory_size: Optional[int] = None,
    ):
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []
        self.state_dimension = state_dimension
        self.action_state_size = action_space_size
        self.max_size = 10000 if max_memory_size is None else max_memory_size

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
        self.action_probs = (
            [] if action_probs is None else self.action_probs + [action_probs]
        )

    def limit_size(self) -> None:
        """
        Removes all the oldest data if buffer longer than max size
        """
        self.states = self.states[-self.max_size :]
        self.actions = self.actions[-self.max_size :]
        if self.rewards:
            self.rewards = self.rewards[-self.max_size :]
        if self.action_probs:
            self.action_probs = self.action_probs[-self.max_size :]

    def clear(self) -> None:
        """Empties buffer and sets to lists"""
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.action_probs[:]

    def get_length(self) -> int:
        """Gives number of timesteps of episode of memory."""
        if type(self.states) == np.ndarray:
            return self.states.shape[0]
        else:
            return len(self.states)

    def recall_memory(self) -> Tuple:
        """Returns stored memory as numpy arrays."""
        states = np.array(self.states).reshape((-1, *self.state_dimension))
        actions = np.array(self.actions)
        action_probs = (
            np.array(self.action_probs).reshape((-1, self.action_state_size))
            if self.action_probs
            else None
        )
        return states, actions, action_probs

    def random_sample(self, batch_size: int) -> Tuple:
        """Sample batch_size states"""
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

    def get_rewards(self) -> np.array:
        """Returns rewards from memory."""
        return np.array(self.rewards)

    def to_numpy(self) -> None:
        if not type(self.states) == np.ndarray:
            self.states = np.array(self.states).reshape((-1, *self.state_dimension))
            self.actions = np.array(self.actions).astype(np.uint8)
            self.rewards = np.array(self.rewards) if self.rewards is not None else None
            self.action_probs = (
                np.array(self.action_probs) if self.action_probs is not None else None
            )

    def from_numpy(self) -> None:
        if type(self.states) == np.ndarray:
            self.states = list(self.states)
            self.actions = list(self.actions)
            self.rewards = list(self.rewards) if self.rewards is not None else None
            self.action_probs = (
                list(self.action_probs) if self.action_probs is not None else None
            )


class PPOExperienceBuffer(ExperienceBuffer):
    def __init__(
        self,
        state_dimension: Tuple[int],
        action_space_size: int,
        max_memory_size: Optional[int] = None,
    ):
        super(PPOExperienceBuffer, self).__init__(
            state_dimension, action_space_size, max_memory_size
        )
        self.log_probs = []
        self.is_terminal = []

    def update(
        self,
        state,
        action,
        action_probs=None,
        reward: Optional[float] = None,
        log_probs: Optional[Tuple] = None,
    ) -> None:
        super(PPOExperienceBuffer, self).update(state, action, action_probs, reward)
        assert log_probs is not None
        self.log_probs.append(log_probs)

    def clear(self) -> None:
        super(PPOExperienceBuffer, self).clear()
        del self.log_probs[:]
        del self.is_terminal[:]

    def recall_memory(self) -> Tuple:
        """Returns stored memory."""
        states, actions, action_probs = super(PPOExperienceBuffer, self).recall_memory()
        return (
            states,
            actions,
            action_probs,
            np.array(self.log_probs),
            np.array(self.is_terminal),
        )


class GAILExperienceBuffer(PPOExperienceBuffer):
    def __init__(
        self,
        state_dimension: Tuple[int],
        action_space_size: int,
        max_memory_size: Optional[int] = None,
    ):
        super(GAILExperienceBuffer, self).__init__(
            state_dimension, action_space_size, max_memory_size
        )
        self.discrim_labels = []
        self.state_actions = []


class DemonstrationBuffer(ExperienceBuffer):
    """
    Class that samples, loads and saves demonstrations.
    """

    def __init__(
        self,
        save_path: Path,
        state_dimension: Tuple[int],
        action_space_size: int,
        max_memory_size: Optional[int] = None,
    ):
        super(DemonstrationBuffer, self).__init__(
            state_dimension, action_space_size, max_memory_size
        )
        self.save_path: Path = save_path

    def save_demos(self, demo_number: int) -> None:
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

    def load_demo(self, demo_number: int) -> None:
        """Loads expert demonstrations data for training."""
        self.actions += list(np.load(f"{self.save_path}/{demo_number}/actions.npy"))
        self.states += list(np.load(f"{self.save_path}/{demo_number}/states.npy"))
        try:
            self.rewards += list(np.load(f"{self.save_path}/{demo_number}/rewards.npy"))
        except Exception:
            pass

    def recall_expert_data(self, num_samples: int):
        states = np.array(self.states[:num_samples]).reshape(
            (-1, *self.state_dimension)
        )
        actions = np.array(self.actions[:num_samples])
        del self.states[:num_samples]
        del self.actions[:num_samples]
        return states, actions


class PlayBuffer(DemonstrationBuffer):
    """
    Class for recording demonstrations by human play.

    Args:
        save_path: Location to save demonstrations
    """

    def __init__(
        self,
        save_path: Path,
        state_dimension: Tuple[int],
        action_space_size: int,
        max_memory_size: Optional[int] = None,
    ):
        super(PlayBuffer, self).__init__(
            save_path, state_dimension, action_space_size, max_memory_size
        )
        # self.frame = 0
        # self.reward_count = 0

    def update_play(self, prev_obs, obs, action, rew, env_done, info,) -> None:
        """
        Updates play buffer with most recently observed states, actions,
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

    # def clear(self):
    #     super(PlayBuffer, self).clear()
    # self.frame = 0
    # self.reward_count = 0


class PrioritisedBuffer(DemonstrationBuffer):
    """
    Demonstration buffer that allows prioritised sampling of transitions.
    """

    def __init__(
        self,
        save_path: Path,
        state_dimension: Tuple[int],
        action_space_size: int,
        max_memory_size: Optional[int] = None,
    ):
        super(PrioritisedBuffer, self).__init__(
            save_path, state_dimension, action_space_size, max_memory_size
        )

    def prioritised_sample(self, minibatch_size: int) -> Tuple:
        minibatch_size = np.min([minibatch_size, self.get_length()])
        sample_refs = np.random.randint(
            low=0, high=self.get_length(), size=minibatch_size
        )
