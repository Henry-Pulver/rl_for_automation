import numpy as np
from pathlib import Path

from algorithms.imitation_learning.GAIL import sample_from_buffers
from algorithms.buffer import ExperienceBuffer, DemonstrationBuffer


def test_sample_from_buffers():
    """Tests whether sampling function from buffers works."""
    # Testing params
    state_dim = (2,)
    action_space_size = 3
    mountain_car_demo_path = Path("../mountain_car/imitation/expert_demos")
    demo_num = 0
    batch_size = 8
    experiences = np.array([[-1, 0, 0, 0, 1], [1, 0.002, 0, 1, 0], [-1, 0.005, 1, 0, 0]])

    exp_buffer = ExperienceBuffer(state_dim, action_space_size)
    demo_buffer = DemonstrationBuffer(mountain_car_demo_path, state_dim, action_space_size)
    demo_buffer.load_demos(demo_num)
    for experience in experiences:
        exp_buffer.update(state=experience[:2], action=np.where(experience[2:]==1)[0][0])
    demo_buffer_states = demo_buffer.states

    data, labels = sample_from_buffers(demo_buffer, exp_buffer, batch_size, action_space_size)
    for data_member in data:
        # Checks if data in set from experience with the environment
        data_in_experiences = False
        for experience in experiences:
            data_in_experiences = np.all(data_member == experience) if not data_in_experiences else data_in_experiences

        # Checks if data in expert set
        data_in_expert = False
        for expert_state in demo_buffer_states:
            data_in_expert = np.all(data_member[:2] == expert_state) if not data_in_expert else data_in_expert

        # Check data is either from expert or experience
        assert data_in_experiences or data_in_expert

    # Check labels
    assert np.all(labels == np.array([1] * (batch_size - len(experiences)) + [0] * len(experiences)))
