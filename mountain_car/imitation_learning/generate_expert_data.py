import numpy as np
from pathlib import Path
from typing import Callable

from value_iteration import pick_action, FINAL_SAVE_LOCATION
from algorithms.buffer import DemonstrationBuffer
from mountain_car_runner import test_solution


def record_demonstrations(
    policy: Callable,
    num_demos: int,
    save_path: Path,
    show_recording: bool,
    verbose: bool = False,
):
    demo_buffer = DemonstrationBuffer(
        save_path=save_path, state_dimension=2, action_space_size=3
    )
    for demo in range(num_demos):
        test_solution(
            pick_action=policy,
            record_video=False,
            demo_buffer=demo_buffer,
            show_solution=show_recording,
            verbose=verbose,
        )
        demo_buffer.save_demos(demo_number=demo)
        demo_buffer.clear()


def main():
    value_fn = np.load(f"../{FINAL_SAVE_LOCATION}")
    policy = lambda state: pick_action(state, value_fn)
    save_path = Path("expert_demos")
    record_demonstrations(
        policy=policy, num_demos=50, save_path=save_path, show_recording=False
    )


if __name__ == "__main__":
    main()
