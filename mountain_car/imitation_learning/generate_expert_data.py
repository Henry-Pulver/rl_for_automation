import numpy as np
from pathlib import Path
from typing import Callable

from value_iteration import pick_action, FINAL_SAVE_LOCATION
from buffer import DemonstrationBuffer
from mountain_car_runner import test_solution


def record_demonstrations(policy: Callable, num_demos: int, save_path: Path):
    demo_buffer = DemonstrationBuffer(save_path=save_path)
    for demo in range(num_demos):
        test_solution(pick_action=policy, record_video=False, demo_buffer=demo_buffer)
        demo_buffer.save_demos(demo_number=demo)
        demo_buffer.clear()


def main():
    value_fn = np.load(FINAL_SAVE_LOCATION)
    policy = lambda state: pick_action(state, value_fn)
    save_path = Path("expert_demos")
    record_demonstrations(policy=policy, num_demos=20, save_path=save_path)


if __name__ == "__main__":
    main()
