import numpy as np
from mountain_car.REINFORCE_actions import Policy as ActionPolicy
from mountain_car.REINFORCE_next_states import Policy as StatePolicy
from mountain_car_runner import test_solution


def make_hand_crafted_policy(policy_save: str):
    policy = np.array(
        [
            0,
            -100,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -10,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            100,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    np.save(policy_save, policy)


def main():
    model_known = False

    algorithm_string = "states" if model_known else "actions"
    policy_save = f"REINFORCE_{algorithm_string}/weights/human/policy_weights_0.npy"
    Policy = StatePolicy if model_known else ActionPolicy

    policy = Policy(
        alpha_policy=1e-2, alpha_baseline=1e-2, ref_num=1, policy_load=policy_save,
    )
    test_solution(policy.choose_action, record_video=False)


if __name__ == "__main__":
    main()
