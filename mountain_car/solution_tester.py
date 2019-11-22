from mountain_car.REINFORCE_next_states import Policy
from mountain_car_runner import test_solution


def main():
    policy = Policy(alpha_policy=1e-2,
                    alpha_baseline=1e-2,
                    ref_num=1,
                    policy_load="REINFORCE_states/weights/human/policy_weights_0.npy",
                    )
    test_solution(policy.choose_action, record_video=False)


if __name__ == "__main__":
    # main()
