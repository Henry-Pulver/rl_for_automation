from mountain_car.REINFORCE_actions import Policy
from mountain_car_runner import test_solution


def main():
    policy = Policy(alpha_policy=1e-2,
                    alpha_baseline=1e-2,
                    ref_num=1,
                    baseline_load="REINFORCE_actions/weights/2001/baseline_weights_2001.npy",
                    policy_load="REINFORCE_actions/weights/2001/policy_weights_2001.npy",
                    )
    test_solution(policy.choose_action, record_video=False)


if __name__ == "__main__":
    main()
