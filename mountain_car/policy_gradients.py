import random
import numpy as np

ALPHA_BASELINE = 0.02
ALPHA_POLICY = 0.01


def convert_to_feature(observation: np.array) -> np.array:
    assert observation.shape == [2]
    p, v = observation
    return np.array([1,
                     p,
                     v,
                     p * v,
                     p ** 2,
                     v ** 2,
                     p * (v ** 2),
                     v * (p ** 2),
                     (p ** 2) * (v ** 2)])


def main():
    baseline_weights = np.random.normal(size=9)
    policy_weights = np.random.normal(size=27)


    policy_parameters
    states, actions, run_episode(policy)


if __name__ == "__main__":
    main()