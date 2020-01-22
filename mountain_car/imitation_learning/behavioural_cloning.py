import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from algorithms.discrete_policy import DiscretePolicy
from algorithms.buffer import DemonstrationBuffer
from typing import List, Tuple
import logging

from mountain_car_runner import test_solution
from consts import DISC_CONSTS


class BCTrainer:
    ACTION_SPACE = DISC_CONSTS.ACTION_SPACE

    def __init__(
        self,
        demo_path: Path,
        learning_rate: float,
        state_space_size: Tuple[int],
        action_space_size: int,
    ):
        self.network = DiscretePolicy(
            state_dimension=2,
            action_space=DISC_CONSTS.ACTION_SPACE,
            hidden_layers=(32, 32),
        ).float()
        # self.network = MountainCarNeuralNetwork.float()

        self.demo_buffer = DemonstrationBuffer(
            demo_path, state_space_size, action_space_size
        )

        self.criterion = nn.CrossEntropyLoss()

        # zero the parameter gradients
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()

        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
        #                                           shuffle=True, num_workers=0)

    def train_network(self, num_epochs: int, num_demos: int):
        state_array = np.array([])
        action_array = np.array([])
        state_array = state_array.reshape((-1, 2))
        for demo_num in range(num_demos):
            self.demo_buffer.load_demos(demo_num)
            states, actions, _ = self.demo_buffer.recall_memory()
            state_array = np.append(state_array, states)
            action_array = np.append(action_array, actions)
        state_array = state_array.reshape((-1, 2))
        states = torch.from_numpy(state_array)
        actions = torch.from_numpy(action_array)
        prev_avg_loss = []
        for epoch in range(num_epochs):
            sum_loss, num_steps = 0, 0
            for state, action in zip(states, actions):
                # forward + backward + optimize
                action_probs = self.network(state.float()).unsqueeze(dim=0)
                action_probs = action_probs.float()
                action = action.unsqueeze(dim=0)
                action = action.type(torch.long)

                loss = self.criterion(action_probs, action)
                loss.backward()
                self.optimizer.step()
                num_steps += 1
                sum_loss += loss
            avg_loss = sum_loss / num_steps
            logging.debug(f"Epoch number: {epoch + 1}\tAvg loss: {avg_loss}")
            prev_avg_loss.append(avg_loss)
            if len(prev_avg_loss) > 10:
                if avg_loss == prev_avg_loss[0]:
                    logging.info(
                        f"\n\nAvg loss hasn't changed for 10 iterations.\n Skipping to next seed.\n"
                    )
                    break
                prev_avg_loss.pop()


def train_network(
    save_location: Path, filename: str, demo_path: Path, num_demos: int, num_epochs: int
):
    trainer = BCTrainer(
        demo_path=demo_path,
        learning_rate=1e-5,
        action_space_size=3,
        state_space_size=(2,),
    )
    trainer.train_network(num_epochs=num_epochs, num_demos=num_demos)
    save_location.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.network.state_dict(), f"{save_location}/{filename}.pt")


def get_average_score(
    network_load: Path, episode_timeout: int, show_solution: bool, num_trials: int
):
    network = DiscretePolicy(
        action_space=DISC_CONSTS.ACTION_SPACE,
        state_dimension=DISC_CONSTS.STATE_SPACE.shape[-1],
        hidden_layers=(32, 32),
    ).float()
    # network = MountainCarNeuralNetwork().float()
    # print(torch.load(network_load))
    network.load_state_dict(torch.load(network_load))
    network.eval()

    rewards = []
    for _ in range(num_trials):
        rewards.append(
            test_solution(
                lambda x: pick_action(state=x, network=network),
                record_video=False,
                show_solution=show_solution,
                episode_timeout=episode_timeout,
            )
        )
        mean = np.mean(rewards)
    logging.info(f"\nRewards: {rewards}")
    logging.info(f"\nMean reward: {mean}")
    return mean


def action_probs(state, network):
    state = torch.from_numpy(state)
    return network(state.float()).detach().numpy()


def pick_action(state, network):
    state = torch.from_numpy(state)
    action_probs = network(state.float())
    action_probs = action_probs.detach().numpy()
    chosen_action = np.array(
        [np.random.choice(DISC_CONSTS.ACTION_SPACE, p=action_probs)]
    )
    return chosen_action


def pretty_print(outcomes: List, inputs: List, epochs: List, num_random_seeds: int):
    assert len(outcomes) == (len(inputs) * num_random_seeds)
    for input_count, (input, epoch) in enumerate(zip(inputs, epochs)):
        # logging.debug(f"Number of demos used: {input}\t trained for {epoch}")
        print(f"Number of demos used: {input}\t trained for {epoch}")
        for seed in range(num_random_seeds):
            # logging.debug(
            #     f"Seed: {seed}\t Score: {outcomes[input_count*num_random_seeds + seed]}"
            # )
            print(
                f"Seed: {seed}\t Score: {outcomes[input_count * num_random_seeds + seed]}"
            )
        # logging.debug(" ")
        print(" ")


def main():
    num_demos = [1, 2, 3, 4, 6, 7, 8, 9, 10, 15]
    num_epochs = list(1000 // np.array(num_demos))

    save_location = Path(f"BC/2020/01/21")
    save_location.mkdir(parents=True, exist_ok=True)
    demo_path = Path("expert_demos/")

    # demo_buffer = DemonstrationBuffer(demo_path, (2,), 3)
    # steps = []
    # for demo_num in range(9):
    #     demo_buffer.load_demos(demo_num)
    #     rewards = demo_buffer.get_rewards()
    #     steps_to_completion = np.sum(rewards)
    #     steps.append(steps_to_completion)
    # print(steps)
    # print(np.mean(steps))

    logging.basicConfig(filename=f"{save_location}/BC3.log", level=logging.INFO)
    outer_means = []
    max_random_seed = 10
    for demo, epoch_num in zip(num_demos, num_epochs):
        means = []
        for random_seed in range(max_random_seed):
            torch.manual_seed(random_seed)
            filename = f"demos_{demo}_seed_{random_seed}"
            # filename = f"random_seed_{random_seed}"
            # filename = "network"
            # save_location = Path(f"BC/2019/12/22")
            network_load = Path(f"{save_location}/{filename}.pt")

            # train_network(
            #     save_location,
            #     filename,
            #     num_demos=demo,
            #     num_epochs=epoch_num,
            #     demo_path=demo_path,
            # )
            means.append(
                get_average_score(
                    network_load=network_load,
                    episode_timeout=500,
                    show_solution=False,
                    num_trials=1000,
                )
            )
        print(means)
        outer_means.append(means)
    print(outer_means)
    pretty_print(
        outcomes=outer_means,
        inputs=num_demos,
        num_random_seeds=max_random_seed,
        epochs=num_epochs,
    )


if __name__ == "__main__":
    main()
    # pretty_print(outcomes=[2, 3, 5], inputs=[1, 2, 3], num_random_seeds=1, epochs=[50, 13, 1])

# RESULTS
#  1: [[-500.0, -500.0,  -500.0,  -499.01, -500.0,  -500.0,  -500.0,  -496.43, -500.0,  -499.98],
#  2: [-496.79, -499.99, -129.25, -179.86, -468.09, -115.79, -119.85, -499.44, -500.0,  -499.79],
#  3: [-497.04, -497.77, -137.19, -182.31, -499.96, -122.56, -129.09, -498.39, -499.88, -498.64],
#  4: [-499.77, -497.45, -130.62, -202.03, -500.0,  -147.1,  -126.01, -498.58, -496.38, -498.2],
#  5: [-498.26, -499.74, -122.83, -254.43, -499.96, -148.99, -124.22, -497.78, -500.0,  -498.43],
#  6: [-500.0,  -499.38, -124.66, -279.18, -500.0,  -122.99, -121.79, -498.4,  -499.43, -497.21],
#  7: [-391.37, -500.0,  -124.17, -497.33, -169.99, -499.72, -121.49, -497.41, -496.62, -496.93],
#  8: [-499.78, -500.0,  -125.78, -495.04, -289.38, -499.15, -129.46, -494.69, -499.16, -394.23],
#  9: [-350.89, -491.87, -119.76, -138.77, -491.11, -499.0,  -110.09, -500.0,  -497.62, -149.71],
# 10: [-492.53, -493.99, -128.42, -147.95, -493.09, -498.02, -117.23, -498.22, -499.57, -198.25],
# 15: [-176.29, -498.55, -139.56, -263.85, -496.18, -497.29, -128.75, -497.8,  -499.05, -499.58]]

# target = -119.88888888888889
# t = np.array([[-500.0, -500.0,  -500.0,  -499.01, -500.0,  -500.0,  -500.0,  -496.43, -500.0,  -499.98],
# [-496.79, -499.99, -129.25, -179.86, -468.09, -115.79, -119.85, -499.44, -500.0,  -499.79],
# [-497.04, -497.77, -137.19, -182.31, -499.96, -122.56, -129.09, -498.39, -499.88, -498.64],
# [-499.77, -497.45, -130.62, -202.03, -500.0,  -147.1,  -126.01, -498.58, -496.38, -498.2],
# [-498.26, -499.74, -122.83, -254.43, -499.96, -148.99, -124.22, -497.78, -500.0,  -498.43],
# [-500.0,  -499.38, -124.66, -279.18, -500.0,  -122.99, -121.79, -498.4,  -499.43, -497.21],
# [-391.37, -500.0,  -124.17, -497.33, -169.99, -499.72, -121.49, -497.41, -496.62, -496.93],
# [-499.78, -500.0,  -125.78, -495.04, -289.38, -499.15, -129.46, -494.69, -499.16, -394.23],
# [-350.89, -491.87, -119.76, -138.77, -491.11, -499.0,  -110.09, -500.0,  -497.62, -149.71],
# [-492.53, -493.99, -128.42, -147.95, -493.09, -498.02, -117.23, -498.22, -499.57, -198.25],
# [-176.29, -498.55, -139.56, -263.85, -496.18, -497.29, -128.75, -497.8,  -499.05, -499.58]])
#
# print(np.where(t > target))

# NEW VALUES:
# [[-500.0, -500.0, -500.0, -498.349, -500.0, -499.568, -500.0, -498.916, -500.0, -499.951], [-497.113, -498.794, -129.724, -184.314, -475.205, -115.41, -118.665, -498.451, -498.282, -499.161], [-498.277, -498.471, -139.36, -178.059, -499.165, -124.099, -128.698, -498.607, -498.656, -498.441], [-497.992, -499.091, -133.662, -207.334, -499.616, -147.927, -130.403, -498.657, -498.618, -499.131], [-500.0, -498.373, -121.692, -283.368, -500.0, -127.91, -123.152, -498.79, -498.86, -497.61], [-372.856, -498.607, -126.855, -497.247, -168.191, -496.623, -123.635, -497.854, -499.132, -496.309], [-498.67, -498.421, -126.404, -494.68, -281.145, -496.844, -123.985, -498.339, -499.539, -392.742], [-341.564, -493.636, -123.772, -142.301, -495.49, -498.041, -111.021, -498.371, -498.625, -148.72], [-496.18, -494.726, -126.429, -152.663, -495.717, -497.08, -111.678, -498.699, -498.797, -200.056], [-174.2, -498.06, -134.843, -263.611, -497.189, -497.805, -131.471, -498.375, -498.046, -498.913]]
