import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from algorithms.discrete_policy import DiscretePolicy
from algorithms.buffer import DemonstrationBuffer

from mountain_car_runner import test_solution
from consts import DISC_CONSTS


class BCTrainer:
    ACTION_SPACE = DISC_CONSTS.ACTION_SPACE

    def __init__(
        self,
        demo_path: Path,
        learning_rate: float,
        state_space_size: int,
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

        for epoch in range(num_epochs):
            print(f"Epoch number: {epoch + 1}")
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
            print(f"Avg loss: {sum_loss / num_steps}")


def train_network(save_location: Path, filename: str, num_demos: int, num_epochs: int):
    trainer = BCTrainer(
        demo_path=Path("../expert_demos/"),
        learning_rate=1e-5,
        action_space_size=3,
        state_space_size=2,
    )
    trainer.train_network(num_epochs=num_epochs, num_demos=num_demos)
    save_location.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.network.state_dict(), f"{save_location}/{filename}.pt")


def get_average_score(network_load: Path, episode_timeout: int, show_solution: bool):
    network = DiscretePolicy(
        action_space=DISC_CONSTS.ACTION_SPACE,
        state_dimension=DISC_CONSTS.STATE_SPACE.shape[-1],
        hidden_layers=(32, 32),
    ).float()
    # network = MountainCarNeuralNetwork().float()
    print(torch.load(network_load))
    network.load_state_dict(torch.load(network_load))
    network.eval()

    num_trials = 20
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
    print(f"rewards: {rewards}")
    print(f"Mean reward: {mean}")
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


def main():
    outer_means = []

    # num_demos = [18, 16, 14, 12]
    num_demos = [18]

    # num_epochs = int(1000 / num_demos)
    num_epochs = 50
    save_location = Path(f"BC/2020/01/15")

    # best_random_seeds = [1, 2, 4]
    max_random_seed = 5
    for demo in num_demos:
        means = []
        for random_seed in range(max_random_seed):
            torch.manual_seed(random_seed)
            filename = f"demos_{demo}_seed_{random_seed}"
            # filename = f"random_seed_{random_seed}"
            # filename = "network"
            # save_location = Path(f"BC/2019/12/22")
            network_load = Path(f"{save_location}/{filename}.pt")

            # train_network(save_location, filename, num_demos=demo, num_epochs=num_epochs)
            means.append(
                get_average_score(
                    network_load=network_load, episode_timeout=500, show_solution=False
                )
            )
        print(means)
        outer_means.append(means)
    print(outer_means)


if __name__ == "__main__":
    main()
