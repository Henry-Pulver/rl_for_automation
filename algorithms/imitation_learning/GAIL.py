from pathlib import Path
from neural_network import MountainCarNeuralNetwork


class GAILTrainer:
    def __init__(self, demo_path: Path, learning_rate: float):
        self.network = MountainCarNeuralNetwork().float()

        # self.demo_buffer = DemonstrationBuffer(demo_path)
        #
        # self.criterion = nn.CrossEntropyLoss()
        #
        # # zero the parameter gradients
        # self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        # self.optimizer.zero_grad()
        #
        # # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
        # #                                           shuffle=True, num_workers=0)

    # def train_network(self, num_epochs: int, num_demos: int):

    # state_array = np.array([])
    # action_array = np.array([])
    # state_array = state_array.reshape((-1, 2))
    # for demo_num in range(num_demos):
    #     self.demo_buffer.load_demos(demo_num)
    #     states, actions, _ = self.demo_buffer.recall_memory()
    #     state_array = np.append(state_array, states)
    #     action_array = np.append(action_array, actions)
    # state_array = state_array.reshape((-1, 2))
    # states = torch.from_numpy(state_array)
    # actions = torch.from_numpy(action_array)
    #
    # for epoch in range(num_epochs):
    #     print(f"Epoch number: {epoch + 1}")
    #     sum_loss, num_steps = 0, 0
    #     for state, action in zip(states, actions):
    #         # forward + backward + optimize
    #         action_probs = self.network(state.float()).unsqueeze(dim=0)
    #         action_probs = action_probs.float()
    #         action = action.unsqueeze(dim=0)
    #         action = action.type(torch.long)
    #
    #         loss = self.criterion(action_probs, action)
    #         loss.backward()
    #         self.optimizer.step()
    #         num_steps += 1
    #         sum_loss += loss
    #     print(f"Avg loss: {sum_loss / num_steps}")


# def train_network(save_location: Path, filename: str):
# trainer = BCTrainer(demo_path=Path("../expert_demos/"),
#                     learning_rate=1e-5,
#                     )
# trainer.train_network(num_epochs=50, num_demos=20)
# save_location.mkdir(parents=True, exist_ok=True)
# torch.save(trainer.network.state_dict(), f"{save_location}/{filename}.pt")


# def show_solution(network_load: Path):
# network = MountainCarNeuralNetwork().float()
# network.load_state_dict(torch.load(network_load))
# network.eval()
#
# for i in range(10):
#     test_solution(lambda x: pick_action(state=x, network=network), record_video=False)


# def pick_action(state, network):
# state = torch.from_numpy(state)
# action_probs = network(state.float())
# action_probs = action_probs.detach().numpy()
# chosen_action = np.array([np.random.choice(DISC_CONSTS.ACTION_SPACE, p=action_probs)])
# return chosen_action


def main():
    pass
    # filename = "network50"
    # save_location = Path(f"BC/2019/12/04")
    # network_load = Path(f"{save_location}/{filename}.pt")
    #
    # # train_network(save_location, filename)
    # show_solution(network_load=network_load)


if __name__ == "__main__":
    main()
