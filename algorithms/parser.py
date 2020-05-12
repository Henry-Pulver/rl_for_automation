from argparse import ArgumentParser


def get_actor_critic_parser(*args, **kwargs):
    parser = get_base_parser(*args, **kwargs)
    parser.add_argument(
        "--num_shared_layers",
        type=int,
        help="number of shared layers in the neural network",
        default=3,
    )
    parser.add_argument(
        "--adv_type",
        type=str,
        help="advantage estimation type",
        default="gae",
        choices=["monte_carlo", "monte_carlo_baseline", "gae"],
    )
    parser.add_argument(
        "--lamda", type=float, help="GAE lambda value", default=0.95,
    )
    return parser


def get_base_parser(*args, **kwargs):
    parser = ArgumentParser(*args, **kwargs)
    parser.add_argument(
        "--save_base_path", type=str, help="base path to save plots", default="data"
    )
    parser.add_argument("--lr", type=float, help="learning rate used")
    parser.add_argument(
        "--param_plot_num", type=int, help="number of params to plot", default=10
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        help="number of steps between logging score",
        default=20,
    )
    parser.add_argument(
        "--num_seeds", type=int, help="num of random seeds used", default=5
    )
    parser.add_argument(
        "--max_episodes", type=int, help="the maximum number of episodes run"
    )
    parser.add_argument(
        "--max_timesteps", type=int, help="the maximum number of episodes run"
    )
    parser.add_argument("--env_name", type=str, help="name of env")
    parser.add_argument(
        "--solved_reward",
        type=float,
        help="avg reward corresponding to env being solved",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        help="number of neural network hidden layers",
        default=4,
    )
    parser.add_argument(
        "--activation",
        type=str,
        help="neural network activation function",
        default="relu",
        choices=["relu", "tanh", "sigmoid"],
    )
    parser.add_argument(
        "--neurons_per_layer",
        type=int,
        help="number of neurons per hidden layer",
        default=128,
    )
    parser.add_argument("--date", type=str, help="date run is saved under")
    parser.add_argument("-v", action="store_true", help="verbose flag")
    parser.add_argument(
        "-r",
        action="store_true",
        help="restart flag, deletes data, restarts from random network",
    )
    return parser