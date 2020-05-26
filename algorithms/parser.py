from argparse import ArgumentParser


def get_ppo_parser(*args, **kwargs):
    parser = get_actor_critic_parser(*args, **kwargs)
    parser.add_argument(
        "--ppo_type",
        type=str,
        help="PPO variant to use",
        default="clip",
        choices=["clip", "unclipped", "fixed_KL", "adaptive_KL"],
    )
    parser.add_argument(
        "--epsilon", type=float, help="PPO clipping param", default=0.2,
    )
    parser.add_argument(
        "--beta", type=float, help="PPO fixed KL param", default=0.003,
    )
    parser.add_argument(
        "--d_targ", type=float, help="PPO adaptive KL param", default=0.01,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="number of epochs to train each T timesteps",
        default=3,
    )
    parser.add_argument(
        "--T", type=int, help="number of timeseps between updates", default=1024,
    )
    return parser


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
    parser.add_argument(
        "--value_coeff", type=float, help="value coefficient param", default=0.5,
    )
    parser.add_argument(
        "--entropy_coeff", type=float, help="entropy coefficient param", default=0.01,
    )
    parser.add_argument(
        "--worst_score",
        type=float,
        help="used to determine if unstable - the worst score possible in the env",
    )
    parser.add_argument(
        "--burn_in_steps", type=int, help="number of burn in steps", default=0
    )
    return parser


def get_base_parser(*args, **kwargs):
    parser = ArgumentParser(*args, **kwargs)
    parser.add_argument(
        "--save_base_path", type=str, help="base path to save plots", default="data"
    )
    parser.add_argument(
        "--load_path", type=str, help="path to network that will be loaded",
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
        "--log_type", type=str, help="type of logging to use", default="legacy",
    )
    parser.add_argument(
        "--random_seeds", type=set, help="random seeds used", default={0, 1, 2, 3, 4}
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
