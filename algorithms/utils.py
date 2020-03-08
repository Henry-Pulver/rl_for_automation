from functools import reduce
import operator
import datetime
from pathlib import Path
from typing import Tuple, Optional
import torch.nn as nn
from collections import namedtuple


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def generate_save_location(
    data_save_path: Path,
    nn_layers: Tuple,
    algo: str,
    env_name: str,
    seed: int,
    hyp_str: str,
    date: Optional[str] = None,
) -> Path:
    date = datetime.date.today().strftime("%d-%m-%Y") if date is None else date
    arch_str = "-".join(str(x) for x in nn_layers)
    return Path(
        f"{data_save_path}/{algo}/{env_name}/{date}/hyp-{hyp_str}/{arch_str}/seed-{seed}"
    )


def generate_ppo_hyp_str(ppo_type: str, hyp: namedtuple):
    if ppo_type == "clip":
        return hyp.epsilon
    elif ppo_type == "unclipped":
        return "0"
    elif ppo_type == "fixed_KL":
        return hyp.beta
    elif ppo_type == "adaptive_KL":
        return hyp.d_targ
    else:
        raise NameError("ppo_type variable incorrect!")


def generate_gail_str(ppo_type: str, hyp: namedtuple):
    ppo_str = generate_ppo_hyp_str(ppo_type, hyp)
    gail_num_demos = hyp.num_demos
    return f"{ppo_str}-num_demos_{gail_num_demos}"


def get_activation(activation: str):
    activations = ["tanh", "relu", "sigmoid"]
    assert activation in activations
    if activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
