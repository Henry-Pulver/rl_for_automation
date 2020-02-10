from functools import reduce
import operator
import datetime
from pathlib import Path
from typing import Tuple, Optional


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def generate_save_location(
    data_save_path: Path,
    nn_layers: Tuple,
    algo: str,
    env_name: str,
    seed: int,
    date: Optional[str] = None,
) -> Path:
    date = datetime.date.today().strftime("%d-%m-%Y") if date is None else date
    arch_str = "-".join(str(x) for x in nn_layers)
    return Path(f"{data_save_path}/{algo}/{env_name}/{date}/{arch_str}/seed-{seed}")
