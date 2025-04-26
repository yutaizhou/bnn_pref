from .make_d4rl import make_d4rl_data
from .make_ogbench import make_ogbench_data
from .make_prefcc import make_prefcc_data
from .make_synthetic import make_synthetic_data

__all__ = [
    "make_ogbench_data",
    "make_prefcc_data",
    "make_synthetic_data",
    "make_d4rl_data",
]

dataset_creators = {
    "ogbench": make_ogbench_data,
    "prefcc": make_prefcc_data,
    "synthetic": make_synthetic_data,
    "d4rl": make_d4rl_data,
}
