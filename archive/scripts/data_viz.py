import os
from collections import defaultdict
from functools import partial
from typing import Dict

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
import logging
from datetime import datetime

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from jaxtyping import Array, Float

from bnn_pref.data import dataset_creators
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.data.pref_utils import QueryData
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import MeanStd
from bnn_pref.utils.utils import get_random_seed, nested_defaultdict
from scripts.sweep_tasks_ekf import run_ekf
from scripts.sweep_tasks_ensemble import run_ensemble

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    """
    Plot histogram of returns for each task
    """
    seed = get_random_seed() if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed)

    tasks = [
        "reacher",
        "lunar",
        "cheetah",
        "acrobot",
        "ball",
        "cartpoleSwing",
        "cheetahDMC",
        "hopperHop",
        "pendulum",
        "reacherEasy",
        "reacherHard",
        "walkerWalk",
        "ogbench",
    ]

    fig, axs = plt.subplots(4, 4, figsize=(12, 8))  # 13 tasks total
    axs = axs.flatten()
    for i, task in enumerate(tasks):
        # * update cfg
        new_cfg = hydra.compose("config", overrides=[f"task={task}"])
        cfg["task"].update(new_cfg["task"])

        # * create dataset
        key, key_data, *key_seeds = jr.split(key, 2 + cfg["seeds"])
        data_dict = dataset_creators[cfg["task"]["ds_type"]](key_data, cfg)

        # * plot histogram of returns
        train_returns = data_dict["train_trajs"]["returns"]
        n_trajs = train_returns.shape[0]
        print(f"{task}: {n_trajs} trajectories")
        ax = axs[i]
        ax.hist(train_returns, bins=50, edgecolor="black")
        ax.set_title(f"{task} ({n_trajs} trajs)")
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")

    fig.suptitle("Histogram of returns for each task (training)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
