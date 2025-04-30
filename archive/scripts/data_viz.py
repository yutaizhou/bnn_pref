import os

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
import logging

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig

from bnn_pref.data import dataset_creators
from bnn_pref.utils.utils import get_random_seed

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="config", config_path="../../src/cfg")
def main(cfg):
    """
    Plot histogram of returns for each task
    """
    n_bins = 10
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)

    tasks = [
        # * Gym
        # "reacher",
        # "lunar",
        # "cheetah",
        # * Deepmind Control
        # "acrobot",
        # "ball",
        # "cartpoleSwing",
        # "cheetahDMC",
        # "hopperHop",
        # "pendulum",
        # "reacherEasy",
        # "reacherHard",
        # "walkerWalk",
        # * D4RL
        "cheetahRandom",
        "cheetahMediumReplay",
        "cheetahMediumExpert",
        "hopperRandom",
        "hopperMediumReplay",
        "hopperMediumExpert",
        "walkerRandom",
        "walkerMediumReplay",
        "walkerMediumExpert",
        "penHuman",
        "penExpert",
        "penCloned",
    ]

    fig, axs = plt.subplots(4, 4, figsize=(12, 8))  # 13 tasks total
    axs = axs.flatten()

    # Create empty lists to store legend lines and labels
    lines = []
    labels = []

    for i, task in enumerate(tasks):
        # * update cfg
        new_cfg = hydra.compose("config", overrides=[f"task={task}"])
        cfg["task"].update(new_cfg["task"])

        # * create dataset
        key, key_data = jr.split(key, 2)
        data_dict = dataset_creators[cfg["task"]["ds_type"]](key_data, cfg)

        # * plot histogram of returns
        train_trajs, test_trajs = data_dict["train_trajs"], data_dict["test_trajs"]
        train_returns, test_returns = train_trajs["returns"], test_trajs["returns"]
        train_prefs, test_prefs = data_dict["train_prefs"], data_dict["test_prefs"]
        nt_train, nt_test = train_returns.shape[0], test_returns.shape[0]
        nq_train, nq_test = (
            train_prefs.queries_Q2.shape[0],
            test_prefs.queries_Q2.shape[0],
        )
        nq_train_mislabel = train_prefs.n_mislabels
        mistake_ratio = nq_train_mislabel / nq_train
        print(
            f"{task:13}: train/test {nt_train} / {nt_test} trajs, {nq_train} / {nq_test} queries, {nq_train_mislabel} / {nq_train} train mislabels ({mistake_ratio:.2%})\n"
            f"  train traj obs: {data_dict['train_trajs']['observations'].shape}"
            f"\n"
        )

        # print(f"  Best traj return: {train_returns[-1]}")
        # print(f"  Best traj rewards: {train_trajs['rewards'][-1]}")

        # * plot histogram
        ax = axs[i]
        ax.hist(train_returns, bins=n_bins, edgecolor="black")
        ax.hist(test_returns, bins=n_bins, edgecolor="black")
        ax.set_title(f"{task} ({nt_train} / {nt_test} trajs)", fontsize=8)
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")

        mean = jnp.mean(train_returns)
        median = jnp.median(train_returns)
        mean_line = ax.axvline(mean, color="red", label="Mean")
        median_line = ax.axvline(median, color="green", label="Median")

        # Only store the lines and labels from the first subplot
        if i == 0:
            lines.extend([mean_line, median_line])
            labels.extend(["Mean", "Median"])

    # Add a single legend to the figure
    fig.legend(lines, labels, loc="center right")

    fig.suptitle("Histogram of returns for each task (training)")
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.show()


if __name__ == "__main__":
    main()
