"""
For aggregated logpdf plots over all tasks and seeds, per each algorithm variant.
"""

import itertools as it
import logging
import os
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Tuple

import ipdb

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
logging.getLogger("absl").setLevel(logging.WARNING)

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from hydra.core.hydra_config import HydraConfig

save_dir = "/scr/yutaizho/projects/bnn_pref/_viz"

tasks = [
    # * gym
    # "reacher",
    # "lunar",
    # "cheetah",
    # # * Deepmind Control
    # "acrobot",
    # "ball",
    # "cartpoleSwing",
    # "cheetahDMC",
    # "hopperHop",
    # "pendulum",
    # "walkerWalk",
    # # * D4RL
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
    # "penCloned",
    "kitchenComplete",
    # "kitchenPartial",
    "kitchenMixed",
    "mazeUDense",
    "mazeMediumDense",
    "mazeLargeDense",
]
algs = ["ekf", "sgd"]
is_als = [True, False]

fp = "/scr/yutaizho/projects/bnn_pref/_runs/20250501_002234_rm_d4rl_18tasks/stats.npz"
out = np.load(fp, allow_pickle=True)

test_logpdf_all = defaultdict(lambda: list())
for alg, is_al in it.product(algs, is_als):
    for task in tasks:
        res = out[task].item()[alg][is_al]
        test_logpdf_all[f"{alg}_{is_al}"].append(res["test_logpdf_all"])

test_logpdf_all = {k: np.array(v) for k, v in test_logpdf_all.items()}
# dict[alg_is_al] = (n_tasks, seeds, steps)

# * aggregate over tasks and seeds
test_logpdf_aggregate = {}
for alg, is_al in it.product(algs, is_als):
    arr = test_logpdf_all[f"{alg}_{is_al}"]  # (n_tasks, seeds, steps)
    test_logpdf_aggregate[f"{alg}_{is_al}"] = arr.mean(axis=(0,))


def get_label(alg: str, is_al: bool) -> str:
    if alg == "ekf":
        return "EKF (Active)" if is_al else "EKF (Random)"
    else:
        return "Ensemble (Active)" if is_al else "Ensemble (Random)"


def get_style(alg: str, is_al: bool) -> dict:
    color = "blue" if alg == "ekf" else "orange"
    linestyle = "-" if is_al else "--"
    return {"color": color, "linestyle": linestyle, "linewidth": 1}


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

fig, axs = plt.subplots(5, 4, figsize=(12, 10))  # larger for clarity
axs = axs.flatten()

for i, task in enumerate(tasks):
    ax = axs[i]
    ax.set_ylim(-0.73, 0)  # ln(0.48)
    ax.axhline(y=-0.69, linestyle=":", linewidth=1, color="red")  # ln(0.5) = -0.69
    for alg, is_al in it.product(algs, is_als):
        arr = test_logpdf_all[f"{alg}_{is_al}"][i, :, :]  # (seeds, steps)
        arr_mean = arr.mean(axis=0)  # (steps, )
        arr_std = arr.std(axis=0)  # (steps, )
        label = get_label(alg, is_al)
        style = get_style(alg, is_al)
        ax.plot(arr_mean, label=label, **style)
        ax.fill_between(
            range(len(arr_mean)),
            arr_mean - arr_std,
            arr_mean + arr_std,
            alpha=0.2,
            **style,
        )
    ax.set_title(task, fontsize=8)
    ax.set_xlabel("Number of Queries")
    ax.set_ylabel("Test Log PDF")

# Hide unused subplots
# for j in range(len(tasks), len(axs)):
#     axs[j].axis("off")

# --- Shared legend using dummy lines, outside the subplots ---
dummy_lines = [
    plt.plot([], [], color="blue", linestyle="--", label="EKF (Random)")[0],
    plt.plot([], [], color="blue", linestyle="-", label="EKF (Active)")[0],
    plt.plot([], [], color="orange", linestyle="--", label="Ensemble (Random)")[0],
    plt.plot([], [], color="orange", linestyle="-", label="Ensemble (Active)")[0],
]
fig.legend(
    dummy_lines,
    ["EKF (Random)", "EKF (Active)", "Ensemble (Random)", "Ensemble (Active)"],
    loc="center right",
)
fig.suptitle("log PDF vs. num queries (noise=True, sanity=False)", fontsize=18)
plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for legend and suptitle

save_path = f"{save_dir}/logpdf_comparison_{timestamp}.png"
plt.savefig(save_path)
plt.close()
print(f"Plot saved as: {save_path}")

# * logpdf plot, averaged over all tasks and seeds, for each algorithm variant
fig, ax = plt.subplots(figsize=(10, 6))
for alg, is_al in it.product(algs, is_als):
    key = f"{alg}_{is_al}"
    data_T = test_logpdf_aggregate[key]  # (T: steps, ) (S: seeds)
    data_mean = data_T.mean(axis=0)
    data_std = data_T.std(axis=0)
    label = get_label(alg, is_al)
    style = get_style(alg, is_al)
    ax.plot(data_mean, label=label, **style)
    ax.fill_between(
        range(len(data_mean)),
        data_mean - data_std,
        data_mean + data_std,
        alpha=0.2,
        **style,
    )

ax.set_xlabel("Number of Queries")
ax.set_ylabel("Test Log PDF")
ax.set_title("Test Log PDF Comparison Across Algorithms")
ax.legend()

# Save the plot
save_path = f"{save_dir}/logpdf_comparison_{timestamp}_agg.png"
plt.savefig(save_path)
plt.close()

print(f"Plot saved as: {save_path}")
