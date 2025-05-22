"""
For aggregated logpdf plots over all tasks and seeds, per each algorithm variant.
"""

import itertools as it
import logging
import os
import re
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

from bnn_pref.utils.plotting import (
    get_font_kw,
    get_legend_kw,
    invisible_topright_spines,
    prettify_title,
    rgb_values,
    set_xlim_offset,
    smooth,
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    # "cheetahRandom",
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
    # "kitchenComplete",
    # "kitchenPartial",
    # "kitchenMixed",
    "mazeUDense",
    "mazeMediumDense",
    # "mazeLargeDense",
]
algs = ["ekf", "sgd"]
is_als = [True, False]

use_smooth = True

# * == change this block ==
save_dir = "PATH/TO/YOUR/bnn_pref/_viz/logpdf"
dirp = "PATH/TO/YOUR/bnn_pref/_runs/pref/CHANGEME"
# * == change this block ==

out = np.load(f"{dirp}/stats.npz", allow_pickle=True)

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
        return "PreferenceEKF (Active)" if is_al else "PreferenceEKF (Random)"
    else:
        return "DeepEnsemble (Active)" if is_al else "DeepEnsemble (Random)"


def get_style(alg: str, is_al: bool) -> dict:
    color = rgb_values["orange"] if alg == "ekf" else rgb_values["blue"]
    linestyle = "-" if is_al else "--"
    return {"color": color, "linestyle": linestyle}


fig, axs = plt.subplots(3, 4, figsize=(12, 7.5), sharex=True)
axs = axs.flatten()

for i, task in enumerate(tasks):
    ax = axs[i]
    invisible_topright_spines(ax)
    # ax.axhline(y=-0.69, linestyle=":", linewidth=1, color="red")  # ln(0.5) = -0.69
    # y_lim_min, y_lim_max = -0.73, 0
    for alg, is_al in it.product(algs, is_als):
        arr = test_logpdf_all[f"{alg}_{is_al}"][i, :, :]  # (seeds, steps)
        mean_E = arr.mean(axis=0)  # (steps, )
        std_E = arr.std(axis=0)  # (steps, )
        mean_E = smooth(mean_E) if use_smooth else mean_E
        std_E = smooth(std_E) if use_smooth else std_E
        label = get_label(alg, is_al)
        style = get_style(alg, is_al)
        ax.plot(mean_E, label=label, **style)
        ax.fill_between(
            range(len(mean_E)),
            mean_E - std_E,
            mean_E + std_E,
            alpha=0.2,
            **style,
        )
    ax.set_title(prettify_title(task), **get_font_kw(14))

    y_all = np.concatenate([line.get_ydata() for line in ax.get_lines()])
    y_lim_min = min(y_all) - 0.03
    y_lim_max = max(y_all) + 0.03
    ax.set_ylim(y_lim_min, y_lim_max)

    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x):d}" for x in xticks], **get_font_kw(12))
    set_xlim_offset(ax)
    ax.set_xlim(right=len(mean_E))  # Cut off the graph at x=60

    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.2f}" for y in yticks], **get_font_kw(12))


# --- Shared legend using dummy lines, outside the subplots ---
dummy_lines = [
    plt.plot([], [], **get_style("ekf", True), label=get_label("ekf", True))[0],
    plt.plot([], [], **get_style("ekf", False), label=get_label("ekf", False))[0],
    plt.plot([], [], **get_style("sgd", True), label=get_label("sgd", True))[0],
    plt.plot([], [], **get_style("sgd", False), label=get_label("sgd", False))[0],
]
fig.supxlabel("Number of Queries", **get_font_kw(16))
fig.supylabel("Test Log-Likelihood", **get_font_kw(16))
fig.legend(
    dummy_lines,
    [
        get_label("ekf", True),
        get_label("ekf", False),
        get_label("sgd", True),
        get_label("sgd", False),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    handlelength=2,
    **get_legend_kw(16),
)
plt.tight_layout(rect=[0, 0.03, 1, 1])
save_path = f"{save_dir}/logpdf_{timestamp}.png"
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"Plot saved as: {save_path}")


# * logpdf plot, averaged over all tasks and seeds, for each algorithm variant
fig, ax = plt.subplots(figsize=(10, 6))
invisible_topright_spines(ax)
for alg, is_al in it.product(algs, is_als):
    key = f"{alg}_{is_al}"
    data_T = test_logpdf_aggregate[key]  # (S: seeds), (T: steps, )
    data_mean = data_T.mean(axis=0)
    data_std = data_T.std(axis=0)
    label = get_label(alg, is_al)
    style = get_style(alg, is_al)
    ax.plot(data_mean, label=label, **style, linewidth=2)
    ax.fill_between(
        range(len(data_mean)),
        data_mean - data_std,
        data_mean + data_std,
        alpha=0.2,
        **style,
    )
# --- Add "x% fewer samples" annotation between EKF (Active) and EKF (Random) ---

# Get means for EKF (Active) and EKF (Random)
ekf_active_mean_T = test_logpdf_aggregate["ekf_True"].mean(axis=0)
ekf_random_mean_T = test_logpdf_aggregate["ekf_False"].mean(axis=0)

# Find the y-value at the last step of EKF (Random)
y_tgt = ekf_random_mean_T[-1]
x_random = len(ekf_random_mean_T) - 1

# Find the first x in EKF (Active) that reaches or exceeds y_target
x_active = np.argmax(ekf_active_mean_T >= y_tgt)

frac = 1 - x_active / x_random

# Draw vertical dotted lines down to a lower y for annotation
y_bottom = ax.get_ylim()[0] + 0.20  # adjust as needed for your plot
ax.vlines([x_active, x_random], y_bottom, y_tgt, linestyles="dotted", colors="k")
ax.plot([x_active, x_random], [y_tgt, y_tgt], "ko", markersize=4)  # mark the two points

# Draw double-headed arrow and annotate at the bottom
ax.annotate(
    "",
    xy=(x_active, y_bottom),
    xytext=(x_random, y_bottom),
    arrowprops=dict(
        arrowstyle="<->", color="black", linewidth=1.5, shrinkA=0, shrinkB=0
    ),
)
ax.text(
    (x_active + x_random) / 2,
    y_bottom - 0.01,  # slightly below the arrow
    f"~{frac:.0%} fewer samples",
    ha="center",
    va="top",
    color="black",
    **get_font_kw(18),
)

ax.set_xlabel("Number of Queries", **get_font_kw(18))
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_xticklabels([f"{int(x):d}" for x in xticks], **get_font_kw(16))
set_xlim_offset(ax)
ax.set_xlim(right=60.5)  # Cut off the graph at x=60

ax.set_ylabel("Test Log-Likelihood", **get_font_kw(18))
yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels([f"{y:.2f}" for y in yticks], **get_font_kw(16))

# ax.set_title("Test Log-Likelihood Across Tasks", **get_font_kw(13))
ax.legend(**get_legend_kw(18), loc="lower right")
save_path = f"{save_dir}/logpdf_{timestamp}_agg.png"
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()

print(f"Plot saved as: {save_path}")
