"""
For aggregated logpdf plots over all tasks and seeds, per each algorithm variant.
"""

import itertools as it
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime

import ipdb

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
logging.getLogger("absl").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import numpy as np

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

# tasks = [
#     # # * D4RL
#     "cheetahRandom",
#     "cheetahMediumReplay",
#     "cheetahMediumExpert",
#     "hopperRandom",
#     "hopperMediumReplay",
#     "hopperMediumExpert",
#     "walkerRandom",
#     "walkerMediumReplay",
#     "walkerMediumExpert",
#     "penHuman",
#     "penExpert",
#     "penCloned",
#     "kitchenComplete",
#     "kitchenPartial",
#     "kitchenMixed",
#     "mazeUDense",
#     "mazeMediumDense",
#     "mazeLargeDense",
# ]

algs = ["ekf", "sgd", "do"]
is_als = [True, False]
n_tasks = len(tasks)

use_smooth = True

# * == vv change this block vv ==
# save_dir: where to save
# dirp: where to load
dirp = sys.argv[1]
# save_dir = "/scr/yutaizho/projects/bnn_pref/_viz/logpdf"  # where to save
save_dir = dirp
# * == ^^ change this block ^^ ==


# data["cheetahRandom"]["ekf"][False]["test_logpdf_all"] = (n_seeds, n_steps)
data = {}
for task in tasks:
    for folder in os.listdir(dirp):
        if f"task={task}" in folder:
            path = os.path.join(dirp, folder, "stats.npz")
            data[task] = np.load(path, allow_pickle=True)[task].item()


# stats["logpdf"][alg_is_al] = (n_tasks, seeds, steps); e.g. stats["logpdf"]["ekf_True"]
stats = {
    "logpdf": defaultdict(lambda: list()),
    "acc": defaultdict(lambda: list()),
    "ece": defaultdict(lambda: list()),
    "brier": defaultdict(lambda: list()),
    "coverage": defaultdict(lambda: list()),
}
for alg, is_al in it.product(algs, is_als):
    for task in tasks:
        res = data[task][alg][is_al]
        stats["logpdf"][f"{alg}_{is_al}"].append(res["test_logpdf_all"])
        stats["acc"][f"{alg}_{is_al}"].append(res["test_acc_all"])
        stats["ece"][f"{alg}_{is_al}"].append(res["test_ece_all"])
        stats["brier"][f"{alg}_{is_al}"].append(res["test_brier_all"])
        stats["coverage"][f"{alg}_{is_al}"].append(res["test_coverage_all"])


def list2array(stats: dict) -> dict:
    """
    stats: dict[str, list] -> dict[str, np.ndarray]
    stats[alg_is_al] = List[n_tasks] -> np.ndarray(n_tasks, seeds, steps)
    """
    return {k: np.array(v) for k, v in stats.items()}


stats["logpdf"] = list2array(stats["logpdf"])
stats["acc"] = list2array(stats["acc"])
stats["ece"] = list2array(stats["ece"])
stats["brier"] = list2array(stats["brier"])
stats["coverage"] = list2array(stats["coverage"])

logpdf_all = stats["logpdf"]  # (n_tasks, seeds, steps)
acc_all = stats["acc"]
ece_all = stats["ece"]
brier_all = stats["brier"]
coverage_all = stats["coverage"]

# * aggregate over tasks and seeds
logpdf_agg, acc_agg, ece_agg, brier_agg, coverage_agg = {}, {}, {}, {}, {}
for alg, is_al in it.product(algs, is_als):
    arr = logpdf_all[f"{alg}_{is_al}"]  # (n_tasks, seeds, steps)
    logpdf_agg[f"{alg}_{is_al}"] = arr.mean(axis=(0,))
    acc_agg[f"{alg}_{is_al}"] = acc_all[f"{alg}_{is_al}"].mean(axis=(0,))
    ece_agg[f"{alg}_{is_al}"] = ece_all[f"{alg}_{is_al}"].mean(axis=(0,))
    brier_agg[f"{alg}_{is_al}"] = brier_all[f"{alg}_{is_al}"].mean(axis=(0,))
    coverage_agg[f"{alg}_{is_al}"] = coverage_all[f"{alg}_{is_al}"].mean(axis=(0,))
stats_agg = {
    "logpdf": logpdf_agg,
    "acc": acc_agg,
    "ece": ece_agg,
    "brier": brier_agg,
    "coverage": coverage_agg,
}


# * plot logpdf
def get_label(alg: str, is_al: bool) -> str:
    if alg == "ekf":
        return "PreferenceEKF (Active)" if is_al else "PreferenceEKF (Random)"
    elif alg == "sgd":
        return "DeepEnsemble (Active)" if is_al else "DeepEnsemble (Random)"
    elif alg == "do":
        return "Dropout (Active)" if is_al else "Dropout (Random)"
    else:
        raise ValueError(f"Invalid algorithm: {alg}")


def get_style(alg: str, is_al: bool) -> dict:
    if alg == "ekf":
        color = rgb_values["orange"]
    elif alg == "sgd":
        color = rgb_values["blue"]
    elif alg == "do":
        color = rgb_values["green"]
    else:
        raise ValueError(f"Invalid algorithm: {alg}")
    linestyle = "-" if is_al else "--"
    return {"color": color, "linestyle": linestyle}


# fig, axs = plt.subplots(3, 4, figsize=(12, 7.5), sharex=True)
fig, axs = plt.subplots(5, 4, figsize=(12, 15), sharex=True)
axs = axs.flatten()

for i, task in enumerate(tasks):
    ax = axs[i]
    invisible_topright_spines(ax)
    # ax.axhline(y=-0.69, linestyle=":", linewidth=1, color="red")  # ln(0.5) = -0.69
    # y_lim_min, y_lim_max = -0.73, 0
    for alg, is_al in it.product(algs, is_als):
        arr = logpdf_all[f"{alg}_{is_al}"][i, :, :]  # (seeds, steps)
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
    plt.plot([], [], **get_style("do", True), label=get_label("do", True))[0],
    plt.plot([], [], **get_style("do", False), label=get_label("do", False))[0],
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
        get_label("do", True),
        get_label("do", False),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=3,
    handlelength=2,
    **get_legend_kw(16),
)
plt.tight_layout(rect=[0, 0.03, 1, 1])
save_path = f"{save_dir}/{timestamp}_logpdf_nTasks={n_tasks}.png"
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"Plot saved as: {save_path}")


# * logpdf plot, averaged over all tasks and seeds, for each algorithm variant
fig, ax = plt.subplots(figsize=(10, 6))
invisible_topright_spines(ax)
for alg, is_al in it.product(algs, is_als):
    key = f"{alg}_{is_al}"
    data_T = logpdf_agg[key]  # (S: seeds), (T: steps, )
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
ekf_active_mean_T = logpdf_agg["ekf_True"].mean(axis=0)
ekf_random_mean_T = logpdf_agg["ekf_False"].mean(axis=0)

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
save_path = f"{save_dir}/{timestamp}_logpdf_nTasks={n_tasks}_agg.png"
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()

print(f"Plot saved as: {save_path}")

# * plot all metrics aggregated over tasks: logpdf, acc, ece, brier, coverage
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
axes = axes.flatten()

for i, metric in enumerate(["logpdf", "acc", "ece", "brier", "coverage"]):
    ax = axes[i]
    invisible_topright_spines(ax)
    for alg, is_al in it.product(algs, is_als):
        key = f"{alg}_{is_al}"
        data_T = stats_agg[metric][key]  # (S: seeds), (T: steps, )
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
    ax.set_ylabel(prettify_title(metric), **get_font_kw(18))
fig.supxlabel("Number of Queries", **get_font_kw(16))
fig.legend(
    dummy_lines,
    [
        get_label("ekf", True),
        get_label("ekf", False),
        get_label("sgd", True),
        get_label("sgd", False),
        get_label("do", True),
        get_label("do", False),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=3,
    handlelength=2,
    **get_legend_kw(16),
)
save_path = f"{save_dir}/{timestamp}_metrics_nTasks={n_tasks}_agg.png"
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"Plot saved as: {save_path}")
