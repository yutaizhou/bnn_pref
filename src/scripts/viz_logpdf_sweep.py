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
dirp = sys.argv[1]  # where to load
save_dir = dirp  # where to save
metric_names = ["logpdf", "acc", "ece", "brier", "coverage", "sharpness"]
use_stderr = True  # otherwise use stderr
use_smooth = True
handle_nan = False
nan_mask = False


# neurips tasks
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

# iclr tasks
tasks = [
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
    # "penExpert",
    # "penCloned",
    # "kitchenComplete",
    # "kitchenPartial",
    # "kitchenMixed",
    # "mazeUDense",
    "mazeMediumDense",
    "mazeLargeDense",
]

# # all tasks
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
#     # "kitchenComplete",
#     # "kitchenPartial",
#     # "kitchenMixed",
#     "mazeUDense",
#     "mazeMediumDense",
#     "mazeLargeDense",
# ]

n_tasks = len(tasks)
algs = ["ekf", "sgd", "do"]
is_als = [True, False]


"""
Given arr of shape (nseeds, steps) -> (steps, ) with mean, std, sterr, etc.
two ways to handle nans:
1. use nanmean, nanstd, etc: each timestep may have different number of nans, so sample size may differ across timesteps
2. mask out rows that contain at least one nan: this ensure that the sample size is the same across timesteps
"""


def mean(arr, axis, nan=False, nan_mask=False):
    if nan:
        if nan_mask:
            row_mask = np.isfinite(arr).all(1)  # (nseeds, )
            arr = arr[row_mask]
            return np.mean(arr, axis=axis)
        else:
            return np.nanmean(arr, axis=axis)
    else:
        return np.mean(arr, axis=axis)


def std(arr, axis, nan=False, nan_mask=False):
    if nan:
        if nan_mask:
            row_mask = np.isfinite(arr).all(1)  # (nseeds, )
            arr = arr[row_mask]
            return np.std(arr, axis=axis)
        else:
            return np.nanstd(arr, axis=axis)
    else:
        return np.std(arr, axis=axis)


def sterr(arr, axis, nan=False, nan_mask=False):
    if nan:
        if nan_mask:
            row_mask = np.isfinite(arr).all(1)  # (nseeds, )
            arr = arr[row_mask]
            return np.std(arr, axis=axis) / np.sqrt(arr.shape[0])
        else:
            nonnans_per_step = np.isfinite(arr).sum(0)  # (steps, )
            return np.nanstd(arr, axis=0) / np.sqrt(nonnans_per_step)
    else:
        return np.std(arr, axis=axis) / np.sqrt(arr.shape[0])


# *load data from stats.npz -> stats[metric][alg_is_al] = array(n_tasks, seeds, steps)
# data["cheetahRandom"]["ekf"][False]["test_logpdf_all"] = (n_seeds, n_steps)
data = {}
for task in tasks:
    for folder in os.listdir(dirp):
        if f"task={task}" in folder:
            path = os.path.join(dirp, folder, "stats.npz")
            data[task] = np.load(path, allow_pickle=True)[task].item()


# stats["logpdf"][alg_is_al] = (n_tasks, seeds, steps); e.g. stats["logpdf"]["ekf_True"]
stats = {metric: defaultdict(lambda: list()) for metric in metric_names}

for alg, is_al in it.product(algs, is_als):
    for task in tasks:
        res = data[task][alg][is_al]  # (n_seeds, n_steps)
        for metric in metric_names:
            stats[metric][f"{alg}_{is_al}"].append(res[f"test_{metric}_all"])


def list2array(stats: dict) -> dict:
    """
    stats: dict[str, list] -> dict[str, np.ndarray]
    stats[alg_is_al] = List[n_tasks] -> np.ndarray(n_tasks, seeds, steps)
    """
    return {k: np.array(v) for k, v in stats.items()}


for metric in metric_names:
    stats[metric] = list2array(stats[metric])

# * aggregate over tasks -> stats_agg[metric][alg_is_al] = array(seeds, steps)
stats_agg = {metric: {} for metric in metric_names}

for alg, is_al in it.product(algs, is_als):
    for metric in metric_names:
        arr = stats[metric][f"{alg}_{is_al}"]  # (n_tasks, seeds, steps)
        stats_agg[metric][f"{alg}_{is_al}"] = mean(arr, axis=(0,), nan=handle_nan)


def get_label(alg: str, is_al: bool) -> str:
    if alg == "ekf":
        return "PreferenceEKF (A)" if is_al else "PreferenceEKF (R)"
    elif alg == "sgd":
        return "DeepEnsemble (A)" if is_al else "DeepEnsemble (R)"
    elif alg == "do":
        return "Dropout (A)" if is_al else "Dropout (R)"
    elif alg == "llmcmc":
        return "LLMCMC (A)" if is_al else "LLMCMC (R)"
    elif alg == "laplace":
        return "Laplace (A)" if is_al else "Laplace (R)"
    else:
        raise ValueError(f"Invalid algorithm: {alg}")


def get_style(alg: str, is_al: bool) -> dict:
    if alg == "ekf":
        color = rgb_values["orange"]
    elif alg == "sgd":
        color = rgb_values["blue"]
    elif alg == "do":
        color = rgb_values["green"]
    elif alg == "llmcmc":
        color = rgb_values["purple"]
    elif alg == "laplace":
        color = rgb_values["gray"]
    else:
        raise ValueError(f"Invalid algorithm: {alg}")
    linestyle = "-" if is_al else "--"
    return {"color": color, "linestyle": linestyle}


# * plot logpdf for each task
fig, axs = plt.subplots(3, 4, figsize=(12, 7.5), sharex=True)
# fig, axs = plt.subplots(5, 4, figsize=(12, 15), sharex=True)
axs = axs.flatten()

for i, task in enumerate(tasks):
    ax = axs[i]
    invisible_topright_spines(ax)
    # ax.axhline(y=-0.69, linestyle=":", linewidth=1, color="red")  # ln(0.5) = -0.69
    # y_lim_min, y_lim_max = -0.73, 0
    for alg, is_al in it.product(algs, is_als):
        arr = stats["logpdf"][f"{alg}_{is_al}"]  # (tasks, seeds, steps)
        arr_task = arr[i, :, :]  # (seeds, steps)
        mean_E = mean(arr_task, axis=0, nan=handle_nan)  # (steps, )
        std_E = (
            std(arr_task, axis=0, nan=handle_nan)
            if not use_stderr
            else sterr(arr_task, axis=0, nan=handle_nan)
        )  # (steps, )
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
    plt.plot([], [], **get_style(alg, is_al), label=get_label(alg, is_al))[0]
    for alg, is_al in it.product(algs, is_als)
]
fig.supxlabel("Number of Queries", **get_font_kw(16))
fig.supylabel("Test Log-Likelihood", **get_font_kw(16))
fig.legend(
    dummy_lines,
    [get_label(alg, is_al) for alg, is_al in it.product(algs, is_als)],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=len(algs),
    handlelength=2,
    **get_legend_kw(16),
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
save_path = f"{save_dir}/{timestamp}_logpdf_nTasks={n_tasks}.png"
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"Plot saved as: {save_path}")


# * plot logpdf aggregated over all tasks
fig, ax = plt.subplots(figsize=(10, 6))
invisible_topright_spines(ax)
for alg, is_al in it.product(algs, is_als):
    alg_isactive = f"{alg}_{is_al}"
    arr = stats_agg["logpdf"][alg_isactive]  # (seeds, steps)
    data_mean = mean(arr, axis=0, nan=handle_nan)
    data_std = (
        std(arr, axis=0, nan=handle_nan)
        if not use_stderr
        else sterr(arr, axis=0, nan=handle_nan)
    )
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
# only do so if EKF active outperforms EKF random
if "ekf" in algs:
    ekf_active_mean_T = mean(stats_agg["logpdf"]["ekf_True"], axis=0, nan=handle_nan)
    ekf_random_mean_T = mean(stats_agg["logpdf"]["ekf_False"], axis=0, nan=handle_nan)

    if ekf_active_mean_T[-1] > ekf_random_mean_T[-1]:
        # Find the y-value at the last step of EKF (Random)
        y_tgt = ekf_random_mean_T[-1]
        x_random = len(ekf_random_mean_T) - 1

        # Find the first x in EKF (Active) that reaches or exceeds y_target
        x_active = np.argmax(ekf_active_mean_T >= y_tgt)

        frac = 1 - x_active / x_random

        # Draw vertical dotted lines down to a lower y for annotation
        y_bottom = ax.get_ylim()[0] + 0.20  # adjust as needed for your plot
        ax.vlines(
            [x_active, x_random], y_bottom, y_tgt, linestyles="dotted", colors="k"
        )
        ax.plot(
            [x_active, x_random], [y_tgt, y_tgt], "ko", markersize=4
        )  # mark the two points

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

ax.legend(
    **get_legend_kw(18), loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(algs)
)
save_path = f"{save_dir}/{timestamp}_logpdf_nTasks={n_tasks}_agg.png"
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()

print(f"Plot saved as: {save_path}")

# * plot all metrics aggregated over tasks
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
axes = axes.flatten()

for i, metric in enumerate(metric_names):
    ax = axes[i]
    invisible_topright_spines(ax)
    for alg, is_al in it.product(algs, is_als):
        alg_isactive = f"{alg}_{is_al}"
        arr = stats_agg[metric][alg_isactive]  # (seeds, steps)
        data_mean = mean(arr, axis=0, nan=handle_nan)  # (steps, )
        data_std = (
            std(arr, axis=0, nan=handle_nan)
            if not use_stderr
            else sterr(arr, axis=0, nan=handle_nan)
        )
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
    [get_label(alg, is_al) for alg, is_al in it.product(algs, is_als)],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=len(algs),
    handlelength=2,
    **get_legend_kw(16),
)
save_path = f"{save_dir}/{timestamp}_metrics_nTasks={n_tasks}_agg.png"
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"Plot saved as: {save_path}")


# * plot just ECE and Brier aggregated over tasks
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes = axes.flatten()

for i, metric in enumerate(["ece", "brier"]):
    ax = axes[i]
    invisible_topright_spines(ax)
    for alg, is_al in it.product(algs, is_als):
        alg_isactive = f"{alg}_{is_al}"
        arr = stats_agg[metric][alg_isactive]  # (seeds, steps)
        data_mean = mean(arr, axis=0, nan=handle_nan)  # (steps, )
        data_std = (
            std(arr, axis=0, nan=handle_nan)
            if not use_stderr
            else sterr(arr, axis=0, nan=handle_nan)
        )
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
    ax.set_ylabel(
        prettify_title(metric, all_caps=True)
        if metric == "ece"
        else prettify_title(metric),
        **get_font_kw(18),
    )
fig.supxlabel("Number of Queries", **get_font_kw(16))
fig.legend(
    dummy_lines,
    [get_label(alg, is_al) for alg, is_al in it.product(algs, is_als)],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=len(algs),
    handlelength=2,
    **get_legend_kw(16),
)
save_path = f"{save_dir}/{timestamp}_ECEBrier_nTasks={n_tasks}_agg.png"
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"Plot saved as: {save_path}")
