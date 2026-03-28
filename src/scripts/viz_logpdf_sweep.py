"""
For aggregated logpdf plots over all tasks and seeds, per each algorithm variant.
"""

import itertools as it
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
logging.getLogger("absl").setLevel(logging.WARNING)

import ipdb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

from bnn_pref.utils.plotting import (
    get_font_kw,
    get_legend_kw,
    invisible_topright_spines,
    prettify_title,
    rgb_values,
    set_xlim_offset,
    smooth,
)

task_sets = dict(
    neurips=[
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
    ],
    iclr=[
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
    ],
    unirlhf=[
        "uniCheetahMedium",
        "uniCheetahMediumReplay",
        "uniCheetahMediumExpert",
        "uniHopperMedium",
        "uniHopperMediumReplay",
        "uniHopperMediumExpert",
        "uniWalkerMedium",
        "uniWalkerMediumReplay",
        "uniWalkerMediumExpert",
        "uniPenHuman",
        "uniPenCloned",
    ],
    test=[
        "cheetahMediumReplay",
        "hopperMediumReplay",
        "walkerMediumReplay",
        "penHuman",
        "mazeMediumDense",
    ],
    visual3=[
        "vcheetahMediumExpert",
        "vhumanoidMediumExpert",
        "vwalkerMediumExpert",
    ],
    visual5=[
        # "vcheetahRandom",
        # "vcheetahMediumReplay",
        "vcheetahMediumExpert",
        # "vhumanoidRandom",
        # "vhumanoidMediumReplay",
        "vhumanoidMediumExpert",
        "vwalkerRandom",
        "vwalkerMediumReplay",
        "vwalkerMediumExpert",
    ],
    visual9=[
        "vcheetahRandom",
        "vcheetahMediumReplay",
        "vcheetahMediumExpert",
        "vhumanoidRandom",
        "vhumanoidMediumReplay",
        "vhumanoidMediumExpert",
        "vwalkerRandom",
        "vwalkerMediumReplay",
        "vwalkerMediumExpert",
    ],
)


alg_sets = {
    "all": ["ekf", "sgd", "do", "laplace", "llmcmc"],
    "ekf": ["ekf"],
}


@dataclass
class Args:
    dirp: Path  # for both loading and saving
    task_set: str  # e.g., "neurips", "iclr", "unirlhf", "test", "visual3", "visual5"
    alg_set: str  # e.g., "all", "ekf"
    query_budget: int = -1
    use_stderr: bool = True
    use_smooth: bool = True
    handle_nan: bool = False
    nan_mask: bool = False


args = tyro.cli(Args)
metric_names = ["logpdf", "acc", "ece", "brier", "coverage", "sharpness"]
tasks = task_sets[args.task_set]
n_tasks = len(tasks)
algs = alg_sets[args.alg_set]
is_als = [True, False]
LEGEND_NCOL = 3 if len(algs) == 5 else 1
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print(
    "=====================================================\n"
    f"query_budget: {args.query_budget}\n"
    f"task_set: {args.task_set}\n"
    f"algs: {algs}\n"
    f"load/save dir: {'/'.join(args.dirp.parts[-2:])}\n"
    f"timestamp: {timestamp}\n"
    "=====================================================\n"
)


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
    for folder in os.listdir(args.dirp):
        # if f"task={task}" in folder:
        if re.search(rf"task={re.escape(task)}(?![A-Za-z0-9])", folder):
            path = args.dirp / folder / "stats.npz"
            data[task] = np.load(path, allow_pickle=True)[task].item()


# stats["logpdf"][alg_is_al] = (n_tasks, seeds, steps); e.g. stats["logpdf"]["ekf_True"]
stats = {metric: defaultdict(lambda: list()) for metric in metric_names}
stats["probs"] = defaultdict(lambda: list())
stats["labels"] = defaultdict(lambda: list())

eval_reliability = False
for alg, is_al in it.product(algs, is_als):
    for task in tasks:
        res = data[task][alg][is_al]  # (n_seeds, n_steps)
        for metric in metric_names:
            # stat = res[f"test_{metric}_all"]
            stat = res[f"test_{metric}_all"][:, : args.query_budget]
            stats[metric][f"{alg}_{is_al}"].append(stat)
        if "test_probs_final" in res:
            eval_reliability = True
            # * (n_seeds, Q, 2), (n_seeds, Q, 1)
            test_probs_final = res["test_probs_final"]
            test_labels_final = res["test_labels_final"]
            stats["probs"][f"{alg}_{is_al}"].append(test_probs_final)
            stats["labels"][f"{alg}_{is_al}"].append(test_labels_final)


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
stats_agg["probs"] = {}
stats_agg["labels"] = {}
for alg, is_al in it.product(algs, is_als):
    alg_is_al = f"{alg}_{is_al}"
    for metric in metric_names:
        arr = stats[metric][alg_is_al]  # (n_tasks, seeds, steps)
        stats_agg[metric][alg_is_al] = mean(arr, axis=(0,), nan=args.handle_nan)
    if eval_reliability:
        probs = stats["probs"][alg_is_al]  # [(seeds, Q, 2) * n_tasks]
        labels = stats["labels"][alg_is_al]  # [(seeds, Q, 1) * n_tasks]
        # (n_seeds * n_tasks, Q, ...)
        stats_agg["probs"][alg_is_al] = np.concatenate(
            [a.reshape(-1, 2) for a in probs]
        )
        stats_agg["labels"][alg_is_al] = np.concatenate(
            [a.reshape(-1, 1) for a in labels]
        )


def get_label(alg: str, is_active: Optional[bool] = None) -> str:
    alg2label = {
        "ekf": "PreferenceEKF",
        "sgd": "DeepEnsemble",
        "do": "Dropout",
        "laplace": "Laplace",
        "llmcmc": "LLMCMC",
    }
    alg_label = alg2label[alg]
    if is_active is not None:
        active2label = {True: "A", False: "R"}
        active_label = active2label[is_active]
        return f"{alg_label} ({active_label})"
    else:
        return alg_label


def get_style(alg: str, is_active: bool) -> dict:
    alg2color = {
        "ekf": rgb_values["orange"],
        "sgd": rgb_values["blue"],
        "do": rgb_values["green"],
        "laplace": rgb_values["purple"],
        "llmcmc": rgb_values["gray"],
    }
    linestyle = "-" if is_active else "--"
    return {"color": alg2color[alg], "linestyle": linestyle}


class DualLineHandler(HandlerBase):
    """Custom legend handler that creates two lines (solid and dashed) for a single legend entry."""

    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        # orig_handle is a tuple: (color, label)
        color, _ = orig_handle
        common = {"color": color, "linewidth": 2, "transform": trans}
        # Create two lines: solid on top, dashed below
        l1 = Line2D(
            [x0, x0 + width],
            [0.7 * height, 0.7 * height],
            linestyle="-",
            **common,
        )
        l2 = Line2D(
            [x0, x0 + width],
            [0.3 * height, 0.3 * height],
            linestyle="--",
            **common,
        )
        return [l1, l2]


# Reorder function to fill rows first (left-to-right) instead of columns first
# Based on: https://stackoverflow.com/questions/66783109/matplotlibs-legend-how-to-order-entries-by-row-first-rather-than-by-column
reorder_legend_row_major = lambda lst, nc: sum((lst[i::nc] for i in range(nc)), [])


def plot_logpdf_agg():
    fig, ax = plt.subplots(figsize=(10, 6))
    invisible_topright_spines(ax)
    for alg, is_al in it.product(algs, is_als):
        alg_isactive = f"{alg}_{is_al}"
        arr = stats_agg["logpdf"][alg_isactive]  # (seeds, steps)
        data_mean = mean(arr, axis=0, nan=args.handle_nan)
        data_std = (
            std(arr, axis=0, nan=args.handle_nan)
            if not args.use_stderr
            else sterr(arr, axis=0, nan=args.handle_nan)
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
        ekf_active_mean_T = mean(
            stats_agg["logpdf"]["ekf_True"], axis=0, nan=args.handle_nan
        )
        ekf_random_mean_T = mean(
            stats_agg["logpdf"]["ekf_False"], axis=0, nan=args.handle_nan
        )

        if ekf_active_mean_T[-1] > ekf_random_mean_T[-1]:
            # Find the y-value at the last step of EKF (Random)
            y_tgt = ekf_random_mean_T[-1]
            x_random = len(ekf_random_mean_T) - 1

            # Find the first x in EKF (Active) that reaches or exceeds y_target
            x_active = np.argmax(ekf_active_mean_T >= y_tgt)

            frac = 1 - x_active / x_random

            # Get y-values on EKF active line at the annotation points
            y_active_at_x = ekf_active_mean_T[x_active]
            y_active_at_end = ekf_active_mean_T[x_random]
            y_top = (
                max(y_active_at_x, y_active_at_end) + 0.03
            )  # Position above the EKF line

            # Draw vertical dotted lines up from the EKF line to the top for annotation
            ax.vlines(
                [x_active, x_random], y_tgt, y_top, linestyles="dotted", colors="k"
            )
            ax.plot(
                [x_active, x_random], [y_tgt, y_tgt], "ko", markersize=4
            )  # mark the two points

            # Draw double-headed arrow and annotate at the top
            ax.annotate(
                "",
                xy=(x_active, y_top),
                xytext=(x_random, y_top),
                arrowprops=dict(
                    arrowstyle="<->", color="black", linewidth=1.5, shrinkA=0, shrinkB=0
                ),
            )
            ax.text(
                (x_active + x_random) / 2,
                y_top + 0.01,  # slightly above the arrow
                f"~{frac:.0%} fewer samples",
                ha="center",
                va="bottom",
                color="black",
                **get_font_kw(18),
            )

    ax.set_xlabel("Number of Queries", **get_font_kw(18))
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x):d}" for x in xticks], **get_font_kw(16))
    set_xlim_offset(ax)
    n_queries_xlim = len(data_mean) + 0.5  # n_queries + 0.5
    ax.set_xlim(right=n_queries_xlim)  # Cut off the graph

    ax.set_ylabel("Test Log-Likelihood", **get_font_kw(18))
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.2f}" for y in yticks], **get_font_kw(16))

    # Create custom legend with both solid and dashed lines for each algorithm
    legend_handles = []
    legend_labels = []
    handler_map = {}
    for alg in algs:
        alg_color = get_style(alg, True)["color"]  # Get color (same for both A/R)
        alg_label = get_label(alg)  # Get label only
        # Create a tuple handle that the handler will use
        handle = (alg_color, alg_label)
        legend_handles.append(handle)
        legend_labels.append(alg_label)
        handler_map[tuple] = DualLineHandler()

    # Reorder to fill rows first (left-to-right, then top-to-bottom)
    legend_handles = reorder_legend_row_major(legend_handles, LEGEND_NCOL)
    legend_labels = reorder_legend_row_major(legend_labels, LEGEND_NCOL)

    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map=handler_map,
        **get_legend_kw(18),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=LEGEND_NCOL,
    )
    save_path = (
        args.dirp
        / f"{timestamp}_0_logpdf_nTasks={n_tasks}_agg_Q={args.query_budget}.png"
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    print("Saved logpdf agg")


def plot_logpdf_per_task():
    if n_tasks == 3:
        n_rows = 1
        n_cols = 3
    elif n_tasks == 5:
        n_rows = 2
        n_cols = 3
    elif n_tasks == 9:
        n_rows = 3
        n_cols = 3
    else:
        n_rows = 3
        n_cols = 4
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=True
    )
    axs = axs.flatten()

    for i, task in enumerate(tasks):
        ax = axs[i]
        invisible_topright_spines(ax)
        # ax.axhline(y=-0.69, linestyle=":", linewidth=1, color="red")  # ln(0.5) = -0.69
        # y_lim_min, y_lim_max = -0.73, 0
        for alg, is_al in it.product(algs, is_als):
            arr = stats["logpdf"][f"{alg}_{is_al}"]  # (tasks, seeds, steps)
            arr_task = arr[i, :, :]  # (seeds, steps)
            mean_E = mean(arr_task, axis=0, nan=args.handle_nan)  # (steps, )
            std_E = (
                std(arr_task, axis=0, nan=args.handle_nan)
                if not args.use_stderr
                else sterr(arr_task, axis=0, nan=args.handle_nan)
            )  # (steps, )
            mean_E = smooth(mean_E) if args.use_smooth else mean_E
            std_E = smooth(std_E) if args.use_smooth else std_E
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
        if "v" in task:  # vwalkerMediumExpert -> walkerMediumExpert
            task = task.replace("v", "")
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

    # --- Shared legend using custom handler with both line styles ---
    legend_handles = []
    legend_labels = []
    handler_map = {}
    for alg in algs:
        alg_color = get_style(alg, True)["color"]
        alg_label = get_label(alg)
        handle = (alg_color, alg_label)
        legend_handles.append(handle)
        legend_labels.append(alg_label)
        handler_map[tuple] = DualLineHandler()

    # Reorder to fill rows first (left-to-right, then top-to-bottom)
    legend_handles = reorder_legend_row_major(legend_handles, LEGEND_NCOL)
    legend_labels = reorder_legend_row_major(legend_labels, LEGEND_NCOL)

    fig.supxlabel("Number of Queries", **get_font_kw(16))
    fig.supylabel("Test Log-Likelihood", **get_font_kw(16))
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map=handler_map,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=LEGEND_NCOL,
        handlelength=2,
        **get_legend_kw(16),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_path = (
        args.dirp / f"{timestamp}_1_logpdf_nTasks={n_tasks}_Q={args.query_budget}.png"
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved logpdf per task")


def plot_ece_brier_agg():
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes = axes.flatten()

    for i, metric in enumerate(["ece", "brier"]):
        ax = axes[i]
        invisible_topright_spines(ax)
        for alg, is_al in it.product(algs, is_als):
            alg_isactive = f"{alg}_{is_al}"
            arr = stats_agg[metric][alg_isactive]  # (seeds, steps)
            data_mean = mean(arr, axis=0, nan=args.handle_nan)  # (steps, )
            data_std = (
                std(arr, axis=0, nan=args.handle_nan)
                if not args.use_stderr
                else sterr(arr, axis=0, nan=args.handle_nan)
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
    # Create custom legend
    legend_handles = []
    legend_labels = []
    handler_map = {}
    for alg in algs:
        alg_color = get_style(alg, True)["color"]
        alg_label = get_label(alg)
        handle = (alg_color, alg_label)
        legend_handles.append(handle)
        legend_labels.append(alg_label)
        handler_map[tuple] = DualLineHandler()

    # Reorder to fill rows first (left-to-right, then top-to-bottom)
    legend_handles = reorder_legend_row_major(legend_handles, LEGEND_NCOL)
    legend_labels = reorder_legend_row_major(legend_labels, LEGEND_NCOL)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map=handler_map,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=LEGEND_NCOL,
        handlelength=2,
        **get_legend_kw(16),
    )
    save_path = (
        args.dirp
        / f"{timestamp}_2_ECEBrier_nTasks={n_tasks}_agg_Q={args.query_budget}.png"
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved ECE/Brier agg")


def plot_all_metrics_agg():
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        invisible_topright_spines(ax)
        for alg, is_al in it.product(algs, is_als):
            alg_isactive = f"{alg}_{is_al}"
            arr = stats_agg[metric][alg_isactive]  # (seeds, steps)
            data_mean = mean(arr, axis=0, nan=args.handle_nan)  # (steps, )
            data_std = (
                std(arr, axis=0, nan=args.handle_nan)
                if not args.use_stderr
                else sterr(arr, axis=0, nan=args.handle_nan)
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
    # Create custom legend
    legend_handles = []
    legend_labels = []
    handler_map = {}
    for alg in algs:
        alg_color = get_style(alg, True)["color"]
        alg_label = get_label(alg)
        handle = (alg_color, alg_label)
        legend_handles.append(handle)
        legend_labels.append(alg_label)
        handler_map[tuple] = DualLineHandler()

    # Reorder to fill rows first (left-to-right, then top-to-bottom)
    legend_handles = reorder_legend_row_major(legend_handles, LEGEND_NCOL)
    legend_labels = reorder_legend_row_major(legend_labels, LEGEND_NCOL)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map=handler_map,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=LEGEND_NCOL,
        handlelength=2,
        **get_legend_kw(16),
    )
    save_path = (
        args.dirp
        / f"{timestamp}_3_metrics_nTasks={n_tasks}_agg_Q={args.query_budget}.png"
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved all metrics agg")


def plot_reliability_diagram(n_bins: int = 10):
    """
    Plot reliability diagram with stacked bars (accuracy and gap) for each algorithm.

    Creates a bar chart showing:
    - Blue bars: Accuracy within each confidence bin
    - Yellow bars: Gap (difference between confidence and accuracy)
    - Diagonal line: Perfect calibration reference
    - ECE value: Displayed on each plot

    Uses confidence (max probability) for binning, matching the ECE computation.
    Layout: 5 columns x 2 rows (top row: active, bottom row: random)
    """

    # Create 5x2 subplot layout: top row = active, bottom row = random
    n_cols = len(algs)
    n_rows = 2  # Active and Random
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Create ordered list: all active first, then all random
    alg_order = []
    for is_al in [True, False]:  # Active first, then Random
        for alg in algs:
            alg_order.append((alg, is_al))

    for idx, (alg, is_al) in enumerate(alg_order):
        ax = axes[idx]
        invisible_topright_spines(ax)

        alg_isactive = f"{alg}_{is_al}"
        probs = stats_agg["probs"][alg_isactive]  # (n_seeds * n_tasks * Q, 2)
        labels = stats_agg["labels"][alg_isactive]  # (n_seeds * n_tasks * Q, 1)

        # Flatten labels if needed
        if labels.ndim > 1:
            labels = labels.squeeze()

        # Use confidence (max probability) for binning, matching ECE computation
        # This is the probability of the predicted class
        confidence = np.max(probs, axis=1)  # (n_samples,)
        predictions = np.argmax(probs, axis=1)  # (n_samples,)

        # For calibration curve, we need the probability of the true class
        # But we'll bin by confidence and check if predictions are correct
        correct = (predictions == labels).astype(float)

        # Filter out invalid values
        valid_mask = (
            np.isfinite(confidence) & np.isfinite(labels) & np.isfinite(correct)
        )
        confidence = confidence[valid_mask]
        correct = correct[valid_mask]
        labels = labels[valid_mask]

        # # Check if we have both classes
        # unique_labels = np.unique(labels)
        # if len(unique_labels) == 1:
        #     # Still plot, but it won't be meaningful for calibration assessment
        #     print(
        #         f"Warning: {alg_isactive} has only one class (labels={unique_labels}). "
        #         f"Cannot properly assess calibration. All labels are {unique_labels[0]}."
        #     )

        # For reliability diagram, we bin by confidence and compute accuracy in each bin
        # This matches the ECE computation approach
        # We'll use manual binning to match ECE exactly
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        prob_true = []
        prob_pred = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            bin_mask = (bin_lower < confidence) & (confidence <= bin_upper)
            bin_count = np.sum(bin_mask)

            if bin_count > 0:
                # Accuracy in this bin (fraction of correct predictions)
                bin_acc = np.mean(correct[bin_mask])
                # Mean confidence in this bin
                bin_conf = np.mean(confidence[bin_mask])
                prob_true.append(bin_acc)
                prob_pred.append(bin_conf)
            else:
                # Empty bin - use bin center
                prob_true.append(0.0)
                prob_pred.append((bin_lower + bin_upper) / 2)

        prob_true = np.array(prob_true)
        prob_pred = np.array(prob_pred)

        # Compute ECE
        ece_arr = stats_agg["ece"][alg_isactive]  # (seeds, steps)
        ece_mean = mean(ece_arr[:, -1:], axis=0, nan=args.handle_nan)[0]

        # Bin boundaries for plotting
        bin_width = 1.0 / n_bins
        bin_centers = prob_pred

        # Accuracy (blue bars) - fraction of positives in each bin
        accuracy = prob_true

        # Confidence (mean predicted probability in each bin)
        confidence = prob_pred

        # Gap (yellow bars) - difference between confidence and accuracy
        gap = confidence - accuracy

        # Plot stacked bars
        x_pos = bin_centers
        width = bin_width * 0.8  # Slightly narrower bars

        # Blue bars: Accuracy
        ax.bar(
            x_pos, accuracy, width=width, label="Outputs", color="#1f77b4", alpha=0.8
        )

        # Yellow bars: Gap (stacked on top of accuracy)
        ax.bar(
            x_pos,
            gap,
            width=width,
            bottom=accuracy,
            label="Gap",
            color="#ffbb78",
            alpha=0.8,
        )

        # Diagonal line for perfect calibration
        ax.plot(
            [0, 1], [0, 1], "k--", linewidth=2, alpha=0.7, label="Perfectly calibrated"
        )

        # Set labels and limits
        ax.set_xlabel("Confidence", **get_font_kw(14))
        ax.set_ylabel("Accuracy (%)", **get_font_kw(14))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Set ticks and convert y-axis to percentage
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(y * 100)}" for y in yticks], **get_font_kw(12))
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks], **get_font_kw(12))

        # Add ECE value as text box
        label = get_label(alg, is_al)
        ax.text(
            0.02,
            0.98,
            f"ECE={ece_mean:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Title
        ax.set_title(label, **get_font_kw(14))

        # Legend (only on first subplot)
        if idx == 0:
            ax.legend(**get_legend_kw(12), loc="upper left", bbox_to_anchor=(0.0, 0.9))

        ax.grid(True, alpha=0.3, axis="y")

    # Adjust spacing to fix gap issue
    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.8)
    save_path = (
        args.dirp
        / f"{timestamp}_4_reliability_nTasks={n_tasks}_agg_Q={args.query_budget}.png"
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved reliability diagram")


def _stderr_of_mean_1d(a: np.ndarray) -> float:
    """Standard error of the mean for samples along the only axis; a has shape (S,)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    n = a.shape[0]
    if n <= 1:
        return 0.0
    return float(np.std(a, ddof=1) / np.sqrt(n))


def _duration_per_seed_s(task_stats: dict, alg: str, is_active: bool) -> np.ndarray:
    """
    Per-seed training time in seconds for one task and algorithm variant.
    run_rm saves `duration` (from trainer `train_duration`); older runs may use `train_duration`.
    Returns (S,) with S = number of seeds.
    """
    res = task_stats[alg][is_active]
    if isinstance(res, np.ndarray) and res.dtype == object:
        res = res.item()
    if isinstance(res, dict):
        raw = res.get("duration")
        if raw is None:
            raw = res.get("train_duration")
    else:
        raw = None
    if raw is None:
        raise KeyError(
            f"Missing duration/train_duration for alg={alg}, is_active={is_active}"
        )
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    return arr  # (S,)


def plot_train_duration_bars(include_llmcmc: bool = True) -> None:
    """
    Grouped bar chart: x = one tick per algorithm; two bars per group (active vs random).
    Heights: mean training duration (seconds), mean over tasks then mean over seeds; error bars
    are stderr over seeds of the task-averaged duration (same convention as line-plot stderr).
    """
    # (N_alg,) — one mean height per algorithm for active / random
    means_active: list[float] = []
    means_random: list[float] = []
    errs_active: list[float] = []
    errs_random: list[float] = []

    nalgs = algs
    nalgs = [alg for alg in nalgs if alg != "llmcmc" or include_llmcmc]
    for alg in nalgs:
        if alg == "llmcmc" and not include_llmcmc:
            continue
        # D_true: (T, S) — T tasks, S seeds; seconds per run
        D_true = np.stack(
            [_duration_per_seed_s(data[task], alg, True) for task in tasks],
            axis=0,
        )
        D_false = np.stack(
            [_duration_per_seed_s(data[task], alg, False) for task in tasks],
            axis=0,
        )
        # Per-seed mean over tasks: (S,)
        per_seed_a = np.mean(D_true, axis=0)
        per_seed_r = np.mean(D_false, axis=0)
        means_active.append(float(np.mean(per_seed_a)))
        means_random.append(float(np.mean(per_seed_r)))
        # stderr of mean over seeds: std(S) / sqrt(S); (S,) -> scalar
        errs_active.append(_stderr_of_mean_1d(per_seed_a))
        errs_random.append(_stderr_of_mean_1d(per_seed_r))

    # N_alg: number of algorithms in this sweep
    N_alg = len(nalgs)
    x = np.arange(N_alg, dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    invisible_topright_spines(ax)

    hatch_random = "/"
    for i, alg in enumerate(nalgs):
        if alg == "llmcmc" and not include_llmcmc:
            continue
        color = get_style(alg, True)["color"]
        ax.bar(
            x[i] - width / 2,
            means_active[i],
            width,
            color=color,
            yerr=errs_active[i],
            capsize=3,
            error_kw={"linewidth": 1.2},
        )
        ax.bar(
            x[i] + width / 2,
            means_random[i],
            width,
            facecolor=color,
            hatch=hatch_random,
            edgecolor="black",
            linewidth=0.6,
            yerr=errs_random[i],
            capsize=3,
            error_kw={"linewidth": 1.2},
        )

    ax.set_xticks(x)
    labels = [get_label(alg) for alg in nalgs]
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Training duration (s)", **get_font_kw(18))
    ax.set_xlabel("Algorithm", **get_font_kw(18))
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.0f}" for y in yticks], **get_font_kw(16))

    legend_handles = [
        mpatches.Patch(facecolor="#666666", edgecolor="none", label="Active"),
        mpatches.Patch(
            facecolor="#666666",
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch_random,
            label="Random",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        **get_legend_kw(16),
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )

    save_path = (
        args.dirp
        / f"{timestamp}_6_train_duration_nTasks={n_tasks}_agg_Q={args.query_budget}_mcmc={include_llmcmc}.png"
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved train duration bar plot")


def compute_2sample_t_test():
    """
    stats["logpdf"]["laplace_True"] = array(n_tasks, seeds, steps)
    stats_agg["logpdf"]["laplace_True"] = array(seeds, steps)

    AUC: np.trapz(array.mean(0))

    d["laplace_True"] = AUC (Scalar)

    """
    import jax
    from prettytable import PrettyTable
    from scipy.stats import ttest_rel

    np.set_printoptions(precision=4)

    leaves, _ = jax.tree.flatten(stats_agg["logpdf"])  # list of (seeds, steps)
    min_logpdf = np.min(leaves)
    max_logpdf = 0  # logpdf theoretical maximum
    n_queries = stats_agg["logpdf"]["ekf_True"].shape[1]
    max_auc = (max_logpdf - min_logpdf) * n_queries

    # stats_agg["logpdf"] -> aucs["alg_is_al"] = (seeds, )
    scores = {}
    for alg, is_al in it.product(algs, is_als):
        name = f"{alg}_{is_al}"
        arr = stats_agg["logpdf"][name]  # (seeds, steps)

        auc_score = np.trapz(arr - min_logpdf, axis=1) / max_auc  # (seeds, )
        scores[name] = auc_score
        # logpdf_score = (arr[:, -1] - min_logpdf) / (max_logpdf - min_logpdf)  # (s, )
        # scores[name] = 0.9 * auc_score + 0.1 * logpdf_score

    eces = {}
    for alg, is_al in it.product(algs, is_als):
        name = f"{alg}_{is_al}"
        arr = stats_agg["ece"][name]  # (seeds, steps)
        eces[name] = arr[:, -1]

    def cohens_d(arr1, arr2):
        """
        Compute Cohen's d effect size for two independent samples.
        Uses pooled standard deviation for Welch's t-test (unequal variances).

        Interpretation of Cohen's d:
            |d| < 0.2: negligible
            0.2 ≤ |d| < 0.5: small
            0.5 ≤ |d| < 0.8: medium
            |d| ≥ 0.8: large
        """
        mean1, mean2 = np.mean(arr1), np.mean(arr2)
        std1, std2 = np.std(arr1, ddof=1), np.std(arr2, ddof=1)
        n1, n2 = len(arr1), len(arr2)

        # Pooled standard deviation (weighted by sample size)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        d = (mean1 - mean2) / pooled_std

        val = abs(d)
        if val < 0.2:
            effect_size = "negligible"
        elif val < 0.5:
            effect_size = "small"
        elif val < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        return d, effect_size

    def do_ttest(arr1, arr2, alternative="greater"):
        """
        arr1, arr2: (seeds, )

        ttest_ind: independent samples (different seeds)
        ttest_rel: paired samples (same seeds) -> is the way to go
        """

        result = ttest_rel(arr1, arr2, alternative=alternative)
        t_stat = result.statistic
        p_value = result.pvalue
        ci = result.confidence_interval(confidence_level=0.95)
        d, effect_size = cohens_d(arr1, arr2)
        return t_stat, p_value, ci, d, effect_size

    table_names = {
        "ekf": "EKF",
        "sgd": "Ensemble",
        "do": "Dropout",
        "laplace": "Laplace",
        "llmcmc": "LLMCMC",
    }

    print(
        "\n"
        "================================================================================\n"
        "t-test: LogPDF active vs. random for each algorithm, aggregated over all tasks\n"
        "================================================================================"
    )
    table = PrettyTable()
    table.field_names = ["Comparison", "t", "p-value", "Cohen's d", "95% CI"]
    table.align["Comparison"] = "l"
    # * logpdf: active vs. random for each algorithm
    for alg in algs:
        name_active = f"{alg}_True"
        name_random = f"{alg}_False"
        arr_active = scores[name_active]  # (seeds, )
        arr_random = scores[name_random]  # (seeds, )
        t_stat, p_value, ci, d, effect_size = do_ttest(
            arr_active, arr_random, alternative="greater"
        )
        row = [
            f"{table_names[alg]} (A vs. R)",
            f"{t_stat:.2f}",
            f"{p_value:.3f}",
            f"{d:.2f} ({effect_size})",
            f"({ci.low:.2f}, {ci.high:.2f})",
        ]
        table.add_row(row)

    # * logpdf: Active EKF vs. other active algorithms
    for alg in algs[1:]:
        name_ekf = "ekf_True"
        name_alg = f"{alg}_True"
        arr_ekf = scores[name_ekf]  # (seeds, )
        arr_alg = scores[name_alg]  # (seeds, )
        t_stat, p_value, ci, d, effect_size = do_ttest(
            arr_ekf, arr_alg, alternative="greater"
        )
        row = [
            f"EKF vs. {table_names[alg]}",
            f"{t_stat:.2f}",
            f"{p_value:.3f}",
            f"{d:.2f} ({effect_size})",
            f"({ci.low:.2f}, {ci.high:.2f})",
        ]
        table.add_row(row)
    print(table)

    print(
        "\n"
        "================================================================================\n"
        "t-test: ECE active vs. random for each algorithm, aggregated over all tasks\n"
        "================================================================================"
    )
    table = PrettyTable()
    table.field_names = ["Comparison", "t", "p-value", "Cohen's d", "95% CI"]
    table.align["Comparison"] = "l"
    # * ECE: active vs. random for each algorithm
    for alg in algs:
        name_active = f"{alg}_True"
        name_random = f"{alg}_False"
        arr_active = eces[name_active]  # (seeds, )
        arr_random = eces[name_random]  # (seeds, )
        t_stat, p_value, ci, d, effect_size = do_ttest(
            arr_active, arr_random, alternative="less"
        )
        row = [
            f"{table_names[alg]} (A vs. R)",
            f"{t_stat:.2f}",
            f"{p_value:.3f}",
            f"{d:.2f} ({effect_size})",
            f"({ci.low:.2f}, {ci.high:.2f})",
        ]
        table.add_row(row)

    # * ECE: Active EKF vs. other active algorithms
    for alg in algs[1:]:
        name_ekf = "ekf_True"
        name_alg = f"{alg}_True"
        arr_ekf = eces[name_ekf]  # (seeds, )
        arr_alg = eces[name_alg]  # (seeds, )
        t_stat, p_value, ci, d, effect_size = do_ttest(
            arr_ekf, arr_alg, alternative="less"
        )
        row = [
            f"EKF vs. {table_names[alg]}",
            f"{t_stat:.2f}",
            f"{p_value:.3f}",
            f"{d:.2f} ({effect_size})",
            f"({ci.low:.2f}, {ci.high:.2f})",
        ]
        table.add_row(row)

    print(table)

    save_path = (
        args.dirp
        / f"{timestamp}_5_ttest_nTasks={n_tasks}_agg_Q={args.query_budget}.txt"
    )
    with open(save_path, "w") as f:
        f.write(table.get_string())


plot_logpdf_agg()
plot_logpdf_per_task()
plot_ece_brier_agg()
plot_all_metrics_agg()
plot_reliability_diagram()
plot_train_duration_bars(include_llmcmc=True)
plot_train_duration_bars(include_llmcmc=False)
compute_2sample_t_test()
