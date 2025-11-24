"""
For aggregated logpdf plots over all tasks and seeds, per each algorithm variant.
"""

import itertools as it
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Optional

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
logging.getLogger("absl").setLevel(logging.WARNING)

import ipdb
import matplotlib.pyplot as plt
import numpy as np
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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dirp = sys.argv[1]  # where to load
task_set = sys.argv[2]  # "neurips", "iclr", "visual"
save_dir = dirp  # where to save
metric_names = ["logpdf", "acc", "ece", "brier", "coverage", "sharpness"]
use_stderr = True  # otherwise use stderr
use_smooth = True
handle_nan = False
nan_mask = False


# neurips tasks
neurips_tasks = [
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
iclr_tasks = [
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

visual_tasks = [
    "vcheetahMediumExpert",
    "vhumanoidMediumExpert",
    "vwalkerMediumExpert",
]

visual_tasks = [
    # "vcheetahRandom",
    # "vcheetahMediumReplay",
    "vcheetahMediumExpert",
    # "vhumanoidRandom",
    # "vhumanoidMediumReplay",
    "vhumanoidMediumExpert",
    "vwalkerRandom",
    "vwalkerMediumReplay",
    "vwalkerMediumExpert",
]

task_select = {
    "neurips": neurips_tasks,
    "iclr": iclr_tasks,
    "visual": visual_tasks,
}


tasks = task_select[task_set]
n_tasks = len(tasks)
algs = ["ekf", "sgd", "do", "laplace", "llmcmc"] if task_set == "iclr" else ["ekf"]
is_als = [True, False]
LEGEND_NCOL = 3 if len(algs) == 5 else 1


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
        ekf_active_mean_T = mean(
            stats_agg["logpdf"]["ekf_True"], axis=0, nan=handle_nan
        )
        ekf_random_mean_T = mean(
            stats_agg["logpdf"]["ekf_False"], axis=0, nan=handle_nan
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
    save_path = f"{save_dir}/{timestamp}_0_logpdf_nTasks={n_tasks}_agg.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved as: {save_path}")


def plot_logpdf_per_task():
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
    save_path = f"{save_dir}/{timestamp}_1_logpdf_nTasks={n_tasks}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Plot saved as: {save_path}")


def plot_ece_brier_agg():
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
    save_path = f"{save_dir}/{timestamp}_2_ECEBrier_nTasks={n_tasks}_agg.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Plot saved as: {save_path}")


def plot_all_metrics_agg():
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
    save_path = f"{save_dir}/{timestamp}_3_metrics_nTasks={n_tasks}_agg.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Plot saved as: {save_path}")


def compute_2sample_t_test():
    """
    stats["logpdf"]["laplace_True"] = array(n_tasks, seeds, steps)
    stats_agg["logpdf"]["laplace_True"] = array(seeds, steps)

    AUC: np.trapz(array.mean(0))

    d["laplace_True"] = AUC (Scalar)

    """
    import jax
    from prettytable import PrettyTable
    from scipy.stats import ttest_ind

    np.set_printoptions(precision=4)

    leaves, _ = jax.tree.flatten(stats_agg["logpdf"])  # list of (seeds, steps)
    min_value = np.min(leaves)
    max_value = 0  # logpdf theoretical maximum
    n_queries = stats_agg["logpdf"]["ekf_True"].shape[1]
    max_auc = (max_value - min_value) * n_queries

    # stats_agg["logpdf"] -> aucs["alg_is_al"] = (seeds, )
    aucs = {}
    for alg, is_al in it.product(algs, is_als):
        name = f"{alg}_{is_al}"
        arr = stats_agg["logpdf"][name]  # (seeds, steps)

        arr_shifted = arr - min_value
        auc_score = np.trapz(arr_shifted, axis=1)
        auc_score = auc_score / max_auc
        aucs[name] = auc_score

    def cohens_d(arr1, arr2):
        """
        Compute Cohen's d effect size for two independent samples.
        Uses pooled standard deviation for Welch's t-test (unequal variances).
        """
        mean1, mean2 = np.mean(arr1), np.mean(arr2)
        std1, std2 = np.std(arr1, ddof=1), np.std(arr2, ddof=1)
        n1, n2 = len(arr1), len(arr2)

        # Pooled standard deviation (weighted by sample size)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        d = (mean1 - mean2) / pooled_std
        return d

    def interpret_effect_size(d):
        """
        Cohen's d:
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large
        """
        if abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small"
        elif abs(d) < 0.8:
            return "medium"
        else:
            return "large"

    def compute_stats(arr1, arr2):
        result = ttest_ind(arr1, arr2, equal_var=False, alternative="greater")
        t_stat = result.statistic
        p_value = result.pvalue
        ci = result.confidence_interval()
        d = cohens_d(arr1, arr2)
        effect_size = interpret_effect_size(d)
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
        "2-sample t-test: active vs. random for each algorithm, aggregated over all tasks\n"
        "================================================================================"
    )
    table = PrettyTable()
    table.field_names = ["Active vs. Random", "t", "p-value", "Cohen's d", "95% CI"]
    table.align["Active vs. Random"] = "l"
    for alg in algs:
        name_active = f"{alg}_True"
        name_random = f"{alg}_False"
        arr_active = aucs[name_active]  # (seeds, )
        arr_random = aucs[name_random]  # (seeds, )
        t_stat, p_value, ci, d, effect_size = compute_stats(arr_active, arr_random)
        row = [
            table_names[alg],
            f"{t_stat:.2f}",
            f"{p_value:.4f}",
            f"{d:.2f} ({effect_size})",
            f"({ci.low:.2f}, {ci.high:.2f})",
        ]
        table.add_row(row)
    print(table)

    print(
        "\n"
        "================================================================================\n"
        "2-sample t-test: Active EKF vs. other active algorithms, aggregated over all tasks\n"
        "================================================================================"
    )
    table = PrettyTable()
    table.field_names = ["EKF vs.", "t", "p-value", "Cohen's d", "95% CI"]
    table.align["EKF vs."] = "l"
    for alg in algs[1:]:
        name_ekf = "ekf_True"
        name_alg = f"{alg}_True"
        arr_ekf = aucs[name_ekf]  # (seeds, )
        arr_alg = aucs[name_alg]  # (seeds, )
        t_stat, p_value, ci, d, effect_size = compute_stats(arr_ekf, arr_alg)
        row = [
            f"EKF vs. {table_names[alg]}",
            f"{t_stat:.2f}",
            f"{p_value:.4f}",
            f"{d:.2f} ({effect_size})",
            f"({ci.low:.2f}, {ci.high:.2f})",
        ]
        table.add_row(row)
    print(table)


plot_logpdf_agg()
plot_logpdf_per_task()
plot_ece_brier_agg()
plot_all_metrics_agg()
compute_2sample_t_test()
