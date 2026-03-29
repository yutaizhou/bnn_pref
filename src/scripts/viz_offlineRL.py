import itertools as it
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import ipdb
import jax.numpy as jnp
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
from bnn_pref.utils.task_sets import alg_sets, task_sets

pref_dirp = "/scr/yutaizho/code/p-prefEKF/bnn_pref/_runs/offline_rl/20251126_iql_pref_12tasks_nitersUpdate=10_acq=infogain"
ref_dirp = "/scr/yutaizho/code/p-prefEKF/bnn_pref/_runs/offline_rl/20250501_002013_iql_ref_18tasks"


@dataclass
class Args:
    pref_dirp: Path = Path(pref_dirp)
    ref_dirp: Path = Path(ref_dirp)
    task_set: str = "iclr"
    alg_set: str = "all"
    use_stderr: bool = True
    use_smooth: bool = True


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
args = tyro.cli(Args)
tasks = task_sets[args.task_set]
algs = alg_sets[args.alg_set]
is_als = [True, False]
save_dir = args.pref_dirp


def defaultdict2dict(dd):
    return {k: defaultdict2dict(v) if isinstance(v, dict) else v for k, v in dd.items()}


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


# Based on matplotlib legend ordering trick: reorder entries row-major (left-to-right).
def reorder_legend_row_major(lst, n_cols):
    return sum((lst[i::n_cols] for i in range(n_cols)), [])


def main():
    baseline_scores = get_baseline_score(args.ref_dirp, tasks)  # d[task]["zero", "gt"]

    dir_name = args.pref_dirp.name  # "iql_pref_18tasks_nq60_5seed"
    aux_fname = dir_name.split("iql_pref_12tasks_")[-1]  # gets "_nq60_5seed"

    # pref_scores[task][ekf_False] # (n_evals+1, n_pref_dirps)
    # agg_scores[ekf_False] # (n_evals+1, n_pref_dirps)
    pref_scores = combine_pref_scores(args.pref_dirp)
    agg_scores = aggregate_scores_task(pref_scores)
    n_seeds = agg_scores[f"{algs[0]}_{is_als[0]}"].shape[1]

    fig, axs = plt.subplots(3, 4, figsize=(12, 7.5), sharex=True)
    axs = axs.flatten()
    legend_handles = []
    legend_labels = []
    handler_map = {tuple: DualLineHandler()}
    baseline_lines = ()

    # * plot scores per task
    for i, task in enumerate(tasks):
        ax = axs[i]
        invisible_topright_spines(ax)
        max_score, min_score = 0, jnp.inf
        zero_score = baseline_scores[task]["zero"]
        gt_score = baseline_scores[task]["gt"]
        for alg, is_al in it.product(algs, is_als):
            scores = pref_scores[task][f"{alg}_{is_al}"]  # (n_evals+1, n_eval_workers)
            mean_E = scores.mean(1)
            std_E = (
                scores.std(1)
                if not args.use_stderr
                else scores.std(1) / jnp.sqrt(scores.shape[1])
            )
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
            # for better zooming in visualization
            if task in ["cheetahRandom", "mazeMediumDense", "hopperMediumExpert"]:
                max_score = max(scores[np.isfinite(scores)].max(), gt_score, max_score)
                min_score = min(
                    scores[np.isfinite(scores)].min(), zero_score, min_score
                )
            else:
                max_score = max(mean_E[np.isfinite(mean_E)].max(), gt_score, max_score)
                min_score = min(
                    mean_E[np.isfinite(mean_E)].min(), zero_score, min_score
                )

        zero_line = ax.axhline(
            zero_score, color=rgb_values["gray"], linestyle="--", linewidth=1.0
        )
        gt_line = ax.axhline(
            gt_score, color=rgb_values["gray"], linestyle="-", linewidth=1.0
        )

        ax.set_ylim(min_score - 2, max_score + 2)
        if i == 0:  # Only store baseline legend info from first subplot
            baseline_lines = (gt_line, zero_line)

        ax.set_title(prettify_title(task), **get_font_kw(14))

        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(x):d}" for x in xticks], **get_font_kw(12))
        set_xlim_offset(ax)
        ax.set_xlim(right=len(mean_E))  # Cut off the graph

        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.0f}" for y in yticks], **get_font_kw(12))

    fig.supxlabel("Evaluation Steps", **get_font_kw(16))
    fig.supylabel("Normalized Score", **get_font_kw(16))
    for alg in algs:
        alg_color = get_style(alg, True)["color"]
        alg_label = get_label(alg)
        legend_handles.append((alg_color, alg_label))
        legend_labels.append(alg_label)

    legend_handles.extend(baseline_lines)
    legend_labels.extend(["GT", "Zero"])

    # legend_handles_ordered = legend_handles  # old ordering
    legend_handles_ordered = reorder_legend_row_major(legend_handles, 4)
    # legend_labels_ordered = legend_labels  # old ordering
    legend_labels_ordered = reorder_legend_row_major(legend_labels, 4)

    fig.legend(
        legend_handles_ordered,
        legend_labels_ordered,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        handlelength=2,
        handler_map=handler_map,
        **get_legend_kw(16),
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    fp = f"{save_dir}/offlineRL_{timestamp}_{aux_fname}_{n_seeds}seeds.png"
    plt.savefig(fp, bbox_inches="tight", dpi=300)
    print(f"Saved to {fp}")

    # * plot agg_scores
    fig, ax = plt.subplots(figsize=(10, 6))
    invisible_topright_spines(ax)
    max_score = 0
    min_score = jnp.inf
    agg_gt = baseline_scores["agg_gt"]
    agg_zero = baseline_scores["agg_zero"]
    for alg, is_al in it.product(algs, is_als):
        scores = agg_scores[f"{alg}_{is_al}"]  # (n_evals+1, n_pref_dirps)
        mean_E = scores.mean(1)
        std_E = (
            scores.std(1)
            if not args.use_stderr
            else scores.std(1) / jnp.sqrt(scores.shape[1])
        )
        mean_E = smooth(mean_E) if args.use_smooth else mean_E
        std_E = smooth(std_E) if args.use_smooth else std_E
        max_score = max(max_score, mean_E.max())
        min_score = min(min_score, mean_E.min())
        label = get_label(alg, is_al)
        style = get_style(alg, is_al)
        ax.plot(mean_E, label=label, **style, linewidth=2)
        ax.fill_between(
            range(len(mean_E)),
            mean_E - std_E,
            mean_E + std_E,
            alpha=0.2,
            **style,
        )

    gt_line = ax.axhline(agg_gt, color=rgb_values["gray"], linestyle="-", linewidth=1.0)
    zero_line = ax.axhline(
        agg_zero, color=rgb_values["gray"], linestyle="--", linewidth=1.0
    )

    ax.set_xlabel("Evaluation Steps", **get_font_kw(18))
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x):d}" for x in xticks], **get_font_kw(16))
    ax.set_xlim(left=0, right=len(mean_E))  # Cut off the graph

    ax.set_ylabel("Normalized Score", **get_font_kw(18))
    # ax.set_ylim(min_score - 3, max_score + 3)
    ax.set_ylim(20, 60)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.0f}" for y in yticks], **get_font_kw(16))

    agg_legend_handles = []
    agg_legend_labels = []
    agg_handler_map = {tuple: DualLineHandler()}
    for alg in algs:
        alg_color = get_style(alg, True)["color"]
        alg_label = get_label(alg)
        agg_legend_handles.append((alg_color, alg_label))
        agg_legend_labels.append(alg_label)
    agg_legend_handles.extend([gt_line, zero_line])
    agg_legend_labels.extend(["GT", "Zero"])

    # agg_handles_ordered = agg_legend_handles  # old ordering
    agg_handles_ordered = reorder_legend_row_major(agg_legend_handles, 4)
    # agg_labels_ordered = agg_legend_labels  # old ordering
    agg_labels_ordered = reorder_legend_row_major(agg_legend_labels, 4)

    ax.legend(
        agg_handles_ordered,
        agg_labels_ordered,
        handler_map=agg_handler_map,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        **get_legend_kw(18),
    )

    fp = f"{save_dir}/offlineRL_{timestamp}_{aux_fname}_{n_seeds}seeds_agg.png"
    plt.savefig(fp, bbox_inches="tight", dpi=300)
    print(f"Saved to {fp}")


def combine_pref_scores(parent_dir: str):
    """
    parent_dir: str
        each subdir is a hydra sweep (treated one seed), containing the override directories
        n_seeds := n_pref_dirps

    Turn List[dict[task][ekf_False] # (n_evals+1, n_workers)] * n_seeds
    -> d[task][ekf_False] # (n_seeds, n_evals+1, n_workers)
    -> d[task][ekf_False] # (n_evals+1, n_seeds) ; take mean over n_workers per step
    """
    seed_dirs = [
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    scores_dicts = [get_pref_score(seed_dir, tasks) for seed_dir in seed_dirs]

    combined_scores_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for task in tasks:
        for alg, is_al in it.product(algs, is_als):
            combined_arr = np.stack(
                [scores_dict[task][f"{alg}_{is_al}"] for scores_dict in scores_dicts],
                axis=0,
            )  # (n_seeds, n_evals+1, n_workers)
            combined_arr = combined_arr.mean(2).swapaxes(0, 1)  # (n_evals+1, n_seeds)
            combined_scores_dict[task][f"{alg}_{is_al}"] = combined_arr
    return combined_scores_dict


def aggregate_scores_task(combined_scores_dict: dict):
    """
    Takes output from combine_pref_scores(), combine scores across tasks

    dict[task][ekf_False] # (n_evals+1, n_seeds) ->
    dict[ekf_False] # (n_task, n_evals+1, n_seeds) ->
    dict[ekf_False] # (n_evals+1, n_seeds) ; take mean over n_task
    """
    out = {}
    for alg, is_al in it.product(algs, is_als):
        combined_arrs = np.stack(
            [combined_scores_dict[task][f"{alg}_{is_al}"] for task in tasks],
            axis=0,
        )  # (n_task, n_evals+1, n_pref_dirps)
        out[f"{alg}_{is_al}"] = combined_arrs.mean(0)  # (n_evals+1, n_pref_dirps)

    return out  # d[ekf_False] # (n_evals+1, n_pref_dirps)


def get_pref_score(dir_path: str, tasks: List[str]):
    """
    tasks = ["cheetahRandom", "cheetahMediumReplay", ...]
    dir_path to hydra sweep folder

    stats.npz
        returns: (n_evals+1, n_workers)
        scores: (n_evals+1, n_workers)

    Returns:
        scores_dict[task][ekf_True] = score (float)
        scores_dict[task][ekf_False] = score (float)
        scores_dict[task][sgd_True] = score (float)
        scores_dict[task][sgd_False] = score (float)
    """
    scores_dict = {}
    for task in tasks:
        scores_dict[task] = {}
        for alg, is_al in it.product(algs, is_als):
            # Construct folder path
            subdir_pattern = f"prefAlg={alg}_prefIsAl={is_al}_task={task}"
            folders = [f for f in os.listdir(dir_path) if subdir_pattern in f]

            folder = os.path.join(dir_path, folders[0])

            # Find the only npz file in the folder
            try:
                data_NE = np.load(f"{folder}/stats.npz", allow_pickle=True)
                scores = data_NE["scores"]  # (n_evals+1, n_workers)
                if scores.ndim == 1:
                    # for a cfg mistake where n_eval_workers != n_final_eval_episodes
                    scores = fix_uneven_nworkers(scores)
                scores_dict[task][f"{alg}_{is_al}"] = scores
            except Exception as e:
                print(f"Error loading npz from {folder}: {e}")
    return scores_dict


def fix_uneven_nworkers(scores: np.ndarray):
    """
    scores: npz array with (n_evals+1) rows where
    - first n_evals rows have n_workers columns
    - last single row has n_workers_final columns > n_workers

    ON that last row, drop columns until n_workers_final == n_workers

    Returns:
        scores: (n_evals+1, n_workers)
    """
    first = np.stack(scores[:-1])  # (n_evals, n_workers)
    last = scores[-1]  # (n_workers_final,)
    n_workers = first.shape[1]
    last = last[:n_workers][np.newaxis, :]  # (1, n_workers)
    return np.vstack([first, last])  # (n_evals+1, n_workers)


def get_baseline_score(dir_path: str, tasks: List[str]):
    """
    tasks = ["cheetahRandom", "cheetahMediumReplay", ...]
    dir_path to hydra sweep folder


    Returns:
        scores_dict[task][reward_type] = score (float)
    """
    # Dictionary to store results for each task and reward type
    scores_dict = {}

    for task in tasks:
        scores_dict[task] = {}
        for reward_type in ["zero", "gt"]:
            # Construct folder path
            folder = f"{dir_path}/rl.reward={reward_type}, task={task}"
            # Find the only npz file in the folder
            try:
                npz_files = [f for f in os.listdir(folder) if f.endswith(".npz")]
                data_NE = np.load(f"{folder}/{npz_files[0]}", allow_pickle=True)
                scores = data_NE["final_scores"]  # (n_workers,)
                scores_dict[task][reward_type] = scores.mean()

                # print(f"{task} {reward_type} {scores.mean():.2f}")
            except Exception as e:
                print(f"Error loading npz from {folder}: {e}")

    for reward_type in ["zero", "gt"]:
        scores_dict[f"agg_{reward_type}"] = np.mean(
            [scores_dict[task][reward_type] for task in tasks]
        )

    return scores_dict


if __name__ == "__main__":
    main()
