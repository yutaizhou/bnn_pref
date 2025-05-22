import itertools as it
import os
from collections import defaultdict
from datetime import datetime
from typing import List

import ipdb
import jax.numpy as jnp
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


def defaultdict2dict(dd):
    return {k: defaultdict2dict(v) if isinstance(v, dict) else v for k, v in dd.items()}


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
algs = ["ekf", "sgd"]
is_als = [True, False]

use_stderr = True  # otherwise use stderr
use_smooth = True  # otherwise no smoothing on eval curves

# * == change this block ==
save_dir = "PATH/TO/YOUR/bnn_pref/_viz/offlineRL"
ref_dirp = "PATH/TO/YOUR/bnn_pref/_runs/offline_rl/CHANGEME"
pref_dirp = "PATH/TO/YOUR/bnn_pref/_runs/offline_rl/CHANGEME"
# * == change this block ==


def main():
    baseline_scores = get_baseline_score(ref_dirp, tasks)  # d[task]["zero", "gt"]

    dir_name = pref_dirp.split("/")[-1]
    aux_fname = dir_name.split("iql_pref_18tasks_")[-1]  # gets "_nq60_5seed"

    # d[task][ekf_False] # (n_evals+1, n_pref_dirps)
    pref_scores = combine_pref_scores(pref_dirp)
    # d[ekf_False] # (n_evals+1, n_pref_dirps)
    agg_scores = aggregate_scores_task(pref_scores)

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
    lines = []  # Store lines for the shared legend
    labels = []  # Store labels for the shared legend

    for i, task in enumerate(tasks):
        ax = axs[i]
        invisible_topright_spines(ax)
        for alg, is_al in it.product(algs, is_als):
            scores = pref_scores[task][f"{alg}_{is_al}"]  # (n_evals+1, n_workers)
            mean_E = scores.mean(1)
            std_E = (
                scores.std(1)
                if not use_stderr
                else scores.std(1) / jnp.sqrt(scores.shape[1])
            )
            mean_E = smooth(mean_E) if use_smooth else mean_E
            std_E = smooth(std_E) if use_smooth else std_E
            label = get_label(alg, is_al)
            style = get_style(alg, is_al)
            line = ax.plot(mean_E, label=label, **style)[0]
            if i == 0:  # Only store legend info from first subplot
                lines.append(line)
                labels.append(get_label(alg, is_al))
            ax.fill_between(
                range(len(mean_E)),
                mean_E - std_E,
                mean_E + std_E,
                alpha=0.2,
                **style,
            )
        zero_score = baseline_scores[task]["zero"]
        gt_score = baseline_scores[task]["gt"]
        zero_line = ax.axhline(
            zero_score, color=rgb_values["gray"], linestyle="--", linewidth=1.0
        )
        gt_line = ax.axhline(
            gt_score, color=rgb_values["gray"], linestyle="-", linewidth=1.0
        )
        # for better zooming in visualization
        if task in ["mazeMediumDense", "hopperMediumExpert"]:
            max_score = max(scores[np.isfinite(scores)].max(), gt_score)
            min_score = min(scores[np.isfinite(scores)].min(), zero_score)
        else:
            max_score = max(mean_E[np.isfinite(mean_E)].max(), gt_score)
            min_score = min(mean_E[np.isfinite(mean_E)].min(), zero_score)
        ax.set_ylim(min_score - 2, max_score + 2)
        if i == 0:  # Only store baseline legend info from first subplot
            lines.extend([gt_line, zero_line])
            labels.extend(["GT", "Zero"])

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
    fig.legend(
        lines,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        handlelength=2,
        **get_legend_kw(16),
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    fp = f"{save_dir}/offlineRL_{timestamp}_{aux_fname}.png"
    plt.savefig(fp, bbox_inches="tight", dpi=300)
    print(f"Saved to {fp}")

    # * plot agg_scores
    fig, ax = plt.subplots(figsize=(10, 6))
    invisible_topright_spines(ax)
    max_score = 0
    min_score = jnp.inf
    for alg, is_al in it.product(algs, is_als):
        scores = agg_scores[f"{alg}_{is_al}"]  # (n_evals+1, n_pref_dirps)
        mean_E = scores.mean(1)
        std_E = (
            scores.std(1)
            if not use_stderr
            else scores.std(1) / jnp.sqrt(scores.shape[1])
        )
        mean_E = smooth(mean_E) if use_smooth else mean_E
        std_E = smooth(std_E) if use_smooth else std_E
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

    ax.set_xlabel("Evaluation Steps", **get_font_kw(18))
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x):d}" for x in xticks], **get_font_kw(16))
    ax.set_xlim(left=0, right=len(mean_E))  # Cut off the graph

    ax.set_ylabel("Normalized Score", **get_font_kw(18))
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.0f}" for y in yticks], **get_font_kw(16))
    ax.set_ylim(min_score - 3, max_score + 3)

    ax.legend(**get_legend_kw(18), loc="lower right")
    fp = f"{save_dir}/offlineRL_{timestamp}_{aux_fname}_agg.png"
    plt.savefig(fp, bbox_inches="tight", dpi=300)
    print(f"Saved to {fp}")


def combine_pref_scores(parent_dir: str):
    """
    parent_dir: str
        each subdir is a hydra_sweep run seed, containing the override directories

    Turn List of n_seeds d[task][ekf_False] # (n_evals+1, n_workers)
    -> d[task][ekf_False] # (n_pref_dirps, n_evals+1, n_workers)
    -> d[task][ekf_False] # (n_evals+1, n_pref_dirps) ; take mean over n_workers
    """
    seed_dirs = [
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    combined_scores_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    scores_dicts = [get_pref_score(seed_dir, tasks) for seed_dir in seed_dirs]
    for task in tasks:
        for alg, is_al in it.product(algs, is_als):
            # (n_pref_dirps, n_evals+1, n_workers)
            combined_arr = np.stack(
                [scores_dict[task][f"{alg}_{is_al}"] for scores_dict in scores_dicts],
                axis=0,
            )
            # (n_evals+1, n_pref_dirps)
            combined_arr = combined_arr.mean(2).swapaxes(0, 1)
            combined_scores_dict[task][f"{alg}_{is_al}"] = combined_arr
    return combined_scores_dict


def aggregate_scores_task(combined_scores_dict: dict):
    """
    d[task][ekf_False] # (n_evals+1, n_pref_dirps) ->
    d[ekf_False] # (n_task, n_evals+1, n_pref_dirps) ->
    d[ekf_False] # (n_evals+1, n_pref_dirps) ; take mean over n_task
    """
    new_d = {}
    for alg, is_al in it.product(algs, is_als):
        combined_arrs = np.stack(
            [combined_scores_dict[task][f"{alg}_{is_al}"] for task in tasks],
            axis=0,
        )  # (n_task, n_evals+1, n_pref_dirps)
        new_d[f"{alg}_{is_al}"] = combined_arrs.mean(0)  # (n_evals+1, n_pref_dirps)

    # ipdb.set_trace()
    return new_d  # d[ekf_False] # (n_evals+1, n_pref_dirps)


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
            folder_pattern = (
                f"rl.pref_alg={alg}, rl.pref_is_al={is_al}, rl.reward=pref, task={task}"
            )
            folders = [f for f in os.listdir(dir_path) if folder_pattern in f]
            folder = os.path.join(dir_path, folders[0])

            # Find the only npz file in the folder
            try:
                data_NE = np.load(f"{folder}/stats.npz", allow_pickle=True)
                scores = data_NE["scores"]  # (n_evals+1, n_workers)
                # ipdb.set_trace()
                scores_dict[task][f"{alg}_{is_al}"] = scores

                # print(f"{task} {reward_type} {scores.mean():.2f}")
            except Exception as e:
                print(f"Error loading npz from {folder}: {e}")
    return scores_dict


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

    return scores_dict


if __name__ == "__main__":
    main()
