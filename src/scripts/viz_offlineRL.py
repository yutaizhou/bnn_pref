import itertools as it
import os
from collections import defaultdict
from datetime import datetime
from typing import List

import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


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
is_als = [False, True]
save_dir = "/scr/yutaizho/projects/bnn_pref/_viz"

use_stderr = True  # otherwise use stderr
use_smooth = True  # otherwise no smoothing on eval curves

# * == change this block ==
baseline_dirp = (
    "/scr/yutaizho/projects/bnn_pref/_runs/20250501_002013_iql_baselines_18tasks"
)

pref_dirp = "/scr/yutaizho/projects/bnn_pref/_runs/iql_pref_18tasks_nq60_5seed"
# * == change this block ==


def main():
    baseline_scores = get_baseline_score(baseline_dirp, tasks)  # d[task]["zero", "gt"]

    # pref_dirps = [
    #     "/scr/yutaizho/projects/bnn_pref/_runs/iql_pref_18tasks_5seed_nq60/20250502_005235_rewardNormClip",
    #     "/scr/yutaizho/projects/bnn_pref/_runs/iql_pref_18tasks_5seed_nq60/20250502_032221_rewardNormClip",
    #     "/scr/yutaizho/projects/bnn_pref/_runs/iql_pref_18tasks_5seed_nq60/20250502_032310_rewardNormClip",
    #     "/scr/yutaizho/projects/bnn_pref/_runs/iql_pref_18tasks_5seed_nq60/20250502_063110_rewardNormClip",
    #     "/scr/yutaizho/projects/bnn_pref/_runs/iql_pref_18tasks_5seed_nq60/20250502_065318_rewawrdNormClip",
    # ]
    dir_name = pref_dirp.split("/")[-1]
    aux_fname = dir_name.split("iql_pref_18tasks_")[-1]  # gets "_nq60_5seed"

    # d[task][ekf_False] # (n_evals+1, n_pref_dirps)
    pref_scores = combine_pref_scores(pref_dirp)
    # d[ekf_False] # (n_evals+1, n_pref_dirps)
    agg_scores = aggregate_scores_task(pref_scores)

    def get_label(alg: str, is_al: bool) -> str:
        if alg == "ekf":
            return "EKF (Active)" if is_al else "EKF (Random)"
        else:
            return "Ensemble (Active)" if is_al else "Ensemble (Random)"

    def get_style(alg: str, is_al: bool) -> dict:
        color = "blue" if alg == "ekf" else "orange"
        linestyle = "-" if is_al else "--"
        return {"color": color, "linestyle": linestyle, "linewidth": 1}

    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    axs = axs.flatten()
    lines = []  # Store lines for the shared legend
    labels = []  # Store labels for the shared legend

    def smooth(x_E, window_size=5):
        """Apply running average smoothing to the input array along axis 0.
        Args:
            x_E: array of shape (n_evals,)
            window_size: size of the smoothing window
        Returns:
            smoothed array of shape (n_evals,)
        """
        kernel = np.ones(window_size) / window_size
        return np.convolve(x_E, kernel, mode="valid")

    for i, task in enumerate(tasks):
        ax = axs[i]
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
                jnp.arange(len(mean_E)),
                mean_E - std_E,
                mean_E + std_E,
                alpha=0.2,
                **get_style(alg, is_al),
            )
            # ax.set_xticks(jnp.arange(len(mean_NEW)) * 25000)
            ax.set_title(task)
            ax.set_xlabel("Eval Steps", fontsize=8)
            ax.set_ylabel("Normalized Score", fontsize=8)

        zero_score = baseline_scores[task]["zero"]
        gt_score = baseline_scores[task]["gt"]
        zero_line = ax.axhline(zero_score, color="black", linestyle="--")
        gt_line = ax.axhline(gt_score, color="black", linestyle="-")
        max_score = max(mean_E.max(), gt_score)
        min_score = min(mean_E.min(), zero_score)
        ax.set_ylim(min_score - 5, max_score + 5)
        if i == 0:  # Only store baseline legend info from first subplot
            lines.extend([zero_line, gt_line])
            labels.extend(["Zero", "GT"])

        ax.set_title(task)

    fig.legend(lines, labels, loc="center right")
    fig.suptitle("Normalized Score vs. Eval Steps", fontsize=18)
    plt.tight_layout(rect=[0, 0, 0.87, 1])

    fp = f"{save_dir}/offlineRL_{timestamp}_{aux_fname}.png"
    plt.savefig(fp, bbox_inches="tight", dpi=300)
    print(f"Saved to {fp}")

    # * plot agg_scores
    fig, ax = plt.subplots(figsize=(12, 10))
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
        ax.plot(mean_E, label=get_label(alg, is_al), **get_style(alg, is_al))
        ax.fill_between(
            jnp.arange(len(mean_E)),
            mean_E - std_E,
            mean_E + std_E,
            alpha=0.2,
            **get_style(alg, is_al),
        )
    # ax.set_ylim(min_score - 5, max_score + 5)
    ax.set_xlabel("Eval Steps", fontsize=12)
    ax.set_ylabel("Normalized Score", fontsize=12)
    ax.set_title("Normalized Score Across Tasks")
    ax.legend()
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
