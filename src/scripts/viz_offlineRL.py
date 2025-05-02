import itertools as it
import os
from datetime import datetime
from typing import List

import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


def defaultdict2dict(dd):
    return {k: defaultdict2dict(v) if isinstance(v, dict) else v for k, v in dd.items()}


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
    "penExpert",
    "penCloned",
    "kitchenComplete",
    "kitchenPartial",
    "kitchenMixed",
    "mazeUDense",
    "mazeMediumDense",
    "mazeLargeDense",
]
algs = ["ekf", "sgd"]
is_als = [False, True]


def main():
    run_dir = "/scr/yutaizho/projects/bnn_pref/_runs"
    baseline_dirp = f"{run_dir}/20250501_002013_iql_baselines_18tasks"
    pref_dirp = f"{run_dir}/20250501_220052_iql_pref_18tasks_1seed"
    baseline_scores = get_baseline_score(baseline_dirp, tasks)  # d[task][zero, gt]
    pref_scores = get_pref_score(
        pref_dirp, tasks
    )  # d[task][ekf_False] # (n_evals+1, n_workers)

    """
    stats[task][alg][is_al] = {
        "score": (n_evals+1, n_workers),
        "reward_src": str, (zero, gt, model_fp)
    }
    """

    def get_label(alg: str, is_al: bool) -> str:
        if alg == "ekf":
            return "EKF (Active)" if is_al else "EKF (Random)"
        else:
            return "Ensemble (Active)" if is_al else "Ensemble (Random)"

    def get_style(alg: str, is_al: bool) -> dict:
        color = "blue" if alg == "ekf" else "orange"
        linestyle = "-" if is_al else "--"
        return {"color": color, "linestyle": linestyle, "linewidth": 1}

    fig, axs = plt.subplots(5, 4, figsize=(12, 10))
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
            scores_EN = pref_scores[task][f"{alg}_{is_al}"]  # (n_evals+1, n_workers),
            mean_E, std_E = scores_EN.mean(1), scores_EN.std(1)
            mean_E = smooth(mean_E)
            std_E = smooth(std_E)
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
            ax.set_xlabel("Eval Steps")
            ax.set_ylabel("Normalized Score")

        zero_score = baseline_scores[task]["zero"]
        gt_score = baseline_scores[task]["gt"]
        zero_line = ax.axhline(zero_score, color="black", linestyle="--")
        gt_line = ax.axhline(gt_score, color="black", linestyle="-")
        highest_score = max(mean_E.max(), gt_score)
        if highest_score <= 100:
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(0, highest_score + 10)
        if i == 0:  # Only store baseline legend info from first subplot
            lines.extend([zero_line, gt_line])
            labels.extend(["zero", "gt"])

        ax.set_title(task)

    # Add a single shared legend outside the subplots
    fig.legend(lines, labels, loc="center right", bbox_to_anchor=(1.1, 0.5))
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)  # Reduced right margin from 0.8 to 0.88
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"offline_rl_scores_{timestamp}.png", bbox_inches="tight", dpi=300)


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
    # dir_path = (
    #     "/scr/yutaizho/projects/bnn_pref/_runs/20250429_215500_iql_gt_zero_6tasks"
    # )
    # scores_dict = get_baseline_score(dir_path)

    main()
