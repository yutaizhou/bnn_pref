import itertools as it
import os

import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


def defaultdict2dict(dd):
    return {k: defaultdict2dict(v) if isinstance(v, dict) else v for k, v in dd.items()}


def main():
    tasks = [
        # # * D4RL
        "cheetahRand",
        "cheetahMedExp",
        "hopperRand",
        "hopperMed",
        "walkerRand",
        "walkerMedReplay",
    ]
    algs = ["ekf", "sgd"]
    is_als = [False, True]

    stats_fp = (
        "/scr/yutaizho/projects/bnn_pref/results/offline_rl/20250430_064726/stats.npz"
    )
    baseline_dirp = (
        "/scr/yutaizho/projects/bnn_pref/_runs/20250429_215500_iql_gt_zero_6tasks"
    )
    baseline_scores = get_baseline_score(baseline_dirp)
    stats = {
        task: defaultdict2dict(np.load(stats_fp, allow_pickle=True)[task].item())
        for task in tasks
    }
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

    fig, axs = plt.subplots(3, 4, figsize=(12, 10))
    axs = axs.flatten()
    lines = []  # Store lines for the shared legend
    labels = []  # Store labels for the shared legend

    for i, task in enumerate(tasks):
        ax = axs[i]
        for alg, is_al in it.product(algs, is_als):
            scores_EN = stats[task][alg][is_al]["scores"]  # (n_evals+1, n_workers),
            mean_E, std_E = scores_EN.mean(1), scores_EN.std(1)
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
            ax.set_ylim(0, 100)
            # ax.set_xticks(jnp.arange(len(mean_NEW)) * 25000)
            ax.set_title(task)
            ax.set_xlabel("Eval Steps")
            ax.set_ylabel("Normalized Score")

        zero_score = baseline_scores[task]["zero"]
        gt_score = baseline_scores[task]["gt"]
        zero_line = ax.axhline(zero_score, color="black", linestyle="--")
        gt_line = ax.axhline(gt_score, color="black", linestyle="-")
        if i == 0:  # Only store baseline legend info from first subplot
            lines.extend([zero_line, gt_line])
            labels.extend(["zero", "gt"])

        # Remove individual legends
        ax.set_title(task)

    # Add a single shared legend outside the subplots
    fig.legend(lines, labels, loc="center right", bbox_to_anchor=(1.1, 0.5))
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)  # Reduced right margin from 0.8 to 0.88
    plt.savefig("offline_rl_scores.png", bbox_inches="tight", dpi=300)


def get_baseline_score(dir_path: str):
    # Dictionary to store results for each task and reward type
    scores_dict = {}

    tasks = [
        "walkerMedReplay",
        "walkerRand",
        "hopperRand",
        "hopperMed",
        "cheetahRand",
        "cheetahMedExp",
    ]
    reward_types = ["zero", "gt"]

    for task in tasks:
        scores_dict[task] = {}
        for reward_type in reward_types:
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
    dir_path = (
        "/scr/yutaizho/projects/bnn_pref/_runs/20250429_215500_iql_gt_zero_6tasks"
    )
    scores_dict = get_baseline_score(dir_path)

    main()
