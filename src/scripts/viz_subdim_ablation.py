"""
Subdimension ablation plots for PreferenceEKF.
"""

import itertools as it
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import jax

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
logging.getLogger("absl").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import numpy as np
import tyro

from bnn_pref.utils.plotting import (
    get_font_kw,
    get_legend_kw,
    invisible_topright_spines,
    rgb_values,
)

subdims = [10, 25, 50, 75, 100, 125, 150, 200, 250, 400, 500]
rnd_projs = [False, True]


@dataclass
class Args:
    dirp: Path
    """Hydra sweep directory from sweep_ekf_subdim.sh."""
    save_dir: Optional[Path] = None
    """Output directory; defaults to dirp."""
    task: str = "walker2d-medium-expert-v2"


def get_label(rnd_proj: bool) -> str:
    if rnd_proj:
        return "Random Projection"
    return "SVD"


def get_style(rnd_proj: bool) -> dict:
    color = rgb_values["orange"] if not rnd_proj else rgb_values["gray"]
    return {"color": color, "linestyle": "-", "linewidth": 2}


def main(args: Args) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir or args.dirp
    save_dir.mkdir(parents=True, exist_ok=True)

    stats = {"logpdf_mean": [], "logpdf_std": [], "ece_mean": [], "ece_std": []}

    for rnd_proj, subdim in it.product(rnd_projs, subdims):
        subfolder_str = f"subdim={subdim}_rnd_proj={rnd_proj}"
        for folder in os.listdir(args.dirp):
            if subfolder_str in folder:
                path = args.dirp / folder / "stats.npz"
                arr = np.load(path, allow_pickle=True)[args.task].item()["ekf"]
                for metric in ["logpdf", "ece"]:
                    stat = arr[f"test_{metric}_final"]
                    stats[f"{metric}_mean"].append(stat["mean"].item())
                    stats[f"{metric}_std"].append(stat["std"].item())
    stats = {k: np.array(v) for k, v in stats.items()}

    stats_rnd = jax.tree.map(lambda x: x[: len(subdims)], stats)
    stats_svd = jax.tree.map(lambda x: x[len(subdims) :], stats)
    stats_by_proj = {False: stats_rnd, True: stats_svd}

    axisLabel_kw = get_font_kw(18)
    axisTick_kw = get_font_kw(16)
    legend_kw = get_legend_kw(18)

    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    invisible_topright_spines(ax1)

    for rnd_proj in rnd_projs:
        mean = stats_by_proj[rnd_proj]["logpdf_mean"]
        std = stats_by_proj[rnd_proj]["logpdf_std"]
        style = get_style(rnd_proj)
        label = get_label(rnd_proj)
        ax1.plot(subdims, mean, label=label, **style, marker="o" if not rnd_proj else "s")
        ax1.fill_between(subdims, mean - std, mean + std, alpha=0.2, **style)

    ax1.set_xlabel("Subspace Dimension", **axisLabel_kw)
    xticks = ax1.get_xticks()
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{int(x):d}" for x in xticks], **axisTick_kw)
    ax1.set_ylabel("Test Log-Likelihood", **axisLabel_kw)
    yticks = ax1.get_yticks()[::2]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([f"{y:.2f}" for y in yticks], **axisTick_kw)
    ax1.legend(**legend_kw)
    ax1.set_xlim(0, subdims[-1] + 5)

    plt.tight_layout()
    out_fp = save_dir / f"{timestamp}_{args.task}.png"
    plt.savefig(out_fp, bbox_inches="tight", dpi=300)
    plt.close(fig1)
    print(f"Saved plot to {out_fp}")


if __name__ == "__main__":
    main(tyro.cli(Args))
