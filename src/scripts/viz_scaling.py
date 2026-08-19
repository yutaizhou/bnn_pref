"""
Scaling experiment plots: ensemble size M vs network width.
"""

import itertools as it
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
    set_xlim_offset,
)

fixed_net = "64x2"
fixed_M = 5
nets = [
    "64x2",
    "64x3",
    "128x3",
    "256x3",
    "512x2",
    "512x3",
    "1024x2",
    "1024x3",
]
Ms = [5, 15, 30, 50, 75, 100]
algs = ["ekf", "sgd", "do"]


@dataclass
class Args:
    M_dirp: Path
    """Hydra sweep dir for the M scaling run."""
    net_dirp: Path
    """Hydra sweep dir for the network-size scaling run."""
    save_dir: Path = Path("results/viz/scaling")
    task: str = "walker2d-medium-expert-v2"


def get_stats(
    dirp: Path,
    sweep_type: str,
    fixed_net: str = "64x3",
    fixed_M: int = 5,
    task: str = "walker2d-medium-expert-v2",
):
    assert sweep_type in ["M", "net"]

    def create_alg_dicts():
        return {
            f"{alg}_{metric}": list()
            for alg, metric in it.product(algs, ["duration", "logpdf"])
        }

    if sweep_type == "M":
        out = {
            "M": Ms,
            "net": fixed_net,
            **create_alg_dicts(),
        }
        for i, M in enumerate(Ms):
            seed_dirs = [d for d in os.listdir(dirp) if f"M={M}_net={fixed_net}" in d]
            alg_durations = {alg: [] for alg in algs}
            alg_logpdfs = {alg: [] for alg in algs}

            for seed_dir in seed_dirs:
                fp = dirp / seed_dir / "stats.npz"
                stats = np.load(fp, allow_pickle=True)
                for alg in algs:
                    res = stats[task].item()[alg]
                    alg_durations[alg].append(res["train_duration"])
                    alg_logpdfs[alg].append(res["test_logpdf_final"])

            for alg in algs:
                out[f"{alg}_duration"].append(np.array(alg_durations[alg]))
                out[f"{alg}_logpdf"].append(np.array(alg_logpdfs[alg]))

        for alg in algs:
            out[f"{alg}_duration"] = np.array(out[f"{alg}_duration"])
            out[f"{alg}_logpdf"] = np.array(out[f"{alg}_logpdf"])
        return out

    out = {
        "M": fixed_M,
        "net": nets,
        **create_alg_dicts(),
    }

    for i, net in enumerate(nets):
        seed_dirs = [d for d in os.listdir(dirp) if f"M={fixed_M}_net={net}" in d]
        alg_durations = {alg: [] for alg in algs}
        alg_logpdfs = {alg: [] for alg in algs}

        for seed_dir in seed_dirs:
            fp = dirp / seed_dir / "stats.npz"
            stats = np.load(fp, allow_pickle=True)
            for alg in algs:
                res = stats[task].item()[alg]
                alg_durations[alg].append(res["duration"])
                alg_logpdfs[alg].append(res["test_logpdf_final"])

        for alg in algs:
            out[f"{alg}_duration"].append(np.array(alg_durations[alg]))
            out[f"{alg}_logpdf"].append(np.array(alg_logpdfs[alg]))

    for alg in algs:
        out[f"{alg}_duration"] = np.array(out[f"{alg}_duration"])
        out[f"{alg}_logpdf"] = np.array(out[f"{alg}_logpdf"])
    return out


def get_label(alg: str) -> str:
    alg2label = {
        "ekf": "PreferenceEKF",
        "sgd": "DeepEnsemble",
        "do": "Dropout",
        "laplace": "Laplace",
        "llmcmc": "LLMCMC",
    }
    return alg2label[alg]


def get_style(alg: str) -> dict:
    alg2color = {
        "ekf": rgb_values["orange"],
        "sgd": rgb_values["blue"],
        "do": rgb_values["green"],
        "laplace": rgb_values["purple"],
        "llmcmc": rgb_values["gray"],
    }
    return {"color": alg2color[alg], "linestyle": "-", "linewidth": 2}


def main(args: Args) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.save_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    M_res = get_stats(args.M_dirp, sweep_type="M", fixed_net=fixed_net, task=args.task)
    net_res = get_stats(
        args.net_dirp, sweep_type="net", fixed_M=fixed_M, task=args.task
    )

    axisLabel_kw = get_font_kw(24)
    axisTick_kw = get_font_kw(20)
    legend_kw = get_legend_kw(20)

    # * 1. plot ensemble size vs. duration
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    invisible_topright_spines(ax1)
    for alg in algs:
        durations = M_res[f"{alg}_duration"]
        mean_durations = np.mean(durations, axis=1)
        std_durations = np.std(durations, axis=1)
        style = get_style(alg)
        label = get_label(alg)
        ax1.plot(mean_durations, label=label, **style)
        ax1.fill_between(
            range(len(Ms)),
            mean_durations - std_durations,
            mean_durations + std_durations,
            alpha=0.2,
            **style,
        )
    ax1.set_xlabel("M", **axisLabel_kw)
    ax1.set_xticks(range(len(Ms)))
    ax1.set_xticklabels([f"{int(x):d}" for x in Ms], **axisTick_kw)
    ax1.set_ylabel("Duration (s)", **axisLabel_kw)
    yticks = ax1.get_yticks()[::2]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([f"{y:.0f}" for y in yticks], **axisTick_kw)
    ax1.legend(**legend_kw)
    set_xlim_offset(ax1)
    ax1.set_xlim(0, len(Ms) - 1)
    plt.tight_layout()
    plt.savefig(out_dir / "a_MDuration.png", bbox_inches="tight", dpi=300)
    plt.close(fig1)

    # * 2. plot network size vs. duration
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    invisible_topright_spines(ax2)
    for alg in algs:
        durations = net_res[f"{alg}_duration"]
        mean_durations = np.mean(durations, axis=1)
        std_durations = np.std(durations, axis=1)
        style = get_style(alg)
        label = get_label(alg)
        ax2.plot(mean_durations, label=label, **style)
        ax2.fill_between(
            range(len(nets)),
            mean_durations - std_durations,
            mean_durations + std_durations,
            alpha=0.2,
            **style,
        )
    ax2.set_xlabel("Network Size", **axisLabel_kw)
    ax2.set_xticks(range(len(nets)))
    ax2.set_xticklabels(nets, rotation=45, ha="right", **axisTick_kw)
    set_xlim_offset(ax2)
    ax2.set_xlim(0, len(nets) - 1)
    ax2.set_ylabel("Duration (s)", **axisLabel_kw)
    yticks = ax2.get_yticks()[::2]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f"{y:.0f}" for y in yticks], **axisTick_kw)
    plt.tight_layout()
    plt.savefig(out_dir / "b_ParamDuration.png", bbox_inches="tight", dpi=300)
    plt.close(fig2)

    # * 3. plot ensemble size vs. logpdf
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    invisible_topright_spines(ax3)
    for alg in algs:
        logpdfs = M_res[f"{alg}_logpdf"]
        mean_logpdfs = np.mean(logpdfs, axis=1, where=np.isfinite(logpdfs))
        std_logpdfs = np.std(logpdfs, axis=1, where=np.isfinite(logpdfs))
        style = get_style(alg)
        label = get_label(alg)
        ax3.plot(mean_logpdfs, label=label, **style)
        ax3.fill_between(
            range(len(Ms)),
            mean_logpdfs - std_logpdfs,
            mean_logpdfs + std_logpdfs,
            alpha=0.2,
            **style,
        )
    ax3.set_xlabel("M", **axisLabel_kw)
    ax3.set_xticks(range(len(Ms)))
    ax3.set_xticklabels([f"{int(x):d}" for x in Ms], **axisTick_kw)
    set_xlim_offset(ax3)
    ax3.set_xlim(0, len(Ms) - 1)
    ax3.set_ylabel("Log-Likelihood", **axisLabel_kw)
    yticks = ax3.get_yticks()[::2]
    ax3.set_yticks(yticks)
    ax3.set_yticklabels([f"{y:.2f}" for y in yticks], **axisTick_kw)
    plt.tight_layout()
    plt.savefig(out_dir / "c_MLogpdf.png", bbox_inches="tight", dpi=300)
    plt.close(fig3)

    # * 4. plot network size vs. logpdf
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    invisible_topright_spines(ax4)
    for alg in algs:
        logpdfs = net_res[f"{alg}_logpdf"]
        mean_logpdfs = np.mean(logpdfs, axis=1, where=np.isfinite(logpdfs))
        std_logpdfs = np.std(logpdfs, axis=1, where=np.isfinite(logpdfs))
        label = get_label(alg)
        style = get_style(alg)
        ax4.plot(mean_logpdfs, label=label, **style)
        ax4.fill_between(
            range(len(nets)),
            mean_logpdfs - std_logpdfs,
            mean_logpdfs + std_logpdfs,
            alpha=0.2,
            **style,
        )
    ax4.set_xlabel("Network Size", **axisLabel_kw)
    ax4.set_xticks(range(len(nets)))
    ax4.set_xticklabels(nets, rotation=45, ha="right", **axisTick_kw)
    ax4.set_xlim(0, len(nets) - 1)
    set_xlim_offset(ax4)
    ax4.set_ylabel("Log-Likelihood", **axisLabel_kw)
    yticks = ax4.get_yticks()[::2]
    ax4.set_yticks(yticks)
    ax4.set_yticklabels([f"{y:.2f}" for y in yticks], **axisTick_kw)
    plt.tight_layout()
    plt.savefig(out_dir / "d_ParamLogpdf.png", bbox_inches="tight", dpi=300)
    plt.close(fig4)

    print(f"Saved individual plots to {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
