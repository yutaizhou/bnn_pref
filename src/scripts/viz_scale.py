"""
For aggregated logpdf plots over all tasks and seeds, per each algorithm variant.
"""

import itertools as it
import logging
import os
from datetime import datetime
from typing import Tuple

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
logging.getLogger("absl").setLevel(logging.WARNING)

import ipdb
import matplotlib.pyplot as plt
import numpy as np

from bnn_pref.utils.plotting import (
    get_font_kw,
    get_legend_kw,
    invisible_topright_spines,
    rgb_values,
    set_xlim_offset,
)

fixed_net = "64x3"
fixed_M = 5
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nets = [
    "32x2",
    "64x3",
    "128x3",
    "256x3",
    "512x2",
    "512x3",
    "1024x2",
    "1024x3",
]
Ms = [5, 15, 30, 50, 100, 150, 200]
# task = "acrobot-swingup-v0"
task = "halfcheetah-medium-expert-v2"
algs = ["ekf", "sgd"]

# * == change this block ==
save_dir = "PATH/TO/YOUR/bnn_pref/_viz/scale"
M_dirp = "PATH/TO/YOUR/bnn_pref/results_sweep/scaling/CHANGEME"
net_dirp = "PATH/TO/YOUR/bnn_pref/results_sweep/scaling/CHANGEME"
# * == change this block ==


def get_stats(
    dirp: str,
    sweep_type: str,
    fixed_net: str = "64x3",
    fixed_M: int = 5,
):
    """
    dirp: directory path to hydra sweep folder
        0_M=5_net=64x2/
        1_M=15_net=64x2/
        ...

        each contains stats.npz

    returns:
        dict[alg] = {
            "M": (n_M, ),
            "net": (n_net, ),
            "duration": (n_M, ) if sweep_name="M" else (n_net, ),
            "logpdf": (n_M, ) if sweep_name="M" else (n_net, ),
        }
    """
    assert sweep_type in ["M", "net"]

    def create_alg_dicts():
        alg_dicts = {
            f"{alg}_{metric}": list()
            for alg, metric in it.product(algs, ["duration", "logpdf"])
        }
        return alg_dicts

    if sweep_type == "M":
        out = {
            "M": Ms,
            "net": fixed_net,
            **create_alg_dicts(),
        }
        for i, M in enumerate(Ms):
            # Get all seed directories for this M value
            seed_dirs = [d for d in os.listdir(dirp) if f"M={M}_net={fixed_net}" in d]

            # Initialize lists to store per-algorithm metrics over seeds
            alg_durations = {alg: [] for alg in algs}
            alg_logpdfs = {alg: [] for alg in algs}

            for seed_dir in seed_dirs:
                fp = os.path.join(dirp, seed_dir, "stats.npz")
                stats = np.load(fp, allow_pickle=True)
                for alg in algs:
                    res = stats[task].item()[alg]
                    alg_durations[alg].append(res["duration"])
                    alg_logpdfs[alg].append(res["test_logpdf_final"].mean)

            # Compute mean and std for each metric per algorithm
            for alg in algs:
                # Store mean duration and logpdf
                out[f"{alg}_duration"].append(np.array(alg_durations[alg]))
                out[f"{alg}_logpdf"].append(np.array(alg_logpdfs[alg]))

        # convert to np array (n_M, n_seeds)
        for alg in algs:
            out[f"{alg}_duration"] = np.array(out[f"{alg}_duration"])
            out[f"{alg}_logpdf"] = np.array(out[f"{alg}_logpdf"])
        return out

    elif sweep_type == "net":
        out = {
            "M": fixed_M,
            "net": nets,
            **create_alg_dicts(),
        }

        for i, net in enumerate(nets):
            # Get all seed directories for this network size
            seed_dirs = [d for d in os.listdir(dirp) if f"M={fixed_M}_net={net}" in d]

            # Initialize lists to store per-algorithm metrics over seeds
            alg_durations = {alg: [] for alg in algs}
            alg_logpdfs = {alg: [] for alg in algs}

            for seed_dir in seed_dirs:
                fp = os.path.join(dirp, seed_dir, "stats.npz")
                stats = np.load(fp, allow_pickle=True)
                for alg in algs:
                    res = stats[task].item()[alg]
                    alg_durations[alg].append(res["duration"])
                    alg_logpdfs[alg].append(res["test_logpdf_final"].mean)

            # Store results for each algorithm
            for alg in algs:
                out[f"{alg}_duration"].append(np.array(alg_durations[alg]))
                out[f"{alg}_logpdf"].append(np.array(alg_logpdfs[alg]))

        # convert to np array (n_net, n_seeds)
        for alg in algs:
            out[f"{alg}_duration"] = np.array(out[f"{alg}_duration"])
            out[f"{alg}_logpdf"] = np.array(out[f"{alg}_logpdf"])
        return out


M_res = get_stats(M_dirp, sweep_type="M", fixed_net=fixed_net)
net_res = get_stats(net_dirp, sweep_type="net", fixed_M=fixed_M)


def get_label(alg: str) -> str:
    alg_str = "PreferenceEKF" if alg == "ekf" else "DeepEnsemble"
    return f"{alg_str}"


def get_style(alg: str) -> dict:
    color = rgb_values["orange"] if alg == "ekf" else rgb_values["blue"]
    return {"color": color, "linestyle": "-", "linewidth": 2}


axisLabel_kw = get_font_kw(24)
axisTick_kw = get_font_kw(20)
legend_kw = get_legend_kw(24)


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
plt.savefig(f"{save_dir}/scale_{timestamp}_a.png", bbox_inches="tight", dpi=300)
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
plt.savefig(f"{save_dir}/scale_{timestamp}_b.png", bbox_inches="tight", dpi=300)
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
plt.savefig(f"{save_dir}/scale_{timestamp}_c.png", bbox_inches="tight", dpi=300)
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
plt.savefig(f"{save_dir}/scale_{timestamp}_d.png", bbox_inches="tight", dpi=300)
plt.close(fig4)

print(f"Saved individual plots to {save_dir}")
