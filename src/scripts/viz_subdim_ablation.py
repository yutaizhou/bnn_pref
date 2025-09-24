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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# subdims = [10, 50, 100, 150, 200, 300, 400, 500]
subdims = [10, 25, 50, 75, 100, 125, 150, 200, 250, 400, 500]
rnd_projs = [False, True]
task = "walker2d-medium-expert-v2"

# * == change this block ==
dirp = "/scr/yutaizho/projects/bnn_pref/results_sweep/subdim/20250924_162929_None_nitersInit=420_lr=0.003"
# save_dir = "/scr/yutaizho/projects/bnn_pref/_viz/subdim_ablation"
save_dir = dirp
# * == change this block ==
# os.makedirs(f"{save_dir}/{timestamp}", exist_ok=True)

means, stds = [], []
for rnd_proj, subdim in it.product(rnd_projs, subdims):
    subfolder_str = f"subdim={subdim}_rnd_proj={rnd_proj}"
    for folder in os.listdir(dirp):
        if subfolder_str in folder:
            path = os.path.join(dirp, folder, "stats.npz")
            arr = np.load(path, allow_pickle=True)[task].item()["ekf"]
            logpdfs = arr["test_logpdf_final"]  # mean, std
            means.append(logpdfs["mean"].item())
            stds.append(logpdfs["std"].item())

means = np.array(means)
stds = np.array(stds)

means_rnd, means_svd = means[: len(subdims)], means[len(subdims) :]
stds_rnd, stds_svd = stds[: len(subdims)], stds[len(subdims) :]
stats = {False: (means_rnd, stds_rnd), True: (means_svd, stds_svd)}


def get_label(rnd_proj: bool) -> str:
    if rnd_proj:
        return "Random Projection"
    else:
        return "SVD"


def get_style(rnd_proj: bool) -> dict:
    color = rgb_values["orange"] if not rnd_proj else rgb_values["gray"]
    return {"color": color, "linestyle": "-", "linewidth": 2}


axisLabel_kw = get_font_kw(18)
axisTick_kw = get_font_kw(16)
legend_kw = get_legend_kw(18)

fig1, ax1 = plt.subplots(figsize=(6, 4))
invisible_topright_spines(ax1)

for rnd_proj in rnd_projs:
    mean, std = stats[rnd_proj]
    style = get_style(rnd_proj)
    label = get_label(rnd_proj)
    ax1.plot(subdims, mean, label=label, **style, marker="o" if not rnd_proj else "s")
    ax1.fill_between(
        subdims,
        mean - std,
        mean + std,
        alpha=0.2,
        **style,
    )
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
plt.savefig(f"{save_dir}/{timestamp}_{task}.png", bbox_inches="tight", dpi=300)
plt.close(fig1)
