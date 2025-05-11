"""
For aggregated logpdf plots over all tasks and seeds, per each algorithm variant.
"""

import itertools as it
import logging
import os
from datetime import datetime

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
logging.getLogger("absl").setLevel(logging.WARNING)

import ipdb
import matplotlib.pyplot as plt
import numpy as np

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
Ms = [5, 15, 30, 50, 100, 150, 200, 250]
# Ms = [5, 15, 30, 50, 100, 150, 200]
# task = "halfcheetah-medium-expert-v2"
fixed_net = "64x3"
fixed_M = 5
task = "acrobot-swingup-v0"
algs = ["ekf", "sgd"]
# is_als = [True, False] $ always active
save_dir = "/scr/yutaizho/projects/bnn_pref/_viz"

# * == change this block ==
M_dirp = "/scr/yutaizho/projects/bnn_pref/results_sweep/scaling/20250511_065332_scale_M_random"

net_dirp = "/scr/yutaizho/projects/bnn_pref/results_sweep/scaling/20250511_065334_scale_param_random"
# * == change this block ==


def get_duration(
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
            for alg in algs
            for metric in ["duration", "logpdf"]
        }
        return alg_dicts

    if sweep_type == "M":
        out = {
            "M": Ms,
            "net": fixed_net,
            **create_alg_dicts(),
        }
        for i, M in enumerate(Ms):
            fp = os.path.join(dirp, f"{i}_M={M}_net={fixed_net}", "stats.npz")
            stats = np.load(fp, allow_pickle=True)
            for alg in algs:
                res = stats[task].item()[alg]
                out[f"{alg}_duration"].append(res["duration"])
                # (n_seeds, nq_updates)
                out[f"{alg}_logpdf"].append(res["test_logpdf_all"].mean(0))
        return out
    elif sweep_type == "net":
        out = {
            "M": fixed_M,
            "net": nets,
            **create_alg_dicts(),
        }

        for i, net in enumerate(nets):
            fp = os.path.join(dirp, f"{i}_M={fixed_M}_net={net}", "stats.npz")
            stats = np.load(fp, allow_pickle=True)
            for alg in algs:
                res = stats[task].item()[alg]
                out[f"{alg}_duration"].append(res["duration"])
                # (n_seeds, nq_updates)
                out[f"{alg}_logpdf"].append(res["test_logpdf_all"].mean(0))
        return out


M_res = get_duration(M_dirp, sweep_type="M", fixed_net=fixed_net)
net_res = get_duration(net_dirp, sweep_type="net", fixed_M=fixed_M)
# ipdb.set_trace()


def get_label(alg: str) -> str:
    alg_str = "EKF" if alg == "ekf" else "Ensemble"
    return f"{alg_str}"


def get_style(alg: str) -> dict:
    color = "blue" if alg == "ekf" else "orange"
    return {"color": color, "linestyle": "-", "linewidth": 1}


# * plot ensemble size and network size vs. duration
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ax = axs[0]
for alg in algs:
    durations = M_res[f"{alg}_duration"]
    style = get_style(alg)
    label = get_label(alg)
    ax.plot(durations, label=label, marker="o", markersize=3, **style)
ax.set_xticks(range(len(Ms)))
ax.set_xticklabels(Ms)
ax.set_xlabel("Ensemble Size")
ax.set_title("Ensemble Size vs. Duration (s)")
ax.legend()

ax = axs[1]
for alg in algs:
    durations = net_res[f"{alg}_duration"]
    style = get_style(alg)
    label = get_label(alg)
    ax.plot(durations, label=label, marker="o", markersize=3, **style)
ax.set_xticks(range(len(nets)))
ax.set_xticklabels(nets, rotation=45, ha="right")
ax.set_xlabel("Network Size")
ax.set_title("Network Size vs. Duration (s)")
ax.legend()

fig.suptitle("Runtime of EKF and Ensemble")
plt.tight_layout()
save_fp = f"{save_dir}/scale_{timestamp}_runtime.png"
plt.savefig(save_fp)
print(f"Saved to {save_fp}")

# * plot ensemble size and network size vs. logpdf
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# ax = axs[0]
# for alg in algs:
#     logpdfs = M_res[f"{alg}_logpdf"]
#     style = get_style(alg)
#     label = get_label(alg)
#     ax.plot(logpdfs, label=label, marker="o", markersize=3, **style)
# ax.set_xticks(range(len(Ms)))
# ax.set_xticklabels(Ms)
# ax.set_xlabel("Ensemble Size")
# ax.set_title("Ensemble Size vs. Logpdf")
# ax.legend()

# ax = axs[1]
# for alg in algs:
#     logpdfs = net_res[f"{alg}_logpdf"]
#     style = get_style(alg)
#     label = get_label(alg)
#     ax.plot(logpdfs, label=label, marker="o", markersize=3, **style)
# ax.set_xticks(range(len(nets)))
# ax.set_xticklabels(nets, rotation=45, ha="right")
# ax.set_xlabel("Network Size")
# ax.set_title("Network Size vs. Logpdf")
# ax.legend()

fig.suptitle("EKF and Ensemble Logpdf vs. Network Size and Ensemble Size")
plt.tight_layout()
save_fp = f"{save_dir}/scale_{timestamp}_logpdf.png"
plt.savefig(save_fp)
print(f"Saved to {save_fp}")
