import itertools as it
import logging
import os
from datetime import datetime
from functools import partial
from typing import Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
logging.getLogger("absl").setLevel(logging.WARNING)

import hydra
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from einops import rearrange
from flax.training import orbax_utils
from hydra.core.hydra_config import HydraConfig
from jaxtyping import Array, Float

from bnn_pref.data import dataset_creators
from bnn_pref.data.traj_utils import normalize, subsample
from bnn_pref.rl.rm_util import load_reward_model, relabel_rewards
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.utils import get_random_seed, nested_defaultdict

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)
    save_dir = "PATH/TO/YOUR/bnn_pref/_viz"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks = [
        # * gym
        # "reacher",
        # "lunar",
        # "cheetah",
        # # * Deepmind Control
        # "acrobot",
        # "ball",
        # "cartpoleSwing",
        # "cheetahDMC",
        # "hopperHop",
        # "pendulum",
        # "walkerWalk",
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
    nt_test_tokeep = 100
    algs = ["ekf", "sgd"]
    # algs = ["sgd"]
    # algs = ["ekf"]
    is_als = [False, True]
    # is_als = [False]
    # is_als = [True]

    stats = nested_defaultdict()

    data_cfg = cfg["data"]

    print(
        f"Run:\n"
        f"  Seed: {seed} x {cfg['seeds']} (seed_vmap={cfg['seed_vmap']})\n"
        f"  Sanity: {cfg['sanity']} ({cfg['sanity_frac']} real frac)\n"
        f"  Network: {cfg['network']['hidden_sizes']}\n"
        f"Data:\n"
        f"  prune: {data_cfg['n_bins']} bins, {data_cfg['max_count_per_bin']} max_count_per_bin, {data_cfg['tokeep']} tokeep\n"
        f"  noisy_label: {data_cfg['noisy_label']} (beta={data_cfg['bt_beta']})\n"
        f"  Train/Test: {data_cfg['nq_train']}/{data_cfg['nq_test']}\n"
        # f"EKF:\n"
        # f"  M={ekf_cfg['M']}, use_vmap={ekf_cfg['use_vmap']}\n"
        # f"  prior / dynamics / obs noise: {ekf_cfg['prior_noise']} / {ekf_cfg['dynamics_noise']} / {ekf_cfg['obs_noise']}\n"
        # f"  init: bs={ekf_cfg['bs']}, niters={ekf_cfg['niters']}[{ekf_cfg['warm_burns']}::{ekf_cfg['thinning']}] ({n_eff_iterates} eff), sub_dim={ekf_cfg['sub_dim']}, rnd_proj={ekf_cfg['rnd_proj']}\n"
        # f"Ensemble:\n"
        # f"  M={sgd_cfg['M']}, use_vmap={sgd_cfg['use_vmap']}\n"
        # f"  init: bs={sgd_cfg['bs']}, niters={sgd_cfg['niters']}\n"
    )

    rm_dirp = "PATH/TO/YOUR/bnn_pref/_runs/pref/CHANGEME"
    for task in tasks:
        # * update cfg
        new_cfg = hydra.compose("config", overrides=[f"task={task}"])
        cfg["task"].update(new_cfg["task"])

        # * create dataset + optional subsampling
        key, key_data = jr.split(key, 2)
        data_dict = dataset_creators[cfg["task"]["ds_type"]](key_data, cfg)
        train_trajs, test_trajs = data_dict["train_trajs"], data_dict["test_trajs"]
        test_trajs = subsample(key, test_trajs, tokeep=nt_test_tokeep)
        nt_test, T, D = test_trajs["observations"].shape

        print(f"{task:13} ({T=}, {D=}): test nt=({nt_test})")

        obs = test_trajs["observations"]  # (nt_test, T, D)
        obs = normalize(rearrange(obs, "N T D -> (N T) D"), axis=(0,))
        ret_gt = test_trajs["returns"]  # (nt_test,)
        rho_gt = compute_rho(ret_gt, normalize_ret=True)  # (nt_test,)
        for alg, is_al in it.product(algs, is_als):
            key, key_rm = jr.split(key, 2)
            reward_fn, ckpt_fp = load_reward_model(
                key=key_rm,
                run_dir=rm_dirp,
                task_name=cfg["task"]["name"],
                alg=alg,
                is_al=is_al,
            )

            rhat = relabel_rewards(reward_fn, obs)  # (n_transitions,)
            rhat = normalize(rhat, axis=(0,))
            rhat = rearrange(rhat, "(N T) -> N T", N=nt_test)
            ret_pref = rhat.sum(axis=1)  # (nt_test,)
            rho_pref = compute_rho(ret_pref, normalize_ret=True)  # (nt_test,)
            rho_proj = jnp.linalg.norm(rho_pref - rho_gt, axis=0)
            stats["rho_proj"][task][alg][is_al] = rho_proj

            print(
                f"  {alg} active={str(is_al):5}, ",
                f"rho_proj: {rho_proj:.3f}",
            )
    # * print rho_proj for each alg averaged over tasks
    for alg, is_al in it.product(algs, is_als):
        rho_proj = jnp.array([stats["rho_proj"][task][alg][is_al] for task in tasks])
        print(f"{alg} active={str(is_al):5}, rho_proj: {rho_proj.mean():.3f}")

    # * bar plot
    def get_style(alg: str, is_al: bool) -> dict:
        color = "blue" if alg == "ekf" else "orange"
        return {"color": color, "alpha": 0.7 if is_al else 0.25}

    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    axs = axs.flatten()
    for i, task in enumerate(tasks):
        ax = axs[i]
        x_positions = range(len(list(it.product(algs, is_als))))
        for j, (alg, is_al) in enumerate(it.product(algs, is_als)):
            rho_proj = stats["rho_proj"][task][alg][is_al]
            # label = get_label(alg, is_al)
            style = get_style(alg, is_al)
            ax.bar(x_positions[j], rho_proj, **style)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([])
        ax.set_ylabel("Rho Projection")
        ax.set_title(task, fontsize=8)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="blue", alpha=0.25, label="EKF (Random)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="blue", alpha=0.7, label="EKF (Active)"),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="orange", alpha=0.25, label="Ensemble (Random)"
        ),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="orange", alpha=0.7, label="Ensemble (Active)"
        ),
    ]

    fig.suptitle(f"Rho Projection ({nt_test} test trajs)", fontsize=16)
    fig.legend(handles=legend_elements, loc="center right")
    plt.tight_layout(rect=[0, 0, 0.87, 1])
    plt.savefig(f"{save_dir}/rho_projection_{timestamp}.png")
    plt.close()


def compute_rho(
    ret: Float[Array, "N"],
    normalize_ret: bool = True,
) -> Float[Array, "N"]:
    if normalize_ret:
        ret = normalize(ret, axis=(0,))
    rho = jnp.exp(jax.nn.log_softmax(ret, axis=0))
    return rho


if __name__ == "__main__":
    main()
