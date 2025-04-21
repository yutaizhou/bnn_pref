import os
from collections import defaultdict
from functools import partial
from typing import Dict, Tuple

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
import logging
from datetime import datetime

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from jaxtyping import Array, Float

from bnn_pref.data import dataset_creators
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.data.pref_utils import QueryIndexAndResponses
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import MeanStd
from bnn_pref.utils.utils import get_random_seed, nested_defaultdict
from scripts.sweep_tasks_ekf import run_ekf
from scripts.sweep_tasks_ensemble import run_ensemble

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


def modify_queries(
    pref_data: QueryIndexAndResponses,
    real_frac: float,
    nq_train: int,
    nq_init: int,
) -> Tuple[QueryIndexAndResponses, int]:
    """
    sanity check for active learning acquisition functions
    modify all queries past nq_init: 5% real, and rest duplicate.
    """
    queries_Q2, responses_Q1 = pref_data.queries_Q2, pref_data.responses_Q1
    pool_size = nq_train - nq_init
    n_dups = int(pool_size * (1 - real_frac))
    n_reals = pool_size - n_dups
    dup_queries = jnp.tile(queries_Q2[nq_init + n_reals], (n_dups, 1))
    dup_responses = jnp.tile(responses_Q1[nq_init + n_reals], (n_dups, 1))
    new_queries_Q2 = queries_Q2.at[-n_dups:].set(dup_queries)
    new_responses_Q1 = responses_Q1.at[-n_dups:].set(dup_responses)

    new_pref_data = pref_data.replace(
        queries_Q2=new_queries_Q2,
        responses_Q1=new_responses_Q1,
    )

    return new_pref_data, n_dups


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    seed = get_random_seed() if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed)

    tasks = [
        "reacher",
        "lunar",
        "cheetah",
        "acrobot",
        "ball",
        "cartpoleSwing",
        "cheetahDMC",
        "hopperHop",
        "pendulum",
        "reacherEasy",
        "reacherHard",
        "walkerWalk",
        # "ogbench",
    ]

    stats = nested_defaultdict()

    data_cfg = cfg["data"]
    ekf_cfg = cfg["ekf"]
    sgd_cfg = cfg["sgd"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    nq_init, nsteps = data_cfg["nq_init"], data_cfg["nsteps"]

    warm_burns = ekf_cfg["warm_burns"]
    thinning = ekf_cfg["thinning"]
    sub_dim = ekf_cfg["sub_dim"]
    rnd_proj = ekf_cfg["rnd_proj"]
    n_eff_iterates = (ekf_cfg["niters"] - warm_burns) // thinning

    print(
        f"Run:\n"
        f"  Seed: {seed} x {cfg['seeds']}\n"
        f"  Sanity: {cfg['sanity']} ({cfg['sanity_frac']} real frac)\n"
        f"Data:\n"
        f"  prune: {data_cfg['n_bins']} bins, {data_cfg['max_count_per_bin']} max_count_per_bin, {data_cfg['tokeep']} tokeep\n"
        f"  noisy_label: {data_cfg['noisy_label']} (beta={data_cfg['bt_beta']})\n"
        f"  Train/Test: {nq_train}/{nq_test}\n"
        f"  Init/Update: {nq_init}/{nsteps}\n"
        f"EKF:\n"
        f"  n_models={ekf_cfg['M']}, sub_dim={ekf_cfg['sub_dim']}, rnd_proj={ekf_cfg['rnd_proj']}\n"
        f"  prior / dynamics / obs noise: {ekf_cfg['prior_noise']} / {ekf_cfg['dynamics_noise']} / {ekf_cfg['obs_noise']}\n"
        f"  init: bs={ekf_cfg['bs']}, niters={ekf_cfg['niters']}[{warm_burns}::{thinning}] ({n_eff_iterates} eff), {sub_dim=}, {rnd_proj=}\n"
        f"Ensemble:\n"
        f"  n_models={sgd_cfg['M']}\n"
        f"  init: bs={sgd_cfg['bs']}, niters={sgd_cfg['niters']}\n"
    )

    for task in tasks:
        # * update cfg
        new_cfg = hydra.compose("config", overrides=[f"task={task}"])
        cfg["task"].update(new_cfg["task"])

        # * create dataset
        key, key_data, *key_seeds = jr.split(key, 2 + cfg["seeds"])
        start = datetime.now()
        data_dict = dataset_creators[cfg["task"]["ds_type"]](key_data, cfg)
        duration = (datetime.now() - start).total_seconds()

        # * create env
        train_trajs, test_trajs = data_dict["train_trajs"], data_dict["test_trajs"]
        train_prefs, test_prefs = data_dict["train_prefs"], data_dict["test_prefs"]
        train_trajs_obs = train_trajs["observations"]  # (N, T, D)

        nt_train, T, D = train_trajs_obs.shape
        nt_test = test_trajs["observations"].shape[0]
        nq_train = train_prefs.queries_Q2.shape[0]
        nq_test = test_prefs.queries_Q2.shape[0]

        n_dups = 0
        if cfg["sanity"]:
            train_prefs, n_dups = modify_queries(
                train_prefs,
                real_frac=cfg["sanity_frac"],
                nq_train=nq_train,
                nq_init=nq_init,
            )
        mislabel_ratio = train_prefs.n_mislabels / nq_train
        print(
            f"{task:13} ({T=}, {D=}): train/test nt=({nt_train}/{nt_test}), nq=({nq_train}/{nq_test}), {train_prefs.n_mislabels} train mislabels ({mislabel_ratio:.2%}), {n_dups} dups"
        )

        env = PreferenceEnv(
            items=train_trajs_obs,
            X=train_prefs.queries_Q2,
            Y=jax.nn.one_hot(train_prefs.responses_Q1.squeeze(), num_classes=2),
        )
        # * run
        for alg, run_fn in [("ekf", run_ekf), ("sgd", run_ensemble)]:
            for is_al in [False, True]:
                cfg[alg]["active"] = is_al

                seeds = jnp.array(key_seeds)
                run_fn = partial(run_fn, cfg=cfg, data_dict=data_dict, env=env)

                # run in vmap or lax version (parallel vs. sequential)
                res_m, metadata_m = jax.vmap(run_fn, in_axes=(0,))(seeds)
                # res_m, metadata_m = jax.lax.map(run_fn, seeds)

                # (n_seeds, nq_update)
                res = {
                    "task": task,
                    "active": is_al,
                    "test_logpdf_all": res_m["test_logpdf"],
                    "test_acc_all": res_m["test_acc"],
                    # acc
                    # "train_acc_warm": MeanStd(res_m["train_acc"][:, 0]),
                    # "train_acc": MeanStd(res_m["train_acc"][:, -1]),
                    "test_acc_warm": MeanStd(res_m["test_acc"][:, 0]),
                    "test_acc": MeanStd(res_m["test_acc"][:, -1]),
                    # "test_acc_bma": MeanStd(res_m["test_acc_bma"][:, -1])
                    # if alg == "ekf"
                    # else None,
                    # logpdf
                    # "train_logpdf_warm": MeanStd(res_m["train_logpdf"][:, 0]),
                    # "train_logpdf": MeanStd(res_m["train_logpdf"][:, -1]),
                    "test_logpdf_warm": MeanStd(res_m["test_logpdf"][:, 0]),
                    "test_logpdf": MeanStd(res_m["test_logpdf"][:, -1]),
                    # * metadata
                    "nq_train": nq_train,
                    "nq_test": nq_test,
                }

                stats[task][alg][is_al] = res

                print(
                    f"  {alg} active={str(is_al):5}, "
                    f"acc: {res['test_acc'].mean:.2%} ± {res['test_acc'].std:.2%}, "
                    # f"acc: {res['test_acc_warm']:.2%} ± {res['test_acc_warm_std']:.2%} -> {res['test_acc']:.2%} ± {res['test_acc_std']:.2%}, "
                    f"logpdf: {res['test_logpdf'].mean:.2f} ± {res['test_logpdf'].std:.2f}, "
                    # f"({metadata_m['full_param_count'][0]:,d} -> {metadata_m['subspace_param_count'][0]:,d}) "
                    # f", bma_acc: {res['test_acc_bma'].mean:.2%} ± {res['test_acc_bma'].std:.2%}"
                )

    # * plot logpdf learning curve
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    axs = axs.flatten()

    def find_min_logpdf(stats: Dict, task: str) -> float:
        """
        stats[task][alg][is_al]["test_logpdf_all"]
        """
        best_min = jnp.inf
        for alg in ["ekf", "sgd"]:
            for is_al in [False, True]:
                curr_min = jnp.min(stats[task][alg][is_al]["test_logpdf_all"])
                best_min = jnp.minimum(best_min, curr_min)
        if best_min is -jnp.inf:
            best_min = -4  # rougly 1.8% accuracy / probability of t2 > t1
        return best_min

    def get_label(alg: str, is_al: bool) -> str:
        if alg == "ekf":
            return "EKF (Active)" if is_al else "EKF (Random)"
        else:
            return "Ensemble (Active)" if is_al else "Ensemble (Random)"

    def get_style(alg: str, is_al: bool) -> dict:
        color = "blue" if alg == "ekf" else "orange"
        linestyle = "-" if is_al else "--"
        return {"color": color, "linestyle": linestyle, "linewidth": 1}

    # * plot logpdf eval curve
    for i, task in enumerate(tasks):
        ax = axs[i]
        # y_min = find_min_logpdf(stats, task)
        # ax.set_ylim(y_min, 0)
        ax.set_ylim(-1.25, 0)
        ax.axhline(y=-0.69, linestyle=":", linewidth=1, color="red")
        for alg in ["ekf", "sgd"]:
            for is_al in [False, True]:
                # (n_seeds, nq_update)
                values = stats[task][alg][is_al]["test_logpdf_all"]
                if jnp.isinf(values).any():
                    continue
                label = get_label(alg, is_al)
                style = get_style(alg, is_al)
                ax.plot(values.mean(0), label=label, **style)
                ax.fill_between(
                    jnp.arange(values.shape[1]),
                    values.mean(0) - values.std(0),
                    values.mean(0) + values.std(0),
                    alpha=0.2,
                    **style,
                )

        task_nq_train = stats[task][alg][is_al]["nq_train"]
        task_nq_test = stats[task][alg][is_al]["nq_test"]
        ax.set_title(f"{task} (nq={task_nq_train}/{task_nq_test})", fontsize=8)

    dummy_lines = [
        plt.plot([], [], color="blue", linestyle="--", label="EKF (Random)")[0],
        plt.plot([], [], color="blue", linestyle="-", label="EKF (Active)")[0],
        plt.plot([], [], color="orange", linestyle="--", label="Ensemble (Random)")[0],
        plt.plot([], [], color="orange", linestyle="-", label="Ensemble (Active)")[0],
    ]
    fig.legend(
        dummy_lines,
        ["EKF (Random)", "EKF (Active)", "Ensemble (Random)", "Ensemble (Active)"],
        loc="center right",
    )
    fig.suptitle(
        f"log PDF vs. num queries (noise={data_cfg['noisy_label']}, sanity={cfg['sanity']})"
    )
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # [left, bottom, right, top]
    # plt.show()
    plt.savefig(f"{cfg.paths.output_dir}/logpdf_vs_queries.png")

    # * plot acc eval curve
    fig, axs = plt.subplots(4, 4, figsize=(12, 8))  # 13 tasks total
    axs = axs.flatten()
    for i, task in enumerate(tasks):
        ax = axs[i]
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, linestyle=":", linewidth=1, color="red")
        for alg in ["ekf", "sgd"]:
            for is_al in [False, True]:
                values = stats[task][alg][is_al]["test_acc_all"]  # (n_seeds, nq_update)
                label = get_label(alg, is_al)
                style = get_style(alg, is_al)
                ax.plot(values.mean(0), label=label, **style)
                ax.fill_between(
                    jnp.arange(values.shape[1]),
                    values.mean(0) - values.std(0),
                    values.mean(0) + values.std(0),
                    alpha=0.2,
                    **style,
                )
        ax.set_title(f"{task}")

    dummy_lines = [
        plt.plot([], [], color="blue", linestyle="--", label="EKF (Random)")[0],
        plt.plot([], [], color="blue", linestyle="-", label="EKF (Active)")[0],
        plt.plot([], [], color="orange", linestyle="--", label="Ensemble (Random)")[0],
        plt.plot([], [], color="orange", linestyle="-", label="Ensemble (Active)")[0],
    ]
    fig.legend(
        dummy_lines,
        ["EKF (Random)", "EKF (Active)", "Ensemble (Random)", "Ensemble (Active)"],
        loc="center right",
    )
    fig.suptitle(f"Accuracy vs. num queries ({nq_train})")
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # [left, bottom, right, top]
    # plt.show()
    plt.savefig(f"{cfg.paths.output_dir}/acc_vs_queries.png")


if __name__ == "__main__":
    main()
