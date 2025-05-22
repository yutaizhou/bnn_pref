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
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from hydra.core.hydra_config import HydraConfig

from bnn_pref.alg.trainer import run_ekf, run_ensemble
from bnn_pref.data import dataset_creators
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.data.pref_utils import QueryIndexAndResponses
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import MeanStd
from bnn_pref.utils.print_utils import get_param_count_msg
from bnn_pref.utils.utils import get_random_seed, nested_defaultdict, slurm_auto_scancel

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


def modify_queries(
    pref_data: QueryIndexAndResponses,
    real_frac: float,
    nq_train: int,
    nq_init: int,
) -> Tuple[QueryIndexAndResponses, int]:
    """
    Sanity check for active learning acquisition functions
    Modify all queries past nq_init: 5% real, and rest duplicate.
    This should hinder performance of random querying, but not active querying.
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


@hydra.main(version_base=None, config_name="configPref", config_path="../cfg")
def main(cfg):
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)
    run_fns = {
        "ekf": run_ekf,
        "sgd": run_ensemble,
    }
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
    # algs = ["sgd"]
    # algs = ["ekf"]
    is_als = [False, True]
    # is_als = [False]
    # is_als = [True]

    stats = nested_defaultdict()

    data_cfg = cfg["data"]
    ekf_cfg = cfg["ekf"]
    sgd_cfg = cfg["sgd"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    nq_init, nsteps = data_cfg["nq_init"], data_cfg["nsteps"]
    n_eff_iterates = (ekf_cfg["niters"] - ekf_cfg["warm_burns"]) // ekf_cfg["thinning"]

    print(
        f"Run:\n"
        f"  Seed: {seed} x {cfg['seeds']} (seed_vmap={cfg['seed_vmap']})\n"
        f"  Sanity: {cfg['sanity']} ({cfg['sanity_frac']} real frac)\n"
        f"  Network: {cfg['network']['hidden_sizes']}\n"
        f"Data:\n"
        f"  prune: {data_cfg['n_bins']} bins, {data_cfg['max_count_per_bin']} max_count_per_bin, {data_cfg['tokeep']} tokeep\n"
        f"  noisy_label: {data_cfg['noisy_label']} (beta={data_cfg['bt_beta']})\n"
        f"  Train/Test: {nq_train}/{nq_test}\n"
        f"  Init/Update: {nq_init}/{nsteps}\n"
        f"EKF:\n"
        f"  M={ekf_cfg['M']}, use_vmap={ekf_cfg['use_vmap']}\n"
        f"  prior / dynamics / obs noise: {ekf_cfg['prior_noise']} / {ekf_cfg['dynamics_noise']} / {ekf_cfg['obs_noise']}\n"
        f"  init: bs={ekf_cfg['bs']}, niters={ekf_cfg['niters']}[{ekf_cfg['warm_burns']}::{ekf_cfg['thinning']}] ({n_eff_iterates} eff), sub_dim={ekf_cfg['sub_dim']}, rnd_proj={ekf_cfg['rnd_proj']}\n"
        f"Ensemble:\n"
        f"  M={sgd_cfg['M']}, use_vmap={sgd_cfg['use_vmap']}\n"
        f"  init: bs={sgd_cfg['bs']}, niters={sgd_cfg['niters']}\n"
    )

    ckpter = ocp.PyTreeCheckpointer()
    total_duration = datetime.now()
    for task in tasks:
        # * update cfg
        new_cfg = hydra.compose("config", overrides=[f"task={task}"])
        cfg["task"].update(new_cfg["task"])

        # * create dataset
        key, key_data = jr.split(key, 2)
        data_dict = dataset_creators[cfg["task"]["ds_type"]](key_data, cfg)

        # * create env
        train_trajs, test_trajs = data_dict["train_trajs"], data_dict["test_trajs"]
        train_prefs, test_prefs = data_dict["train_prefs"], data_dict["test_prefs"]

        nt_train, T, D = train_trajs["observations"].shape
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
            f"{task:13} ({T=}, {D=}): train/test nt=({nt_train}/{nt_test}), nq=({nq_train}/{nq_test}), {train_prefs.n_mislabels} mislabels ({mislabel_ratio:.1%}), {n_dups} dups"
        )

        env = PreferenceEnv(
            items=train_trajs["observations"],
            X=train_prefs.queries_Q2,
            Y=jax.nn.one_hot(train_prefs.responses_Q1.squeeze(), num_classes=2),
        )
        # * run
        key, *key_seeds = jr.split(key, 1 + cfg["seeds"])
        seeds = jnp.array(key_seeds)
        for alg, is_al in it.product(algs, is_als):
            cfg[alg]["active"] = is_al

            run_fn = run_fns[alg]
            run_fn = partial(run_fn, cfg=cfg, data_dict=data_dict, env=env)

            # run in vmap or lax version (parallel vs. sequential)
            start = datetime.now()

            res_m = (
                jax.block_until_ready(jax.vmap(run_fn)(seeds))
                if cfg["seed_vmap"]
                else jax.block_until_ready(jax.lax.map(run_fn, seeds))
            )

            duration = (datetime.now() - start).total_seconds()

            # (n_seeds, 1 + nq_update)
            res = {
                "task": task,
                "task_name": cfg["task"]["name"],
                "is_active": is_al,
                "nq_train": nq_train,
                "nq_test": nq_test,
                "duration": duration,
                # * logpdf
                "test_logpdf_all": res_m["test_logpdf"],
                "test_logpdf_final": MeanStd(res_m["test_logpdf"][:, -1]).get_stats(),
                # * acc
                "test_acc_all": res_m["test_acc"],
                "test_acc_final": MeanStd(res_m["test_acc"][:, -1]).get_stats(),
            }
            best_seed = jnp.argmax(res_m["test_logpdf"][:, -1])
            best_model = jax.tree.map(lambda x: x[best_seed], res_m["model"])

            # * save best model
            save_args = orbax_utils.save_args_from_target(best_model)
            ckpt_name = f"{cfg['task']['name']}_{alg}_al={is_al}"
            ckpter.save(
                f"{cfg.paths.ckpts_dir}/{ckpt_name}",
                best_model,
                save_args=save_args,
            )

            stats[task][alg][is_al] = res
            test_logpdf_all = res_m["test_logpdf"]
            nans = ~jnp.isfinite(test_logpdf_all)

            print(
                f"  {alg} active={str(is_al):5}, "
                f"acc: {res['test_acc_final']['mean']:.2%} ± {res['test_acc_final']['std']:.2%}, "
                f"logpdf: {res['test_logpdf_final']['mean']:.2f} ± {res['test_logpdf_final']['std']:.2f}; "
                f"{get_param_count_msg(cfg, alg, res_m)}, "
                f"({res['duration']:.1f}s)"
            )
            if nans.any():
                print(f"nans: {nans.sum(1)}")
    total_duration = (datetime.now() - total_duration).total_seconds()
    print(f"Total duration: {total_duration:.1f}s")
    jnp.savez(f"{cfg.paths.output_dir}/stats.npz", **stats)

    # * plot logpdf learning curve
    fig, axs = plt.subplots(5, 4, figsize=(12, 10))
    axs = axs.flatten()

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
        ax.set_ylim(-0.73, 0)  # ln(0.48)
        ax.axhline(y=-0.69, linestyle=":", linewidth=1, color="red")  # ln(0.5) = -0.69
        for alg, is_al in it.product(algs, is_als):
            # (n_seeds, 1 + nq_update)
            stat = stats[task][alg][is_al]
            values = stat["test_logpdf_all"]
            nans = ~jnp.isfinite(values)
            if nans.any():
                continue
            label = get_label(alg, is_al)
            style = get_style(alg, is_al)
            mean, std = values.mean(0), values.std(0)
            ax.plot(mean, label=label, **style)
            ax.fill_between(
                jnp.arange(len(mean)), mean - std, mean + std, alpha=0.2, **style
            )

        task_name = stat["task_name"]
        ax.set_title(
            f"{task_name} (nq={stat['nq_train']}/{stat['nq_test']})", fontsize=8
        )

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
    fig, axs = plt.subplots(5, 4, figsize=(12, 10))
    axs = axs.flatten()
    for i, task in enumerate(tasks):
        ax = axs[i]
        ax.set_ylim(0.48, 1)
        ax.axhline(y=0.5, linestyle=":", linewidth=1, color="red")
        for alg, is_al in it.product(algs, is_als):
            stat = stats[task][alg][is_al]
            values = stat["test_acc_all"]  # (n_seeds, nq_update)
            label = get_label(alg, is_al)
            style = get_style(alg, is_al)
            mean, std = values.mean(0), values.std(0)
            ax.plot(mean, label=label, **style)
            ax.fill_between(
                jnp.arange(len(mean)), mean - std, mean + std, alpha=0.2, **style
            )
        task_name = stat["task_name"]
        ax.set_title(f"{task_name}")

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

    # * bar plot duration for each task
    fig, axs = plt.subplots(5, 4, figsize=(12, 10))
    axs = axs.flatten()

    # Update legend elements to match logpdf plot style
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="blue", hatch="//", label="EKF (Random)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="blue", label="EKF (Active)"),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="orange", hatch="//", label="Ensemble (Random)"
        ),
        plt.Rectangle((0, 0), 1, 1, facecolor="orange", label="Ensemble (Active)"),
    ]

    for i, task in enumerate(tasks):
        ax = axs[i]
        for j, (alg, is_al) in enumerate(it.product(algs, is_als)):
            stat = stats[task][alg][is_al]
            duration = stat["duration"]
            nq_train, nq_test = stat["nq_train"], stat["nq_test"]
            color = "blue" if alg == "ekf" else "orange"
            bar = ax.bar(j, duration, color=color)
            if not is_al:
                bar.patches[0].set_hatch("//")
        ax.set_xticks([])
        # ax.set_ylabel("Duration (s)")
        task_name = stat["task_name"]
        ax.set_title(f"{task_name} (nq={nq_train}/{nq_test})", fontsize=8)

    fig.suptitle("Task Duration (s)")
    fig.legend(handles=legend_elements, loc="center right")
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f"{cfg.paths.output_dir}/task_durations.png")


if __name__ == "__main__":
    main()
    slurm_auto_scancel()  # prevent completed jobs from hanging on slurm
