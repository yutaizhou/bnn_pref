import itertools as it
import os
from functools import partial
from typing import Tuple

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

from bnn_pref.alg.trainer import run_ekf, run_ensemble
from bnn_pref.data import dataset_creators
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import MeanStd
from bnn_pref.utils.print_utils import get_param_count_msg
from bnn_pref.utils.utils import get_random_seed, nested_defaultdict
from scripts.sweep_tasks_alg import modify_queries

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="configScaling", config_path="../cfg")
def main(cfg):
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)
    run_fns = {
        "ekf": run_ekf,
        "sgd": run_ensemble,
    }

    task = "cheetahMediumExpert"
    algs = ["ekf", "sgd"]

    M = 10
    networks = [
        "32x2",  # 1665, calculated wrt traj_shape = (50,17)
        "64x2",  # 5377
        "64x3",  # 9537
        "128x2",  # 18945
        "128x3",  # 35457
        "256x2",  # 70657
        "256x3",  # 136449
        "512x2",  # 272385
        "512x3",  # 535041
        "1024x2",  # 1069057
        "1024x3",  # 2118657
    ]

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
        # f"  Network: {cfg['network']['hidden_sizes']}\n"
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
        # combine these two loops into one
        for alg in algs:
            for network in networks:
                new_cfg = hydra.compose(
                    "config", overrides=[f"task={task}", f"network={network}"]
                )
                cfg["network"].update(new_cfg["network"])
                cfg["ekf"]["hidden_sizes"] = new_cfg["ekf"]["hidden_sizes"]
                cfg["sgd"]["hidden_sizes"] = new_cfg["sgd"]["hidden_sizes"]

                start = datetime.now()

                run_fn = partial(run_fns[alg], cfg=cfg, data_dict=data_dict, env=env)
                res_m = (
                    jax.block_until_ready(jax.vmap(run_fn)(seeds))
                    if cfg["seed_vmap"]
                    else jax.block_until_ready(jax.lax.map(run_fn, seeds))
                )

                duration = (datetime.now() - start).total_seconds()

                # (n_seeds, 1 + nq_update)
                res = {
                    "task": task,
                    "nq_train": nq_train,
                    "nq_test": nq_test,
                    "duration": duration,
                    # * logpdf
                    "test_logpdf_all": res_m["test_logpdf"],
                    "test_logpdf_final": MeanStd(res_m["test_logpdf"][:, -1]),
                    # * acc
                    "test_acc_all": res_m["test_acc"],
                    "test_acc_final": MeanStd(res_m["test_acc"][:, -1]),
                }

                stats[task][alg][network] = res

                test_logpdf_all = res_m["test_logpdf"]
                nans = ~jnp.isfinite(test_logpdf_all)

                print(
                    f"  {alg} {network}, "
                    f"acc: {res['test_acc_final'].mean:.2%} ± {res['test_acc_final'].std:.2%}, "
                    f"logpdf: {res['test_logpdf_final'].mean:.2f} ± {res['test_logpdf_final'].std:.2f}; "
                    f"{get_param_count_msg(cfg, alg, res_m)}, "
                    f"({res['duration']:.1f}s)"
                )
                if nans.any():
                    print(f"nans: {nans.sum(1)}")
    total_duration = (datetime.now() - total_duration).total_seconds()
    print(f"Total duration: {total_duration:.1f}s")

    # * plot logpdf learning curve
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    axs = axs.flatten()

    def get_label(alg: str, network: str) -> str:
        alg_str = "EKF" if alg == "ekf" else "Ensemble"
        return f"{alg_str} ({network})"

    def get_style(alg: str) -> dict:
        color = "blue" if alg == "ekf" else "orange"
        return {"color": color, "linestyle": "-", "linewidth": 1}

    for i, task in enumerate(tasks):
        ax = axs[i]
        ax.set_ylim(-0.73, 0)  # ln(0.48)
        ax.axhline(y=-0.69, linestyle=":", linewidth=1, color="red")  # ln(0.5) = -0.69
        for alg, network in it.product(algs, networks):
            # (n_seeds, 1 + nq_update)
            stat = stats[task][alg][network]
            values = stat["test_logpdf_all"]
            nans = ~jnp.isfinite(values)
            if nans.any():
                continue
            label = get_label(alg, network)
            style = get_style(alg)
            mean, std = values.mean(0), values.std(0)
            ax.plot(mean, label=label, **style)
            ax.fill_between(
                jnp.arange(len(mean)), mean - std, mean + std, alpha=0.2, **style
            )

        ax.set_title(f"{task} (nq={stat['nq_train']}/{stat['nq_test']})", fontsize=8)

    dummy_lines = [
        plt.plot([], [], color="blue", linestyle="-", label="EKF (small)")[0],
        plt.plot([], [], color="blue", linestyle="-", label="EKF (medium)")[0],
        plt.plot([], [], color="blue", linestyle="-", label="EKF (large)")[0],
        plt.plot([], [], color="orange", linestyle="-", label="Ensemble (small)")[0],
        plt.plot([], [], color="orange", linestyle="-", label="Ensemble (medium)")[0],
        plt.plot([], [], color="orange", linestyle="-", label="Ensemble (large)")[0],
    ]
    fig.legend(
        dummy_lines,
        [
            "EKF (small)",
            "EKF (medium)",
            "EKF (large)",
            "Ensemble (small)",
            "Ensemble (medium)",
            "Ensemble (large)",
        ],
        loc="center right",
    )
    fig.suptitle(
        f"log PDF vs. num queries (noise={data_cfg['noisy_label']}, sanity={cfg['sanity']})"
    )
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # [left, bottom, right, top]
    # plt.show()
    plt.savefig(f"{cfg.paths.output_dir}/logpdf_vs_queries.png")

    # * plot acc eval curve
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    axs = axs.flatten()
    for i, task in enumerate(tasks):
        ax = axs[i]
        ax.set_ylim(0.48, 1)
        ax.axhline(y=0.5, linestyle=":", linewidth=1, color="red")
        for alg, network in it.product(algs, networks):
            # (n_seeds, nq_update)
            stat = stats[task][alg][network]
            values = stat["test_acc_all"]
            label = get_label(alg, network)
            style = get_style(alg)
            mean, std = values.mean(0), values.std(0)
            ax.plot(mean, label=label, **style)
            ax.fill_between(
                jnp.arange(len(mean)), mean - std, mean + std, alpha=0.2, **style
            )
        ax.set_title(f"{task}")

    dummy_lines = [
        plt.plot([], [], color="blue", linestyle="-", label="EKF (small)")[0],
        plt.plot([], [], color="blue", linestyle="-", label="EKF (medium)")[0],
        plt.plot([], [], color="blue", linestyle="-", label="EKF (large)")[0],
        plt.plot([], [], color="orange", linestyle="-", label="Ensemble (small)")[0],
        plt.plot([], [], color="orange", linestyle="-", label="Ensemble (medium)")[0],
        plt.plot([], [], color="orange", linestyle="-", label="Ensemble (large)")[0],
    ]
    fig.legend(
        dummy_lines,
        [
            "EKF (small)",
            "EKF (medium)",
            "EKF (large)",
            "Ensemble (small)",
            "Ensemble (medium)",
            "Ensemble (large)",
        ],
        loc="center right",
    )
    fig.suptitle(f"Accuracy vs. num queries ({nq_train})")
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # [left, bottom, right, top]
    # plt.show()
    plt.savefig(f"{cfg.paths.output_dir}/acc_vs_queries.png")

    # * bar plot: duration for each task
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    axs = axs.flatten()

    # Update legend elements to match logpdf plot style
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="blue", label="EKF"),
        plt.Rectangle((0, 0), 1, 1, facecolor="orange", label="Ensemble"),
    ]

    for i, task in enumerate(tasks):
        ax = axs[i]
        for j, (alg, network) in enumerate(it.product(algs, networks)):
            stat = stats[task][alg][network]
            duration = stat["duration"]
            nq_train, nq_test = stat["nq_train"], stat["nq_test"]
            color = "blue" if alg == "ekf" else "orange"
            bar = ax.bar(j, duration, color=color)
        ax.set_xticks([])
        # ax.set_ylabel("Duration (s)")
        ax.set_title(f"{task} (nq={nq_train}/{nq_test})", fontsize=8)

    fig.suptitle("Task Duration (s)")
    fig.legend(handles=legend_elements, loc="center right")
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f"{cfg.paths.output_dir}/task_durations.png")

    # * plot network size vs. duration
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    axs = axs.flatten()
    for i, task in enumerate(tasks):
        ax = axs[i]
        for alg in algs:
            durations = []
            for j, network in enumerate(networks):
                stat = stats[task][alg][network]
                duration = stat["duration"]
                nq_train, nq_test = stat["nq_train"], stat["nq_test"]
                durations.append(duration)
            style = get_style(alg)
            label = get_label(alg, network)
            ax.plot(durations, label=label, **style)
            ax.set_title(f"{task}")

    fig.suptitle("Task Duration (s) vs. Network Size")
    dummy_lines = [
        plt.plot([], [], color="blue", linestyle="-", label="EKF")[0],
        plt.plot([], [], color="orange", linestyle="-", label="Ensemble")[0],
    ]
    fig.legend(dummy_lines, ["EKF", "Ensemble"], loc="center right")
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f"{cfg.paths.output_dir}/network_size_vs_duration.png")


if __name__ == "__main__":
    main()
