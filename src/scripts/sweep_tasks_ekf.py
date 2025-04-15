import itertools as it
import os
from collections import defaultdict
from dataclasses import dataclass

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
import logging
from datetime import datetime
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig

from bnn_pref.alg.ekf_subspace import SubspaceNeuralEKF
from bnn_pref.alg.trainer import alg_pipeline
from bnn_pref.data import dataset_creators
from bnn_pref.data.data_env import DataEnvironment
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import (
    MeanStd,
    compute_acc_nn,
    compute_acc_nn_bma,
    compute_logpdf_nn,
)
from bnn_pref.utils.utils import get_random_seed

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


def run_ekf(key, cfg, data_dict, env):
    ekf_cfg = cfg["ekf"]
    data_cfg = cfg["data"]
    train_prefs, test_prefs = data_dict["train_prefs"], data_dict["test_prefs"]

    # * build + run bandit alg
    key, key_pipe, key_bma = jr.split(key, 3)
    bel_trace, bandit = alg_pipeline(
        key_pipe, SubspaceNeuralEKF, env, ekf_cfg, data_cfg
    )

    # * compute metrics
    sub2full_logits_fn = bandit.sub2full_predict_logits  # (params, N2TD) -> (N2,)

    def eval_bel(_, bel):
        mean, cov, t = bel
        key = jr.fold_in(key_bma, t)
        fn = partial(sub2full_logits_fn, mean)
        # train_logpdf = compute_logpdf_nn(fn, train_prefs)
        test_logpdf = compute_logpdf_nn(fn, test_prefs)
        # train_acc = compute_acc_nn(fn, train_prefs)
        test_acc = compute_acc_nn(fn, test_prefs)
        # train_acc_bma = compute_acc_nn_bma(key, sub2full_logits_fn, bel, train_prefs)
        test_acc_bma = compute_acc_nn_bma(key, sub2full_logits_fn, bel, test_prefs)

        # all arrays of (1 + nq_updates, )
        result = {
            # * logpdf
            # "train_logpdf": train_logpdf,
            "test_logpdf": test_logpdf,
            # * acc
            # "train_acc": train_acc,
            "test_acc": test_acc,
            # "train_acc_bma": train_acc_bma,
            "test_acc_bma": test_acc_bma,
        }
        return (), result

    *_, al_results = jax.lax.scan(eval_bel, init=(), xs=bel_trace)

    results = al_results
    metadata = {
        "full_param_count": bandit.full_params_count,
        "subspace_param_count": bandit.subspace_params_count,
    }
    return results, metadata


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
        "ogbench",
    ]
    stats = defaultdict(lambda: defaultdict(dict))

    data_cfg = cfg["data"]
    ekf_cfg = cfg["ekf"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    nq_init, nsteps = data_cfg["nq_init"], data_cfg["nsteps"]
    batch_size = ekf_cfg["bs"]
    niters = ekf_cfg["niters"]
    warm_burns = ekf_cfg["warm_burns"]
    thinning = ekf_cfg["thinning"]
    sub_dim = ekf_cfg["sub_dim"]
    rnd_proj = ekf_cfg["rnd_proj"]
    n_eff_iterates = (niters - warm_burns) // thinning
    print(
        f"Seed: {seed} x {cfg['seeds']}\n"
        f"Data:\n"
        f"  Train/Test: {nq_train}/{nq_test}\n"
        f"  Init/Update: {nq_init}/{nsteps}\n"
        f"EKF:\n"
        f"  sub_dim={ekf_cfg['sub_dim']}, rnd_proj={ekf_cfg['rnd_proj']}\n"
        f"  init: bs={batch_size}, niters={niters}[{warm_burns}::{thinning}] ({n_eff_iterates} eff), {sub_dim=}, {rnd_proj=}\n"
    )

    for task in tasks:
        print(f"{task}: ")
        key, key_data, *key_seeds = jr.split(key, 2 + cfg["seeds"])
        data_dict = dataset_creators[cfg["task"]["ds_type"]](key_data, cfg)
        train_prefs = data_dict["train_prefs"]
        env = DataEnvironment(
            X=train_prefs.queries_Q2TD,
            Y=jax.nn.one_hot(train_prefs.responses_Q1.squeeze(), num_classes=2),
        )
        for is_al in [False, True]:
            # * update cfg
            new_cfg = hydra.compose(
                "config",
                overrides=[f"task={task}", f"ekf.active={is_al}"],
            )
            cfg["task"].update(new_cfg["task"])
            cfg["ekf"]["active"] = is_al

            # * run
            seeds = jnp.array(key_seeds)
            run_vmap = jax.vmap(run_ekf, in_axes=(0, None, None, None))
            start_time = datetime.now()
            res_m, metadata_m = run_vmap(seeds, cfg, data_dict, env)
            duration = (datetime.now() - start_time).total_seconds()

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
                "test_acc_bma": MeanStd(res_m["test_acc_bma"][:, -1]),
                # logpdf
                # "train_logpdf_warm": MeanStd(res_m["train_logpdf"][:, 0]),
                # "train_logpdf": MeanStd(res_m["train_logpdf"][:, -1]),
                "test_logpdf_warm": MeanStd(res_m["test_logpdf"][:, 0]),
                "test_logpdf": MeanStd(res_m["test_logpdf"][:, -1]),
            }

            stats[task][is_al] = res

            print(
                f"  active={str(is_al):5}, "
                f"acc: {res['test_acc'].mean:.2%} ± {res['test_acc'].std:.2%}, "
                # f"acc: {res['test_acc_warm']:.2%} ± {res['test_acc_warm_std']:.2%} -> {res['test_acc']:.2%} ± {res['test_acc_std']:.2%}, "
                f"logpdf: {res['test_logpdf'].mean:.2f} ± {res['test_logpdf'].std:.2f}, "
                # f"({metadata_m['full_param_count'][0]:,d} -> {metadata_m['subspace_param_count'][0]:,d}) "
                f"({duration:.1f}s)"
                f", bma_acc: {res['test_acc_bma'].mean:.2%} ± {res['test_acc_bma'].std:.2%}, "
            )

    # print("\n === Printing extra stats ===")
    # for stat in stats:
    #     print(
    #         f"{stat['task']} ({duration:.1f}s):\n"
    #         f"  ({metadata_m['full_param_count'][0]:,d} -> {metadata_m['subspace_param_count'][0]:,d})\n"
    #         f"  Train acc:    {stat['train_acc_warm']:.2%} ± {stat['train_acc_warm_std']:.2%} -> {stat['train_acc']:.2%} ± {stat['train_acc_std']:.2%}\n"
    #         f"  Acc:          {stat['test_acc_warm']:.2%} ± {stat['test_acc_warm_std']:.2%} -> {stat['test_acc']:.2%} ± {stat['test_acc_std']:.2%}\n"
    #         f"  Acc BMA:      {stat['test_acc_bma']:.2%} ± {stat['test_acc_bma_std']:.2%}\n"
    #         f"  Train logpdf: {stat['train_logpdf_warm']:.2f} ± {stat['train_logpdf_warm_std']:.2f} -> {stat['train_logpdf']:.2f} ± {stat['train_logpdf_std']:.2f}\n"
    #         f"  logpdf:       {stat['test_logpdf_warm']:.2f} ± {stat['test_logpdf_warm_std']:.2f} -> {stat['test_logpdf']:.2f} ± {stat['test_logpdf_std']:.2f}\n"
    #     )

    # * plot logpdf learning curve
    fig, axs = plt.subplots(4, 4, figsize=(12, 8))  # 13 tasks total
    axs = axs.flatten()

    for i, task in enumerate(tasks):
        ax = axs[i]
        y_min = min(stat["test_logpdf_all"].min() for stat in stats[task].values())
        ax.set_ylim(y_min, 0)
        for is_al in [False, True]:
            values = stats[task][is_al]["test_logpdf_all"]  # (n_seeds, nq_update)
            ax.plot(values.mean(0), label="Active" if is_al else "Random")
            ax.fill_between(
                jnp.arange(values.shape[1]),
                values.mean(0) - values.std(0),
                values.mean(0) + values.std(0),
                alpha=0.2,
            )
        ax.set_title(f"{task}")

    dummy_lines = [plt.plot([], [], label=label)[0] for label in ["Random", "Active"]]
    fig.legend(dummy_lines, ["Random", "Active"], loc="center right")
    fig.suptitle("log PDF vs. num queries ")
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # [left, bottom, right, top]
    plt.show()

    # * plot acc learning curve
    fig, axs = plt.subplots(4, 4, figsize=(12, 8))  # 13 tasks total
    axs = axs.flatten()
    for i, task in enumerate(tasks):
        ax = axs[i]
        ax.set_ylim(0, 1)
        for is_al in [False, True]:
            values = stats[task][is_al]["test_acc_all"]  # (n_seeds, nq_update)
            ax.plot(values.mean(0), label="Active" if is_al else "Random")
            ax.fill_between(
                jnp.arange(values.shape[1]),
                values.mean(0) - values.std(0),
                values.mean(0) + values.std(0),
                alpha=0.2,
            )
        ax.set_title(f"{task}")

    dummy_lines = [plt.plot([], [], label=label)[0] for label in ["Random", "Active"]]
    fig.legend(dummy_lines, ["Random", "Active"], loc="center right")
    fig.suptitle("Accuracy vs. num queries")
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # [left, bottom, right, top]
    plt.show()


if __name__ == "__main__":
    main()
