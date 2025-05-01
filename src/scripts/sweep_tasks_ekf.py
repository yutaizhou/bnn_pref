import itertools as it
import os
from functools import partial

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

from bnn_pref.alg.trainer import run_ekf
from bnn_pref.data import dataset_creators
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import MeanStd
from bnn_pref.utils.utils import get_random_seed, nested_defaultdict

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="configPref", config_path="../cfg")
def main(cfg):
    seed = get_random_seed(cfg["seed"])
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
        # "reacherEasy",
        # "reacherHard",
        "walkerWalk",
    ]
    stats = nested_defaultdict()

    data_cfg = cfg["data"]
    ekf_cfg = cfg["ekf"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    nq_init, nsteps = data_cfg["nq_init"], data_cfg["nsteps"]
    niters = ekf_cfg["niters"]
    warm_burns = ekf_cfg["warm_burns"]
    thinning = ekf_cfg["thinning"]
    sub_dim = ekf_cfg["sub_dim"]
    n_eff_iterates = (niters - warm_burns) // thinning
    print(
        f"Run:\n"
        f"  Seed: {seed} x {cfg['seeds']} (seed_vmap={cfg['seed_vmap']})\n"
        f"  Sanity: {cfg['sanity']} ({cfg['sanity_frac']} real frac)\n"
        f"Data:\n"
        f"  prune: {data_cfg['n_bins']} bins, {data_cfg['max_count_per_bin']} max_count_per_bin, {data_cfg['tokeep']} tokeep\n"
        f"  noisy_label: {data_cfg['noisy_label']} (beta={data_cfg['bt_beta']})\n"
        f"  Train/Test: {nq_train}/{nq_test}\n"
        f"  Init/Update: {nq_init}/{nsteps}\n"
        f"EKF:\n"
        f"  prior / dynamics / obs noise: {ekf_cfg['prior_noise']} / {ekf_cfg['dynamics_noise']} / {ekf_cfg['obs_noise']}\n"
        f"  n_models={ekf_cfg['M']}, use_vmap={ekf_cfg['use_vmap']}\n"
        f"  init: bs={ekf_cfg['bs']}, niters={ekf_cfg['niters']}[{warm_burns}::{thinning}] ({n_eff_iterates} eff), {sub_dim=}, rnd_proj={ekf_cfg['rnd_proj']}\n"
    )

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

        print(
            f"{task:13} ({T=}, {D=}): train/test nt=({nt_train}/{nt_test}), nq=({nq_train}/{nq_test}), {train_prefs.n_mislabels} mislabels ({train_prefs.n_mislabels / nq_train:.1%})"
        )
        env = PreferenceEnv(
            items=train_trajs["observations"],
            X=train_prefs.queries_Q2,
            Y=jax.nn.one_hot(train_prefs.responses_Q1.squeeze(), num_classes=2),
        )

        # * run
        key, *key_seeds = jr.split(key, 1 + cfg["seeds"])
        seeds = jnp.array(key_seeds)
        for is_al, ekf_vmap in it.product([False, True], [False, True]):
            cfg["ekf"]["active"] = is_al
            cfg["ekf"]["use_vmap"] = ekf_vmap

            # run in vmap or lax version (parallel vs. sequential)
            run_fn = partial(run_ekf, cfg=cfg, data_dict=data_dict, env=env)
            start = datetime.now()

            res_m = (
                jax.block_until_ready(jax.vmap(run_fn)(seeds))
                if cfg["seed_vmap"]
                else jax.block_until_ready(jax.lax.map(run_fn, seeds))
            )

            duration = (datetime.now() - start).total_seconds()

            res = {
                "task": task,
                "active": is_al,
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

            stats[task][is_al] = res
            param_count = res_m["param_count"][0].item()
            subspace_param_count = res_m["subspace_param_count"][0].item()

            print(
                f"  active={str(is_al):5}, vmap={str(ekf_vmap):5}, "
                f"acc: {res['test_acc_final'].mean:.2%} ± {res['test_acc_final'].std:.2%}, "
                f"logpdf: {res['test_logpdf_final'].mean:.2f} ± {res['test_logpdf_final'].std:.2f}, "
                f"({param_count:,d} -> {subspace_param_count:,d}) "
                f"({duration:.1f}s)"
            )

    # * plot logpdf learning curve
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))  # 13 tasks total
    axs = axs.flatten()

    for i, task in enumerate(tasks):
        ax = axs[i]
        ax.set_ylim(-0.73, 0)
        ax.axhline(y=-0.69, linestyle=":", linewidth=1, color="red")  # ln(0.5) = -0.69
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
    # plt.show()

    # * plot acc learning curve
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))  # 13 tasks total
    axs = axs.flatten()
    for i, task in enumerate(tasks):
        ax = axs[i]
        ax.set_ylim(0.48, 1)
        ax.axhline(y=0.5, linestyle=":", linewidth=1, color="red")
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
    # plt.show()


if __name__ == "__main__":
    main()
