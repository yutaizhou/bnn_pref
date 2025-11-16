import logging
import os
from datetime import datetime
from functools import partial

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr

from bnn_pref.alg.trainer import run_alg
from bnn_pref.data import dataset_creators
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import MeanStd
from bnn_pref.utils.print_utils import get_param_count_msg
from bnn_pref.utils.utils import get_random_seed, nested_defaultdict, slurm_auto_scancel

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="configEKFSubdim", config_path="../cfg")
def main(cfg):
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)
    algs = ["ekf"]

    task = cfg["task"]["name"]
    task_cfg = cfg["task"]
    data_cfg = cfg["data"]
    ekf_cfg = cfg["ekf"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    nq_init, nsteps = data_cfg["nq_init"], data_cfg["nsteps"]
    n_eff_iterates = (ekf_cfg["niters_init"] - ekf_cfg["warm_burns"]) // ekf_cfg[
        "thinning"
    ]

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
        f"  sub_dim={ekf_cfg['sub_dim']}, rnd_proj={ekf_cfg['rnd_proj']}\n"
        f"  prior / dynamics / obs noise: {ekf_cfg['prior_noise']} / {ekf_cfg['dynamics_noise']} / {ekf_cfg['obs_noise']}\n"
        f"  init: bs={ekf_cfg['bs']}, niters={ekf_cfg['niters_init']}[{ekf_cfg['warm_burns']}::{ekf_cfg['thinning']}] ({n_eff_iterates} eff), sub_dim={ekf_cfg['sub_dim']}, rnd_proj={ekf_cfg['rnd_proj']}\n"
    )
    print(jax.devices())

    total_duration = datetime.now()
    # * create dataset
    key, key_data = jr.split(key, 2)
    data_dict = dataset_creators[task_cfg["ds_type"]](key_data, cfg)

    # * create env
    train_trajs, test_trajs = data_dict["train_trajs"], data_dict["test_trajs"]
    train_prefs, test_prefs = data_dict["train_prefs"], data_dict["test_prefs"]

    nt_train, T, D = train_trajs["observations"].shape
    nt_test = test_trajs["observations"].shape[0]
    nq_train = train_prefs.queries_Q2.shape[0]
    nq_test = test_prefs.queries_Q2.shape[0]

    print(
        f"{task} ({T=}, {D=}): train/test nt=({nt_train}/{nt_test}), nq=({nq_train}/{nq_test})"
    )

    env = PreferenceEnv(
        items=train_trajs["observations"],
        X=train_prefs.queries_Q2,
        Y=jax.nn.one_hot(train_prefs.responses_Q1.squeeze(), num_classes=2),
    )

    # * run algorithm
    key, *key_seeds = jr.split(key, 1 + cfg["seeds"])
    seeds = jnp.array(key_seeds)
    stats = nested_defaultdict()
    for alg in algs:
        # start = datetime.now()
        # res_m = run_alg(key_seed, alg=alg, cfg=cfg, data_dict=data_dict, env=env)
        # duration = (datetime.now() - start).total_seconds()

        # run in vmap or lax version (parallel vs. sequential)
        run_fn = partial(run_alg, alg_name=alg, cfg=cfg, data_dict=data_dict, env=env)
        start = datetime.now()

        res_m = (
            jax.block_until_ready(jax.vmap(run_fn)(seeds))
            if cfg["seed_vmap"]
            else jax.block_until_ready(jax.lax.map(run_fn, seeds))
        )

        duration = (datetime.now() - start).total_seconds()
        # (1 + nq_update)
        res = {
            "task": task,
            "nq_train": nq_train,
            "nq_test": nq_test,
            "duration": duration,
            # * logpdf
            "test_logpdf_all": res_m["test_logpdf"],
            "test_logpdf_final": MeanStd(res_m["test_logpdf"][:, -1]).get_stats(),
            # * acc
            "test_acc_all": res_m["test_acc"],
            "test_acc_final": MeanStd(res_m["test_acc"][:, -1]).get_stats(),
            # * ece
            "test_ece_all": res_m["test_ece"],
            "test_ece_final": MeanStd(res_m["test_ece"][:, -1]).get_stats(),
            # * brier
            "test_brier_all": res_m["test_brier"],
            "test_brier_final": MeanStd(res_m["test_brier"][:, -1]).get_stats(),
            # * coverage
            "test_coverage_all": res_m["test_coverage"],
            "test_coverage_final": MeanStd(res_m["test_coverage"][:, -1]).get_stats(),
            # * sharpness
            "test_sharpness_all": res_m["test_sharpness"],
            "test_sharpness_final": MeanStd(res_m["test_sharpness"][:, -1]).get_stats(),
        }

        stats[task][alg] = res

        test_logpdf_all = res_m["test_logpdf"]
        nans = ~jnp.isfinite(test_logpdf_all)

        print(
            f"  {alg}, "
            f"logpdf: {res['test_logpdf_final']['mean']:.2f} ± {res['test_logpdf_final']['std']:.2f}; "
            f"{get_param_count_msg(cfg, alg, res_m)}, "
            f"({res['duration']:.1f}s)"
        )
        if nans.any():
            print(f"nans: {nans.sum(1)}")
    total_duration = (datetime.now() - total_duration).total_seconds()
    print(f"Total duration: {total_duration:.1f}s")
    jnp.savez(f"{cfg.paths.output_dir}/stats.npz", **stats)


if __name__ == "__main__":
    main()
    slurm_auto_scancel()
