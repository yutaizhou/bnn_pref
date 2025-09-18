import itertools as it
import logging
import os
from datetime import datetime
from functools import partial

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
logging.getLogger("absl").setLevel(logging.WARNING)

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from hydra.core.hydra_config import HydraConfig

from bnn_pref.alg.trainer import run_alg
from bnn_pref.data import dataset_creators
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.data.pref_utils import modify_queries
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import MeanStd
from bnn_pref.utils.print_utils import get_param_count_msg
from bnn_pref.utils.utils import get_random_seed, nested_defaultdict, slurm_auto_scancel

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="configPref", config_path="../cfg")
def main(cfg):
    """
    Run all pref learning algs, both random and active, for one tasks.
    Meant to be used with slurm.
    """
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)

    stats = nested_defaultdict()
    algs = ["ekf", "sgd", "do"]
    is_als = [False, True]

    task_cfg = cfg["task"]
    data_cfg = cfg["data"]
    ekf_cfg = cfg["ekf"]
    sgd_cfg = cfg["sgd"]
    do_cfg = cfg["do"]

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
        f"  prior / dynamics / obs noise: {ekf_cfg['prior_noise']} / {ekf_cfg['dynamics_noise']} / {ekf_cfg['obs_noise']}\n"
        f"  init: bs={ekf_cfg['bs']}, niters={ekf_cfg['niters_init']}[{ekf_cfg['warm_burns']}::{ekf_cfg['thinning']}] ({n_eff_iterates} eff), sub_dim={ekf_cfg['sub_dim']}, rnd_proj={ekf_cfg['rnd_proj']}\n"
        f"Ensemble:\n"
        f"  M={sgd_cfg['M']}, use_vmap={sgd_cfg['use_vmap']}\n"
        f"  init: bs={sgd_cfg['bs']}, niters={sgd_cfg['niters_init']}\n"
        f"  update: bs={sgd_cfg['bs']}, niters={sgd_cfg['niters_update']}\n"
        f"Dropout:\n"
        f"  M={do_cfg['M']}, use_vmap={do_cfg['use_vmap']}\n"
        f"  init: bs={do_cfg['bs']}, niters={do_cfg['niters_init']}\n"
        f"  update: bs={do_cfg['bs']}, niters={do_cfg['niters_update']}\n"
    )

    ckpter = ocp.PyTreeCheckpointer()
    total_duration = datetime.now()
    task_choice = HydraConfig.get()["runtime"]["choices"]["task"]  # no hyphen

    # * update cfg for a specific task
    # new_cfg = hydra.compose("config", overrides=[f"task={task}"])
    # cfg["task"].update(new_cfg["task"])

    # create dataset
    key, key_data = jr.split(key, 2)
    data_dict = dataset_creators[task_cfg["ds_type"]](key_data, cfg)

    # create env
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
    # get hydra choice override name
    print(
        f"{task_choice:13} ({T=}, {D=}): train/test nt=({nt_train}/{nt_test}), nq=({nq_train}/{nq_test}), {train_prefs.n_mislabels} mislabels ({mislabel_ratio:.1%})"
    )

    env = PreferenceEnv(
        items=train_trajs["observations"],
        X=train_prefs.queries_Q2,
        Y=jax.nn.one_hot(train_prefs.responses_Q1.squeeze(), num_classes=2),
    )

    # * run algorithm
    key, *key_seeds = jr.split(key, 1 + cfg["seeds"])
    seeds = jnp.array(key_seeds)
    for alg, is_al in it.product(algs, is_als):
        cfg[alg]["active"] = is_al

        run_fn = partial(run_alg, alg=alg, cfg=cfg, data_dict=data_dict, env=env)

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
            "task": task_choice,
            "task_name": task_cfg["name"],
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
        best_seed = jnp.argmax(res_m["test_logpdf"][:, -1])
        best_model = jax.tree.map(lambda x: x[best_seed], res_m["model"])

        # * save best model
        save_args = orbax_utils.save_args_from_target(best_model)
        ckpt_name = f"{task_cfg['name']}_{alg}_al={is_al}"
        ckpter.save(
            f"{cfg.paths.ckpts_dir}/{ckpt_name}",
            best_model,
            save_args=save_args,
        )

        stats[task_choice][alg][is_al] = res
        nonfinites = ~jnp.isfinite(res_m["test_logpdf"])  # (n_seeds, 1 + nq_update)

        print(
            f"  {alg} active={str(is_al):5}, "
            f"acc: {res['test_acc_final']['mean']:.2%} ± {res['test_acc_final']['std']:.2%}, "
            f"logpdf: {res['test_logpdf_final']['mean']:.2f} ± {res['test_logpdf_final']['std']:.2f}; "
            f"ece: {res['test_ece_final']['mean']:.2f} ± {res['test_ece_final']['std']:.2f}, "
            f"brier: {res['test_brier_final']['mean']:.2f} ± {res['test_brier_final']['std']:.2f}, "
            f"coverage: {res['test_coverage_final']['mean']:.2%} ± {res['test_coverage_final']['std']:.2%}, "
            f"sharpness: {res['test_sharpness_final']['mean']:.2f} ± {res['test_sharpness_final']['std']:.2f}, "
            f"{get_param_count_msg(cfg, alg, res_m)}, "
            f"({res['duration']:.1f}s)"
        )
        if nonfinites.any():
            print(f"nonfinites: {nonfinites.sum(1)}")

    total_duration = (datetime.now() - total_duration).total_seconds()
    print(f"Total duration: {total_duration:.1f}s")
    jnp.savez(f"{cfg.paths.output_dir}/stats.npz", **stats)
    slurm_auto_scancel()  # prevent completed jobs from hanging on slurm


if __name__ == "__main__":
    main()
