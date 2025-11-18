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
from bnn_pref.utils.print_utils import get_param_count_msg, get_run_cfg_msg
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

    algs = ["ekf", "sgd", "do", "laplace", "llmcmc"]
    # algs = ["ekf", "sgd"]

    is_als = [False, True]
    # is_als = [True]

    task_cfg = cfg["task"]
    data_cfg = cfg["data"]

    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    nq_init = data_cfg["nq_init"]
    get_run_cfg_msg(seed, cfg)

    ckpter = ocp.PyTreeCheckpointer()
    total_duration = datetime.now()
    task_choice = HydraConfig.get()["runtime"]["choices"]["task"]  # no hyphen

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
    stats = nested_defaultdict()
    for alg, is_al in it.product(algs, is_als):
        cfg[alg]["active"] = is_al

        run_fn = partial(run_alg, alg_name=alg, cfg=cfg, data_dict=data_dict, env=env)
        res_m = []
        bel_m = []
        for seed in seeds:
            res = jax.block_until_ready(run_fn(seed))
            bel_final = res.pop("final_belief")  # tree stack can't handle ts.apply_fn
            res_m.append(res)
            bel_m.append(bel_final)
        res_m = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *res_m)

        res = {
            "task": task_choice,
            "task_name": task_cfg["name"],
            "is_active": is_al,
            "nq_train": nq_train,
            "nq_test": nq_test,
            "duration": res_m["train_duration"].mean(),
            "eval_duration": res_m["eval_duration"].mean(),
            # * metrics below are of shape (n_seeds, 1 + nq_update)
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
        best_belief = bel_m[best_seed]

        # * save best belief among seeds
        save_args = orbax_utils.save_args_from_target(best_belief)
        ckpt_name = f"{task_cfg['name']}_{alg}_al={is_al}"
        ckpter.save(
            f"{cfg.paths.ckpts_dir}/{ckpt_name}",
            best_belief,
            save_args=save_args,
        )

        stats[task_choice][alg][is_al] = res
        nonfinites = ~jnp.isfinite(res_m["test_logpdf"])  # (n_seeds, 1 + nq_update)

        print(
            f"\t{alg} active={str(is_al):5}, "
            f"logpdf: {res['test_logpdf_final']['mean']:.2f} ± {res['test_logpdf_final']['std']:.2f}; "
            # f"acc: {res['test_acc_final']['mean']:.2%} ± {res['test_acc_final']['std']:.2%}, "
            # f"ece: {res['test_ece_final']['mean']:.2f} ± {res['test_ece_final']['std']:.2f}, "
            # f"brier: {res['test_brier_final']['mean']:.2f} ± {res['test_brier_final']['std']:.2f}, "
            # f"coverage: {res['test_coverage_final']['mean']:.2%} ± {res['test_coverage_final']['std']:.2%}, "
            # f"sharpness: {res['test_sharpness_final']['mean']:.2f} ± {res['test_sharpness_final']['std']:.2f}, "
            f"{get_param_count_msg(cfg, alg, res_m)}, "
            f"({res['duration']:.1f}s / {res['eval_duration']:.1f}s)"
        )
        if nonfinites.any():
            print(f"nonfinites: {nonfinites.sum(1)}")

    total_duration = (datetime.now() - total_duration).total_seconds()
    fp = f"{cfg.paths.output_dir}/stats.npz"
    jnp.savez(fp, **stats)
    print(f"Total duration: {total_duration:.1f}s")
    print(f"Saved stats to {fp}")
    slurm_auto_scancel()  # prevent completed jobs from hanging on slurm


if __name__ == "__main__":
    main()
