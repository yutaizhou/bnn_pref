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
from bnn_pref.utils.print_utils import get_param_count_msg, print_run_cfg
from bnn_pref.utils.utils import get_random_seed, nested_defaultdict, slurm_auto_scancel

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="configScaling", config_path="../cfg")
def main(cfg):
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)
    algs = ["ekf", "sgd", "do", "laplace", "llmcmc"]

    task = cfg["task"]["name"]
    task_cfg = cfg["task"]
    data_cfg = cfg["data"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]

    print_run_cfg(seed, cfg)

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
    key, key_seed = jr.split(key)
    stats = nested_defaultdict()
    for alg in algs:
        res_m = run_alg(key_seed, alg_name=alg, cfg=cfg, data_dict=data_dict, env=env)

        # (1 + nq_update)
        res = {
            "task": task,
            "nq_train": nq_train,
            "nq_test": nq_test,
            # * duration: ( )
            "train_duration": res_m["train_duration"],
            "eval_duration": res_m["eval_duration"],
            "total_duration": res_m["total_duration"],
            # * increment eval results: (1 + nq_update,)
            # logpdf
            "test_logpdf_all": res_m["test_logpdf"],
            "test_logpdf_final": res_m["test_logpdf"][-1],
            # acc
            "test_acc_all": res_m["test_acc"],
            "test_acc_final": res_m["test_acc"][-1],
        }

        stats[task][alg] = res

        test_logpdf_all = res_m["test_logpdf"]
        nans = ~jnp.isfinite(test_logpdf_all)

        print(
            f"  {alg}, "
            f"acc: {res['test_acc_final']:.2%}, "
            f"logpdf: {res['test_logpdf_final']:.2f}; "
            f"{get_param_count_msg(cfg, alg, res_m)}, "
            f"({res['total_duration']:.1f}s, {res['train_duration']:.1f}s, {res['eval_duration']:.1f}s)"
        )
        if nans.any():
            print(f"nans: {nans.sum(1)}")
    total_duration = (datetime.now() - total_duration).total_seconds()
    print(f"Total duration: {total_duration:.1f}s")
    jnp.savez(f"{cfg.paths.output_dir}/stats.npz", **stats)


if __name__ == "__main__":
    main()
    slurm_auto_scancel()
