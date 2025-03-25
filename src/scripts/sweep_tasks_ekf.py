import os

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

from bnn_pref.alg.ekf_trainer import bandit_pipeline
from bnn_pref.data import dataset_creators
from bnn_pref.data.ekf_env import EKFEnvironment
from bnn_pref.utils.metrics import (
    compute_accuracy_nn,
    compute_accuracy_nn_bel,
    compute_logpdf_nn,
)
from bnn_pref.utils.utils import get_random_seed

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


def run_ekf(key, cfg, data_dict):
    ekf_kw = cfg["ekf"]
    task_kw = cfg["task"]

    train_prefs, test_prefs = data_dict["train_prefs"], data_dict["test_prefs"]

    # * build + run bandit alg
    key, key1, key2 = jr.split(key, 3)
    env = EKFEnvironment(
        key1,
        X=train_prefs.queries_Q2TD,
        Y=jax.nn.one_hot(train_prefs.responses_Q1.squeeze(), num_classes=2),
    )

    key, key_bma = jr.split(key)

    rewards_info, bel_trace, bandit = bandit_pipeline(key2, env, ekf_kw)
    bel0 = jax.tree.map(lambda x: x[0], bel_trace)  # init belief, assume zero vec
    bel = jax.tree.map(lambda x: x[-1], bel_trace)  # final belief
    pref_predictor = jax.vmap(partial(bandit.sub2full_predict_logits, bel.mean))
    init_logit_predictor = jax.vmap(partial(bandit.sub2full_predict_logits, bel0.mean))

    train_acc_warm = compute_accuracy_nn(init_logit_predictor, train_prefs)
    train_acc = compute_accuracy_nn(pref_predictor, train_prefs)
    train_logpdf_warm = compute_logpdf_nn(init_logit_predictor, train_prefs)
    train_logpdf = compute_logpdf_nn(pref_predictor, train_prefs)
    test_acc_warm = compute_accuracy_nn(init_logit_predictor, test_prefs)
    test_acc = compute_accuracy_nn(pref_predictor, test_prefs)
    test_acc_bma = compute_accuracy_nn_bel(
        key_bma, bandit.sub2full_predict_logits, bel, test_prefs
    )
    test_logpdf_warm = compute_logpdf_nn(init_logit_predictor, test_prefs)
    test_logpdf = compute_logpdf_nn(pref_predictor, test_prefs)

    results = {
        "train_acc_warm": train_acc_warm,
        "train_acc": train_acc,
        "train_logpdf_warm": train_logpdf_warm,
        "train_logpdf": train_logpdf,
        "test_acc_warm": test_acc_warm,
        "test_acc": test_acc,
        "test_acc_bma": test_acc_bma,
        "test_logpdf_warm": test_logpdf_warm,
        "test_logpdf": test_logpdf,
    }
    metadata = {
        "n_train_queries": train_prefs.queries_Q2TD.shape[0],
        "n_test_queries": test_prefs.queries_Q2TD.shape[0],
        "full_param_count": bandit.full_params_count,
        "subspace_param_count": bandit.subspace_params_count,
    }
    return results, metadata


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    seed = get_random_seed() if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed)

    tasks = [
        "ogbench",
        "lunar",
        "reacher",
        "cheetah",
        "acrobot",
        "ball",
        "cartpoleSwing",
        "cheetahDMC",
        "hopperHop",
        "lunar",
        "pendulum",
        "reacherEasy",
        "reacherHard",
        "walkerWalk",
    ]
    stats = []
    for task in tasks:
        new_cfg = hydra.compose("config", overrides=[f"task={task}"])
        cfg["task"] = new_cfg["task"]

        key_data, *key_seeds = jr.split(key, 1 + cfg["seeds"])  # m = 1 + n_seeds
        data_dict = dataset_creators[cfg["task"]["ds_type"]](key_data, cfg)

        start_time = datetime.now()
        vmap_run_ekf = jax.vmap(run_ekf, in_axes=(0, None, None))
        results_m, metadata_m = vmap_run_ekf(jnp.array(key_seeds), cfg, data_dict)
        duration = (datetime.now() - start_time).total_seconds()

        results = {
            "task": task,
            "train_acc_warm": results_m["train_acc_warm"].mean().item(),
            "train_acc_warm_std": results_m["train_acc_warm"].std().item(),
            "train_acc": results_m["train_acc"].mean().item(),
            "train_acc_std": results_m["train_acc"].std().item(),
            "test_acc_warm": results_m["test_acc_warm"].mean().item(),
            "test_acc_warm_std": results_m["test_acc_warm"].std().item(),
            "test_acc": results_m["test_acc"].mean().item(),
            "test_acc_std": results_m["test_acc"].std().item(),
            "test_acc_bma": results_m["test_acc_bma"].mean().item(),
            "test_acc_bma_std": results_m["test_acc_bma"].std().item(),
            "train_logpdf_warm": results_m["train_logpdf_warm"].mean().item(),
            "train_logpdf_warm_std": results_m["train_logpdf_warm"].std().item(),
            "train_logpdf": results_m["train_logpdf"].mean().item(),
            "train_logpdf_std": results_m["train_logpdf"].std().item(),
            "test_logpdf_warm": results_m["test_logpdf_warm"].mean().item(),
            "test_logpdf_warm_std": results_m["test_logpdf_warm"].std().item(),
            "test_logpdf": results_m["test_logpdf"].mean().item(),
            "test_logpdf_std": results_m["test_logpdf"].std().item(),
        }

        stats.append(results)

        print(
            f"{task}: \n"
            f"  N train: {metadata_m['n_train_queries'][0]}, N test: {metadata_m['n_test_queries'][0]}\n"
            f"  Param count: {metadata_m['full_param_count'][0]} -> {metadata_m['subspace_param_count'][0]}\n"
            f"  Train acc:   {results['train_acc_warm']:.2%} ± {results['train_acc_warm_std']:.2%} -> {results['train_acc']:.2%} ± {results['train_acc_std']:.2%}\n"
            f"  Test acc:    {results['test_acc_warm']:.2%} ± {results['test_acc_warm_std']:.2%} -> {results['test_acc']:.2%} ± {results['test_acc_std']:.2%}\n"
            f"  Test acc BMA: {results['test_acc_bma']:.2%} ± {results['test_acc_bma_std']:.2%}\n"
            f"  Train logpdf: {results['train_logpdf_warm']:.2f} ± {results['train_logpdf_warm_std']:.2f} -> {results['train_logpdf']:.2f} ± {results['train_logpdf_std']:.2f}\n"
            f"  Test logpdf:  {results['test_logpdf_warm']:.2f} ± {results['test_logpdf_warm_std']:.2f} -> {results['test_logpdf']:.2f} ± {results['test_logpdf_std']:.2f}\n"
            f"  Time: {duration:.1f} seconds\n"
        )


if __name__ == "__main__":
    main()
