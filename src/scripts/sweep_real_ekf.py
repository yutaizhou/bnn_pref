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
from bnn_pref.utils.metrics import compute_accuracy_nn, compute_logpdf_nn
from bnn_pref.utils.utils import get_random_seed

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


def run_ekf(key, cfg):
    data_kw = cfg["data"]
    ekf_kw = cfg["ekf"]
    task_kw = cfg["task"]

    # * generate true params + preference data
    output = dataset_creators[task_kw["ds_type"]](key, cfg)
    train_data, test_data = output["train_prefs"], output["test_prefs"]

    # * build + run bandit alg
    key, key1, key2 = jr.split(key, 3)
    env = EKFEnvironment(
        key1,
        X=train_data.queries_Q2TD,
        Y=jax.nn.one_hot(train_data.responses_Q1.squeeze(), num_classes=2),
    )

    rewards_info, bel_trace, bandit = bandit_pipeline(key2, env, ekf_kw)
    bel = jax.tree_util.tree_map(lambda x: x[-1], bel_trace)  # final belief

    pref_predictor = jax.vmap(partial(bandit.sub2full_predict_logits, bel.mean))
    test_acc = compute_accuracy_nn(pref_predictor, test_data)
    test_logpdf = compute_logpdf_nn(pref_predictor, test_data)

    results = {
        "test_acc": test_acc,
        "test_logpdf": test_logpdf,
    }
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
        new_cfg = hydra.compose(
            "config",
            overrides=[f"task={task}"],
        )
        cfg["task"] = new_cfg["task"]
        # Run multiple seeds
        key, *subkeys = jr.split(key, 1 + cfg["seeds"])  # m = 1 + n_seeds

        start_time = datetime.now()
        vmap_run_ekf = jax.vmap(run_ekf, in_axes=(0, None))
        results_m, metadata_m = vmap_run_ekf(jnp.array(subkeys), cfg)
        duration = (datetime.now() - start_time).total_seconds()

        # Compute statistics
        results = {
            "accs_mean": results_m["test_acc"].mean(),
            "accs_std": results_m["test_acc"].std(),
            "logpdf_mean": results_m["test_logpdf"].mean(),
            "logpdf_std": results_m["test_logpdf"].std(),
        }
        stats.append(results)

        print(
            f"{task:10}: "
            f"acc = {results['accs_mean']:.2%} ± {results['accs_std']:.1%}, "
            f"logpdf = {results['logpdf_mean']:.2f} ± {results['logpdf_std']:.1f}, "
            f"Param count: {metadata_m['full_param_count'][0]} -> {metadata_m['subspace_param_count'][0]}, "
            f"Time: {duration:.1f} seconds"
        )


if __name__ == "__main__":
    main()
