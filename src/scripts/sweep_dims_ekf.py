import logging
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from bnn_pref.alg.ekf_trainer import bandit_pipeline
from bnn_pref.data import make_synthetic_data
from bnn_pref.data.ekf_env import EKFEnvironment
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import (
    MeanStd,
    compute_acc_nn,
    compute_logpdf_nn,
    compute_pref_ranking_acc,
)
from bnn_pref.utils.utils import get_random_seed

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


def run_ekf(key, cfg):
    data_kw = cfg["data"]
    task_kw = cfg["task"]
    ekf_kw = cfg["ekf"]

    # * generate true params + preference data
    output = make_synthetic_data(key, cfg)
    train_data, test_data = output["train_prefs"], output["test_prefs"]

    # * build + run bandit alg
    key, key1, key2 = jr.split(key, 3)
    env = EKFEnvironment(
        key1,
        X=train_data.queries_Q2TD,
        Y=jax.nn.one_hot(train_data.responses_Q1.squeeze(), num_classes=2),
    )

    bel_trace, bandit = bandit_pipeline(key2, env, ekf_kw)
    bel = jax.tree_util.tree_map(lambda x: x[-1], bel_trace)

    key, key1 = jr.split(key, 2)
    pref_predictor = jax.vmap(partial(bandit.sub2full_predict_logits, bel.mean))
    reward_predictor = jax.vmap(partial(bandit.sub2full_predict_return, bel.mean))
    train_acc = compute_acc_nn(pref_predictor, train_data)
    test_acc = compute_acc_nn(pref_predictor, test_data)
    test_logpdf = compute_logpdf_nn(pref_predictor, test_data)
    pref_acc = compute_pref_ranking_acc(reward_predictor, test_data)

    results = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "pref_acc": pref_acc,
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

    # n_feats_list = [3, 10, 30]
    n_feats_list = [3, 10, 30, 50, 100, 150, 300, 500, 1000, 5000, 10000]
    # n_feats_list = [16000, 32000, 64000, 128000, 256000]

    stats = []
    for n_feats in n_feats_list:
        new_cfg = hydra.compose(
            "config",
            overrides=["task=synthetic", f"task.n_feats={n_feats}"],
        )
        # Run multiple seeds
        key, *subkeys = jr.split(key, 1 + cfg["seeds"])  # m = 1 + n_seeds

        start_time = datetime.now()
        vmap_run_ekf = jax.vmap(run_ekf, in_axes=(0, None))
        res_m, metadata_m = vmap_run_ekf(jnp.array(subkeys), new_cfg)
        duration = (datetime.now() - start_time).total_seconds()

        # Compute statistics
        res = {
            "n_feats": n_feats,
            "test_acc": MeanStd(res_m["test_acc"]),
            "test_logpdf": MeanStd(res_m["test_logpdf"]),
        }
        stats.append(res)

        print(
            f"n_feats={n_feats:4}, acc = {res['test_acc'].mean:.2%} ± {res['test_acc'].std:.1%}, "
            f"logpdf = {res['test_logpdf'].mean:.2f} ± {res['test_logpdf'].std:.1f}, "
            f"Param count: {metadata_m['full_param_count'][0]} -> {metadata_m['subspace_param_count'][0]}, "
            f"Time: {duration:.1f} seconds"
        )

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot accuracy on left y-axis
    color1 = "tab:blue"
    ax1.errorbar(
        [stat["n_feats"] for stat in stats],
        [stat["test_acc"].mean for stat in stats],
        yerr=[stat["test_acc"].std for stat in stats],
        label="Accuracy",
        marker="o",
        markersize=3,
        color=color1,
    )
    ax1.set_xlabel("Num Dimensions")
    ax1.set_ylabel("Accuracy", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1)

    # Create second y-axis and plot logpdf
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.errorbar(
        [stat["n_feats"] for stat in stats],
        [stat["test_logpdf"].mean for stat in stats],
        yerr=[stat["test_logpdf"].std for stat in stats],
        label="Logpdf",
        marker="o",
        markersize=3,
        color=color2,
    )
    ax2.set_ylabel("Logpdf", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("EKF Sweep")
    fp = Path(cfg.paths.output_dir) / "ekf_sweep.png"
    plt.savefig(fp)
    plt.show()


if __name__ == "__main__":
    main()
