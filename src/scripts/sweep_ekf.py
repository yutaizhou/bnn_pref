import logging
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from bnn_pref.alg.ekf_subspace import SubspaceNeuralBandit
from bnn_pref.alg.ekf_trainer import bandit_pipeline, summarize_results
from bnn_pref.data.ekf_env import EKFEnvironment
from bnn_pref.data.synthetic import make_synthetic_data
from bnn_pref.utils.metrics import (
    compute_accuracy_nn,
    compute_logpdf_nn,
    compute_pref_ranking_acc,
)
from bnn_pref.utils.utils import get_random_seed

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@partial(jax.jit, static_argnums=(1, 2))
def run_ekf(key, cfg, n_feats=None):
    data_kw = cfg["data"]
    ekf_kw = cfg["ekf"]
    data_kw["n_feats"] = n_feats if n_feats is not None else data_kw["n_feats"]

    # * generate true params + preference data
    output = make_synthetic_data(key, cfg)
    train_data, test_data = output["train_prefs"], output["test_prefs"]
    true_param_D, true_reward_fn = output["true_param"], output["true_reward_fn"]
    train_traj_obs = output["train_trajs"]["observations"]
    feature_bounds = (train_traj_obs.min(), train_traj_obs.max())

    # * build + run bandit alg
    key, key1, key2 = jr.split(key, 3)
    env = EKFEnvironment(
        key1,
        X=train_data.queries_Q2TD,
        Y=jax.nn.one_hot(train_data.responses_Q1.squeeze(), num_classes=2),
    )

    rewards_info, bel_trace, bandit = bandit_pipeline(
        key2,
        SubspaceNeuralBandit,
        env,
        warmup_obs=ekf_kw["warm_obs"],
        bandit_kw=ekf_kw,
    )
    bel = jax.tree_util.tree_map(lambda x: x[-1], bel_trace)
    warmup_rewards, rewards_trace, _ = rewards_info

    key, key1 = jr.split(key, 2)
    pref_predictor = jax.vmap(partial(bandit.apply_model, bel.mean))
    reward_predictor = jax.vmap(partial(bandit.predict_reward, bel.mean))
    train_acc = compute_accuracy_nn(pref_predictor, train_data)
    test_acc = compute_accuracy_nn(pref_predictor, test_data)
    test_logpdf = compute_logpdf_nn(pref_predictor, test_data)
    pref_acc = compute_pref_ranking_acc(reward_predictor, test_data)

    results = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "pref_acc": pref_acc,
        "test_logpdf": test_logpdf,
    }

    results = jax.tree.map(lambda x: jnp.float32(x), results)

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
    n_feats_list = [3, 10, 30, 50, 100, 150, 300, 500, 1000]
    # n_feats_list = [16000, 32000, 64000, 128000, 256000]

    stats = []
    for n_feats in n_feats_list:
        # Run multiple seeds
        key, *subkeys = jr.split(key, 1 + cfg["seeds"])  # m = 1 + n_seeds

        start_time = datetime.now()
        results_m, metadata_m = jax.vmap(run_ekf, in_axes=(0, None, None))(
            jnp.array(subkeys), cfg, n_feats
        )
        duration = (datetime.now() - start_time).total_seconds()

        # Compute statistics
        results = {
            "n_feats": n_feats,
            "accs_mean": results_m["test_acc"].mean(),
            "accs_std": results_m["test_acc"].std(),
            "logpdf_mean": results_m["test_logpdf"].mean(),
            "logpdf_std": results_m["test_logpdf"].std(),
        }
        stats.append(results)

        print(
            f"n_feats={n_feats:4}, acc = {results['accs_mean']:.2%} ± {results['accs_std']:.1%}, "
            f"logpdf = {results['logpdf_mean']:.2f} ± {results['logpdf_std']:.1f}, "
            f"Param count: {metadata_m['full_param_count'][0]} -> {metadata_m['subspace_param_count'][0]}, "
            f"Time: {duration:.1f} seconds"
        )

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot accuracy on left y-axis
    color1 = "tab:blue"
    ax1.errorbar(
        [stat["n_feats"] for stat in stats],
        [stat["accs_mean"] for stat in stats],
        yerr=[stat["accs_std"] for stat in stats],
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
        [stat["logpdf_mean"] for stat in stats],
        yerr=[stat["logpdf_std"] for stat in stats],
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
