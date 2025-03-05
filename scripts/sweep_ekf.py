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

from bnn_pref.alg.bandit_env import BanditEnvironment
from bnn_pref.alg.ekf_subspace import SubspaceNeuralBandit
from bnn_pref.alg.train_utils import bandit_pipeline, summarize_results
from bnn_pref.data import BradleyTerry, QueryWithResponse, generate_pref_data
from bnn_pref.utils.metrics import compute_accuracy_nn, compute_pref_ranking_acc
from bnn_pref.utils.plotting import plot_logpdf, plot_reward_heatmap
from bnn_pref.utils.test_functions import test_functions_dict
from bnn_pref.utils.utils import get_gaussian_vector

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


def run_ekf(key, cfg, n_feats=None):
    data_kw = cfg["data"]
    ekf_kw = cfg["ekf"]
    dist = BradleyTerry()
    data_kw["n_feats"] = n_feats if n_feats is not None else data_kw["n_feats"]
    n_feats = data_kw["n_feats"]
    n_queries = data_kw["n_queries"]
    n_demos = data_kw["n_demos"]

    # * generate true params + preference data
    true_reward_fn = test_functions_dict[cfg["f"]]
    key, key1, key2, key3 = jr.split(key, 4)
    true_param_D = get_gaussian_vector(key1, dim=n_feats, normalize=True)
    demos_ND, returns_N, pref_data = generate_pref_data(
        key2, reward_fn=true_reward_fn, params_D=true_param_D, **data_kw
    )
    features_Q2D, response_Q1 = pref_data.queries_Q2D, pref_data.responses_Q1
    train_idxes, test_idxes = jnp.split(
        jr.permutation(key3, jnp.arange(n_queries)),
        [int(n_queries * 0.8)],
    )
    train_data = QueryWithResponse(features_Q2D[train_idxes], response_Q1[train_idxes])
    test_data = QueryWithResponse(features_Q2D[test_idxes], response_Q1[test_idxes])

    # * build + run bandit alg
    key, key1, key2 = jr.split(key, 3)
    env = BanditEnvironment(
        key1,
        X=train_data.queries_Q2D,
        Y=jax.nn.one_hot(train_data.responses_Q1.squeeze(), num_classes=2),
    )

    rewards_info, bel, bandit = bandit_pipeline(
        key2,
        SubspaceNeuralBandit,
        env,
        warmup_obs=ekf_kw["warm_obs"],
        n_trials=ekf_kw["n_trials"],
        bandit_kw=ekf_kw,
    )
    warmup_rewards, rewards_trace, opt_rewards = rewards_info
    # rtotal, rstd = summarize_results(warmup_rewards, rewards_trace)

    # todo n_trials beliefs..
    key, key1 = jr.split(key, 2)
    pref_predictor = jax.vmap(partial(bandit.apply_model, bel.mean[0]))
    reward_predictor = jax.vmap(partial(bandit.predict_reward, bel.mean[0]))
    train_acc = compute_accuracy_nn(pref_predictor, train_data)
    test_acc = compute_accuracy_nn(pref_predictor, test_data)
    pref_acc = compute_pref_ranking_acc(reward_predictor, test_data)

    results = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "pref_acc": pref_acc,
    }

    metadata = {
        "full_param_count": bandit.full_params_count,
        "subspace_param_count": bandit.subspace_params_count,
    }

    return results, metadata


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    seed = int(datetime.now().timestamp()) if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed=seed)
    n_seeds = 3

    # n_feats_list = [3, 10, 30]
    n_feats_list = [3, 10, 15, 30, 40, 50, 80, 100, 150, 300, 500, 1000, 2000]
    stats = []
    for n_feats in n_feats_list:
        # Run multiple seeds
        key, *subkeys = jr.split(key, 1 + n_seeds)  # m = 1 + n_seeds
        results_m, metadata_m = jax.vmap(run_ekf, in_axes=(0, None, None))(
            jnp.array(subkeys), cfg, n_feats
        )

        # Compute statistics
        stats.append(
            {
                "n_feats": n_feats,
                "accs_mean": results_m["test_acc"].mean(),
                "accs_std": results_m["test_acc"].std(),
            }
        )
        print(
            f"{n_feats=}, acc = {results_m['test_acc'].mean():.2%} ± {results_m['test_acc'].std():.1%}, "
            f"Param count: {metadata_m['full_param_count'][0]} -> {metadata_m['subspace_param_count'][0]}"
        )

    fig, axs = plt.subplots(1, 1)
    axs.errorbar(
        [stat["n_feats"] for stat in stats],
        [stat["accs_mean"] for stat in stats],
        yerr=[stat["accs_std"] for stat in stats],
        label="Accuracy",
        marker="o",
        markersize=3,
    )
    axs.set_title("EKF Sweep")
    axs.legend()
    axs.set_xlabel("Num Dimensions")
    axs.set_ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    main()
