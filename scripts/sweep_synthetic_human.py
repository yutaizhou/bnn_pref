import logging
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import arviz as az
import hydra
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpl
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np

from bnn_pref.alg.mcmc import build_hmc, build_mh, plot_samples, plot_trace, run_mcmc
from bnn_pref.data import BradleyTerry, QueryWithResponse, generate_pref_data
from bnn_pref.utils.metrics import alignment_metric, compute_accuracy2_mcmc
from bnn_pref.utils.test_functions import test_functions_dict
from bnn_pref.utils.utils import get_gaussian_vector


# @partial(jax.jit, static_argnames=("cfg"))
def run_experiment(cfg, key, n_feats=None):
    # check RLHF paper
    data_kw = cfg["data"]
    mcmc_kw = cfg["mcmc"]
    dist = BradleyTerry()
    data_kw["n_feats"] = n_feats if n_feats is not None else data_kw["n_feats"]
    n_feats = data_kw["n_feats"]
    n_queries = data_kw["n_queries"]

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

    # * build + run sampler
    key, key1, key2 = jr.split(key, 3)
    init_sample = jnp.zeros_like(true_param_D)
    alg = build_mh(
        partial(dist.potential, data=train_data, reward_fn=true_reward_fn),
        sigma=mcmc_kw["sigma"],
    )
    samples_SD, states, infos = run_mcmc(
        key1,
        alg=alg,
        init_sample=init_sample,
        **{k: mcmc_kw[k] for k in ["n_samples", "burn_in", "thinning", "normalize"]},
    )

    accs = compute_accuracy2_mcmc(samples_SD, train_data, true_reward_fn)
    aligns = alignment_metric(true_param_D, samples_SD)

    results = {
        "accs": accs,
        "aligns": aligns,
    }

    metadata = {
        "features": features_Q2D,
        "response": response_Q1,
        "true_reward": true_param_D,
    }

    return results, metadata


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def run_dimensinality_exp(cfg):
    seed = int(datetime.now().timestamp()) if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed=seed)
    n_seeds = 3

    # n_feats_list = [2, 3]
    # n_feats_list = [2, 3, 5, 8, 15, 30, 50, 100, 200, 500]
    n_feats_list = [3, 10, 15, 30, 40, 50, 80, 100, 150, 300, 500, 1000, 2000]
    stats = []
    for n_feats in n_feats_list:
        key, *subkeys = jr.split(key, 1 + n_seeds)  # m = 1 + n_seeds
        # samples_SD, metadata = run_experiment(cfg, subkey, n_feats=n_feats)
        results, metadata = jax.vmap(run_experiment, in_axes=(None, 0, None))(
            cfg, jnp.array(subkeys), n_feats
        )

        stats.append(
            {
                "n_feats": n_feats,
                "accs_mean": results["accs"].mean(),
                "accs_std": results["accs"].std(),
                "aligns_mean": results["aligns"].mean(),
                "aligns_std": results["aligns"].std(),
            }
        )
        print(
            f"{n_feats=}, acc = {results['accs'].mean():.2%} ± {results['accs'].std():.1%}, "
            f"align = {results['aligns'].mean():.2%} ± {results['aligns'].std():.1%}"
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
    axs.errorbar(
        [stat["n_feats"] for stat in stats],
        [stat["aligns_mean"] for stat in stats],
        yerr=[stat["aligns_std"] for stat in stats],
        label="Alignment",
        marker="o",
        markersize=3,
    )
    axs.set_title("MCMC Sweep")
    axs.legend()
    axs.set_xlabel("Num Dimensions")
    axs.set_ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    run_dimensinality_exp()
