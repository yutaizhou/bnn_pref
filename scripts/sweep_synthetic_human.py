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
from bnn_pref.utils.utils import get_gaussian_vector, tile_first_dim


# @partial(jax.jit, static_argnames=("cfg"))
def run_experiment(cfg, key, n_feats=None):
    # check RLHF paper
    data_kw = cfg["data"]
    mcmc_kw = cfg["mcmc"]
    dist = BradleyTerry()
    data_kw["n_feats"] = n_feats if n_feats is not None else data_kw["n_feats"]
    n_feats = data_kw["n_feats"]

    # * generate true params + preference data
    key, key1, key2 = jr.split(key, 3)
    true_reward_D = get_gaussian_vector(key1, dim=n_feats, normalize=True)
    features_Q2D, response_Q1 = generate_pref_data(key2, true_reward_D, **data_kw)
    data = QueryWithResponse(features_Q2D, response_Q1)

    # * build + run sampler
    key, key1, key2 = jr.split(key, 3)
    init_sample = jnp.zeros_like(true_reward_D)
    alg = build_mh(partial(dist.potential, data=data), sigma=mcmc_kw["sigma"])
    samples_SD, states, infos = run_mcmc(
        key1,
        alg=alg,
        init_sample=init_sample,
        **{k: mcmc_kw[k] for k in ["n_samples", "burn_in", "thinning", "normalize"]},
    )

    metadata = {
        "features": features_Q2D,
        "response": response_Q1,
        "true_reward": true_reward_D,
    }

    return samples_SD, metadata


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def run_dimensinality_exp(cfg):
    seed = int(datetime.now().timestamp()) if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed=seed)
    n_seeds = 3

    # n_feats_list = [2, 3]
    # n_feats_list = [2, 3, 5, 8, 15, 30, 50, 100, 200, 500]
    n_feats_list = [3, 5, 10, 15, 30, 40, 50, 80, 100, 150, 300, 400]
    stats = []
    for n_feats in n_feats_list:
        key, *subkeys = jr.split(key, 1 + n_seeds)  # m = 1 + n_seeds
        # samples_SD, metadata = run_experiment(cfg, subkey, n_feats=n_feats)
        samples_mSD, metadata = jax.vmap(run_experiment, in_axes=(None, 0, None))(
            cfg, jnp.array(subkeys), n_feats
        )
        features_mQ2D = metadata["features"]
        response_mQ1 = metadata["response"]
        true_reward_mD = metadata["true_reward"]

        accs = jax.vmap(compute_accuracy2_mcmc, in_axes=(0, 0, 0))(
            samples_mSD, features_mQ2D, response_mQ1
        )
        aligns = jax.vmap(alignment_metric, in_axes=(0, 0))(true_reward_mD, samples_mSD)
        stats.append(
            {
                "n_feats": n_feats,
                "accs_mean": accs.mean(),
                "aligns_mean": aligns.mean(),
                "accs_std": accs.std(),
                "aligns_std": aligns.std(),
            }
        )
        print(
            f"{n_feats=}, acc = {accs.mean():.3f} ± {accs.std():.1f}, align = {aligns.mean():.3f} ± {aligns.std():.3f}"
        )

    fig, axs = plt.subplots(1, 1)
    axs.errorbar(
        [stat["n_feats"] for stat in stats],
        [stat["accs_mean"] for stat in stats],
        yerr=[stat["accs_std"] for stat in stats],
        label="Accuracy",
    )
    axs.errorbar(
        [stat["n_feats"] for stat in stats],
        [stat["aligns_mean"] for stat in stats],
        yerr=[stat["aligns_std"] for stat in stats],
        label="Alignment",
    )
    axs.legend()
    axs.set_xlabel("Num Dimensions")
    axs.set_ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    run_dimensinality_exp()
