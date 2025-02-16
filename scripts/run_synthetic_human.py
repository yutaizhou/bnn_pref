from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

from bnn_pref.data import create_pref_data
from bnn_pref.mcmc import build_hmc, build_mh, plot_samples, run_mcmc


def logpdf(x: Tuple[float, float], beta: float = 1.0) -> float:
    raise NotImplementedError


if __name__ == "__main__":
    key = jr.key(0)
    n_demos = 200  # check RLHF
    n_feats = 1000
    n_queries = 200

    demos_ND = jr.normal(key, (n_demos, n_feats))
    true_reward_D = jr.normal(key, (n_feats,))

    returns_N = demos_ND @ true_reward_D
    returns_N = jnp.sort(returns_N)

    key, subkey = jr.split(key)
    queries_Q2, labels_Q1, num_mislabels = create_pref_data(
        subkey, returns_N, n_queries
    )
    init_samples = jnp.zeros_like(true_reward_D)

    mh_kwargs = {"sigma": {}}
    sigma = {"loc1": 0.0, "loc2": 0.0, "logvar": 1.0}
    step_size = 0.1
    sigma = jnp.ones(len(init_samples)) * step_size
    alg = build_mh(logpdf, sigma)

    sampler_kwargs = {
        "init_samples": init_samples,
        "n_samples": 21000,
        "burn_in": 1000,
        "thinning": 2,
    }
    samples = run_mcmc(
        key=key,
        alg=alg,
        **sampler_kwargs,
    )
