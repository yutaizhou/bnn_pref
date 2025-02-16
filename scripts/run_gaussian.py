from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

from bnn_pref.mcmc import build_hmc, build_mh, plot_samples, run_mcmc


def sample_bimodal_gaussian(key, n_samples, loc1, var1, loc2, var2, weight, dim=2):
    key1, key2, key3 = jr.split(key, 3)

    # Sample from uniform to decide which mode to sample from
    mode_selector = jr.uniform(key1, (n_samples,)) < weight

    # Sample from both modes
    samples1 = loc1 + jr.normal(key2, (n_samples, dim)) * jnp.sqrt(var1)
    samples2 = loc2 + jr.normal(key3, (n_samples, dim)) * jnp.sqrt(var2)

    # Select samples based on the mode selector
    samples = jnp.where(mode_selector[:, None], samples1, samples2)

    return samples


def generate_learned_samples(key, loc1, loc2, logvar, dim=5):
    weight = 0.5
    num_samples = 1
    var = jnp.exp(logvar)
    key1, key2, key3 = jr.split(key, 3)

    # Sample from uniform to decide which mode to sample from
    mode_selector = jr.uniform(key1, (num_samples,)) < weight

    # Sample from both modes
    samples1 = loc1 + jr.normal(key2, (num_samples, dim)) * jnp.sqrt(var)
    samples2 = loc2 + jr.normal(key3, (num_samples, dim)) * jnp.sqrt(var)

    # Select samples based on the mode selector
    samples = jnp.where(mode_selector[:, None], samples1, samples2)

    return samples


def bimodal_gaussian_logpdf(loc1, loc2, logvar, dim=5, x=None):
    var = jnp.exp(logvar)
    mode1 = jsp.stats.multivariate_normal.logpdf(x, mean=loc1, cov=var)
    mode2 = jsp.stats.multivariate_normal.logpdf(x, mean=loc2, cov=var)
    return jnp.logaddexp(mode1, mode2).sum()


def logpdf(params, data):
    return bimodal_gaussian_logpdf(x=data, **params)


if __name__ == "__main__":
    key = jr.key(0)

    dim = 1
    data_kwargs = {
        "n_samples": 1000,
        "loc1": 2,
        "var1": 2,
        "loc2": -2,
        "var2": 2,
        "weight": 0.5,
        "dim": dim,
    }
    key, key_data = jr.split(key)
    data = sample_bimodal_gaussian(key_data, **data_kwargs)
    logpdf = partial(logpdf, data=data)

    init_samples = {"loc1": 0.0, "loc2": 0.0, "logvar": 1.0}

    # mh_kwargs = {"sigma": {}}
    # sigma = {"loc1": 0.0, "loc2": 0.0, "logvar": 1.0}
    step_size = 0.1
    sigma = jnp.ones(len(init_samples)) * step_size
    alg = build_mh(logpdf, sigma)

    sampler_kwargs = {
        "init_samples": init_samples,
        "n_samples": 21000,
        "burn_in": 1000,
        "thinning": 2,
    }

    key, key_mcmc = jr.split(key)
    samples = run_mcmc(
        key=key_mcmc,
        alg=alg,
        **sampler_kwargs,
    )

    learned_samples = jax.vmap(generate_learned_samples)(
        jr.split(key, len(samples["loc1"])),
        samples["loc1"],
        samples["loc2"],
        samples["logvar"],
    ).squeeze()

    # print(samples.shape)
    # likelihoods = jnp.exp(jax.vmap(logpdf)(samples))
    # print(likelihoods.mean())

    # plt.hist(likelihoods, bins=50, density=True)
    # plt.title("Distribution of Log-Likelihood Values")
    # plt.xlabel("Log-Likelihood")
    # plt.ylabel("Density")
    # plt.show()

    plot_samples(learned_samples)
