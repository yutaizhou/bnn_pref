from dataclasses import dataclass
from functools import partial

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt

from bnn_pref.alg.mcmc import build_hmc, build_mh, plot_samples, run_mcmc


@dataclass
class BimodalGaussian:
    var: float = None

    @staticmethod
    def sample(key, n_samples, mean1, mean2, var, weight=0.5):
        key1, key2, key3 = jr.split(key, 3)

        # Sample from uniform to decide which mode to sample from
        mode_selector = jr.uniform(key1, (n_samples, 1)) < weight

        # Sample from both modes
        samples1 = mean1 + jr.normal(key2, (n_samples, 1)) * jnp.sqrt(var)
        samples2 = mean2 + jr.normal(key3, (n_samples, 1)) * jnp.sqrt(var)

        # Select samples based on the mode selector
        samples_ND = jnp.where(mode_selector, samples1, samples2)

        return samples_ND

    @staticmethod
    def logpdf(x, mean1, mean2, var, weight=0.5):
        std = jnp.sqrt(var)
        mode1 = jsp.stats.norm.logpdf(x, loc=mean1, scale=std) + jnp.log(weight)
        mode2 = jsp.stats.norm.logpdf(x, loc=mean2, scale=std) + jnp.log(1 - weight)
        return jnp.logaddexp(mode1, mode2)

    def potential(self, params, data):
        if "logvar" in params:
            var = jnp.exp(params["logvar"])
        else:
            var = self.var
        mean1 = params["mean1"]
        mean2 = params["mean2"]

        return BimodalGaussian.logpdf(x=data, var=var, mean1=mean1, mean2=mean2).sum()


if __name__ == "__main__":
    key = jr.key(0)
    dist = BimodalGaussian()

    data_kwargs = {
        "n_samples": 1000,
        "mean1": 5,
        "mean2": -5,
        "var": 1,
    }
    key, data_key = jr.split(key)
    data = dist.sample(data_key, **data_kwargs)

    fig, ax = plt.subplots(2, 1)
    plot_samples(ax[0], data, label="Data", range=(-10, 10))

    # init_samples = {"mean1": 0, "mean2": 0, "logvar": 1.0}
    init_samples = {"mean1": 4, "mean2": -4}

    step_size = 0.1
    sigma = jnp.ones(len(init_samples)) * step_size
    alg = build_mh(partial(dist.potential, data=data), sigma)

    mcmc_kwargs = {
        "init_sample": init_samples,
        "n_samples": 50000,
        "burn_in": 10000,
        "thinning": 2,
    }

    key, mcmc_key = jr.split(key)
    samples, states, infos = run_mcmc(key=mcmc_key, alg=alg, **mcmc_kwargs)

    key, *samples_key = jr.split(key, 1 + len(samples["mean1"]))
    learned_samples = jax.vmap(partial(dist.sample, n_samples=1))(
        key=jnp.asarray(samples_key),
        mean1=samples["mean1"],
        mean2=samples["mean2"],
        var=jnp.exp(samples["logvar"]),
    ).squeeze(axis=1)
    print(learned_samples.shape)

    x = jnp.linspace(-10, 10, 500)
    true_pdf = jnp.exp(
        dist.logpdf(
            x=x,
            mean1=data_kwargs["mean1"],
            mean2=data_kwargs["mean2"],
            var=data_kwargs["var"],
        )
    )
    plot_samples(
        ax[1],
        learned_samples,
        label="Posterior Samples",
        range=(-10, 10),
        x=x,
        true_pdf=true_pdf,
    )
    # plot_samples(ax[1], learned_samples)

    fig.legend()
    plt.tight_layout()
    plt.show()
