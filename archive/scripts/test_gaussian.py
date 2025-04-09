from functools import partial

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
from run_gaussian import BimodalGaussian

# # Parameters for the bimodal Gaussian
# mean1 = -2.0  # Mean of the first Gaussian
# mean2 = 3.0  # Mean of the second Gaussian
# shared_variance = 1.0  # Fixed shared variance
# weight = 0.5  # Equal weight for both Gaussians

# # Number of samples
# total_samples = 1000

# # Generate data
# samples1 = np.random.normal(
#     mean1, np.sqrt(shared_variance), int(total_samples * weight)
# )
# samples2 = np.random.normal(
#     mean2, np.sqrt(shared_variance), int(total_samples * weight)
# )
# data = np.concatenate([samples1, samples2])

# # Shuffle the data
# np.random.shuffle(data)


# Plot the data
# plt.hist(data, bins=50, density=True, alpha=0.6, color="g")
# plt.title("Histogram of Bimodal Gaussian Data")
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.show()


# # Define the log-likelihood function for the bimodal Gaussian
# def log_likelihood(params, data):
#     mean1, mean2 = params
#     variance = shared_variance  # Fixed shared variance

#     # Compute the likelihood for each Gaussian
#     likelihood1 = jax.scipy.stats.norm.logpdf(data, mean1, jnp.sqrt(variance))
#     likelihood2 = jax.scipy.stats.norm.logpdf(data, mean2, jnp.sqrt(variance))

#     # Combine the likelihoods with equal weights
#     combined_likelihood = jnp.logaddexp(likelihood1, likelihood2) - jnp.log(2)
#     return jnp.sum(combined_likelihood)


# # Define the log-prior function (assume flat priors)
# def log_prior(params):
#     return 0.0  # Flat prior


# # Define the log-posterior function
# def log_posterior(params, data):
#     return log_prior(params) + log_likelihood(params, data)


# # Wrap the log-posterior for BlackJAX
# def potential(params, data):
#     return log_posterior(params, data)


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


if __name__ == "__main__":
    data_cfg = {
        "mean1": -2.0,
        "mean2": 3.0,
        "var": 1.0,
        "weight": 0.5,
        "n_samples": 1000,
    }

    key = jr.key(0)
    dist = BimodalGaussian()
    key, data_key = jr.split(key)
    data = dist.sample(data_key, **data_cfg)

    # Initialize parameters and sampler
    initial_params = {"mean1": 0.0, "mean2": 0.0}
    rng_key = jax.random.PRNGKey(42)
    inv_mass_matrix = np.array([0.5, 0.01])
    step_size = 1e-3

    sampler = blackjax.nuts(
        partial(dist.potential, data=data, fixed_logvar=jnp.log(data_cfg["var"])),
        step_size,
        inv_mass_matrix,
    )

    mcmc_cfg = {
        "init_sample": initial_params,
        "n_samples": 2000,
        "burn_in": 1000,
        "thinning": 2,
    }

    keys, subkey = jax.random.split(rng_key)
    kernel = sampler.step
    state = sampler.init(initial_params)
    states = inference_loop(subkey, kernel, state, mcmc_cfg["n_samples"])
    states_sampling = jax.tree.map(
        lambda x: x[mcmc_cfg["burn_in"] :: mcmc_cfg["thinning"]],
        states,
    )
    posterior_samples = states_sampling.position
    posterior_samples = jnp.stack(
        [posterior_samples["mean1"], posterior_samples["mean2"]], axis=1
    )

    # Sort the means in each sample so mean1 is always less than mean2
    sorted_samples = jnp.sort(posterior_samples, axis=1)

    # Convert to ArviZ InferenceData format with sorted samples
    idata = az.from_dict(
        posterior={
            "mean1": jnp.expand_dims(sorted_samples[:, 0], 0),  # Smaller mean
            "mean2": jnp.expand_dims(sorted_samples[:, 1], 0),  # Larger mean
        }
    )

    # Plot posterior distributions with ArviZ
    az.plot_posterior(
        idata,
        var_names=["mean1", "mean2"],
        ref_val=[data_cfg["mean1"], data_cfg["mean2"]],
        hdi_prob=0.94,
    )
    plt.show()

    # Trace plots to check convergence of chains
    plt.figure()
    az.plot_trace(idata)
    plt.show()

    # Summary statistics (means, HDIs, etc.)
    summary_stats = az.summary(idata, hdi_prob=0.94)
    print(summary_stats)
