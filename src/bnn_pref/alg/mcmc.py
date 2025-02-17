from typing import Callable

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt


def inference_loop(key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, key):
        state, _ = kernel(key, state)
        return state, state

    keys = jr.split(key, num_samples)
    _, states = jax.lax.scan(f=one_step, init=initial_state, xs=keys)
    return states


def run_mcmc(
    key,
    alg,
    init_samples,
    n_samples: int,
    burn_in: int = 0,
    thinning: int = 1,
):
    state = alg.init(init_samples)
    kernel = alg.step
    states = inference_loop(key, kernel, state, n_samples)
    samples = states.position
    samples = jax.tree_map(lambda x: x[burn_in::thinning], samples)
    return samples


def build_mh(log_pdf: Callable, sigma):
    kernel = blackjax.mcmc.random_walk.normal(sigma)
    rmh = blackjax.rmh(log_pdf, kernel)
    return rmh


def build_hmc(
    log_pdf: Callable,
    init_samples: jnp.ndarray,
    step_size: float,
    num_integration_steps: int = 60,
):
    inv_mass_matrix = jnp.ones_like(init_samples)
    hmc = blackjax.hmc(log_pdf, step_size, inv_mass_matrix, num_integration_steps)
    return hmc


def plot_samples(ax, samples, label, range, x=None, true_pdf=None):
    ax.hist(
        samples[:, 0],
        bins=50,
        density=True,
        alpha=0.7,
        label=label,
        range=range,
    )

    if x is not None and true_pdf is not None:
        ax.plot(x, true_pdf, label="True Distribution (1st dimension)", color="red")


def plot_trace(samples):
    plt.figure(figsize=(12, 10))
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.plot(samples[:, i])
        plt.title(f"Trace plot for dimension {i + 1}")
        plt.xlabel("Iteration")
        plt.ylabel("Value")

    plt.tight_layout()
