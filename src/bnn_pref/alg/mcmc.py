from typing import Callable

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt


def run_mcmc(
    key,
    alg,
    init_sample,
    n_samples: int,
    burn_in: int = 0,
    thinning: int = 1,
    normalize: bool = False,
):
    def inference_loop(key, kernel, initial_state, n_samples, normalize=False):
        @jax.jit
        def one_step(state, key):
            state, info = kernel(key, state)

            if normalize:
                w_normalized = state.position / jnp.linalg.norm(state.position)
                state = state._replace(position=w_normalized)
            return state, (state, info)

        keys = jr.split(key, n_samples)
        _, (states, infos) = jax.lax.scan(f=one_step, init=initial_state, xs=keys)
        return states, infos

    state = alg.init(init_sample)
    kernel = alg.step
    states, infos = inference_loop(key, kernel, state, n_samples, normalize)
    samples = jax.tree_map(lambda x: x[burn_in::thinning], states.position)
    return samples, states, infos


def build_mh(log_pdf: Callable, sigma: float):
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


def plot_trace(key, samples_SD, true_D, bbox_dict=None):
    """limit to 8 dimensions"""
    limit = 8
    n_dims = samples_SD.shape[1]
    n_display = min(n_dims, limit)

    fig, axs = plt.subplots(4, 2, figsize=(12, 8))
    dims_display = jnp.sort(jr.choice(key, n_dims, shape=(n_display,), replace=False))
    for i in range(n_display):
        ax = axs[i // 2, i % 2]

        d = dims_display[i] if n_dims > limit else i
        true = true_D[d]
        est = samples_SD[:, d].mean()

        ax.plot(samples_SD[:, d], label="Samples")
        ax.axhline(true, color="red", label="True", lw=0.5)
        ax.set_title(f"Trace for dim {d + 1}: {true:.2f} vs. {est:.2f} ")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.set_ylim(-1.1, 1.1)

    # legends
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("True vs. Estimated Params")
    fig.tight_layout()

    # Hyperparameters to display in box
    text_content = "\n".join([f"{k}: {v:.2f}" for k, v in bbox_dict.items()])
    fig.text(
        x=0.05,  # Right edge alignment
        y=0.98,  # Top edge alignment
        s=text_content,
        fontfamily="monospace",
        fontsize=8,
        linespacing=1.0,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
            edgecolor="black",
            pad=0.8,
        ),
    )
