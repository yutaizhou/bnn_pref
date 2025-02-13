from typing import Callable, Tuple

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr

from data import create_pref_data

# def log_prob(x: Tuple[float, float], beta: float = 1.0) -> float:
#     raise NotImplementedError


def log_pdf(x):
    return -0.5 * jnp.sum(x**2)


def run_mcmc(
    key,
    log_prob: Callable,
    initial_position: jnp.ndarray,
    n_samples: int,
    step_size: float,
    burn_in: int = 0,
    thinning: int = 1,
):
    def inference_loop(key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, key):
            state, _ = kernel(key, state)
            return state, state.position

        key = jax.random.split(key, num_samples)
        _, positions = jax.lax.scan(one_step, initial_state, key)
        return positions

    kernel = blackjax.mcmc.random_walk.normal(
        jnp.ones_like(initial_position) * step_size
    )
    rwm = blackjax.rmh(log_prob, kernel)
    state = rwm.init(initial_position)
    samples = inference_loop(key, rwm.step, state, n_samples)
    samples = samples[burn_in::thinning]
    return samples


if __name__ == "__main__":
    key = jr.key(0)
    n_demos = 200  # check RLHF
    n_feats = 1000
    n_queries = 200

    demos = jr.normal(key, (n_demos, n_feats))
    true_reward = jr.normal(key, (n_feats,))

    returns = demos @ true_reward
    returns = jnp.sort(returns)

    key, subkey = jr.split(key)
    queries_Q2, labels_Q1, num_mislabels = create_pref_data(subkey, returns, n_queries)

    mcmc_kwargs = {
        "n_samples": 11000,
        "step_size": 0.1,
        "initial_position": jnp.zeros_like(true_reward),
        "burn_in": 1000,
        "thinning": 2,
    }
    samples = run_mcmc(
        key=key,
        log_prob=log_pdf,
        **mcmc_kwargs,
    )
    print(samples.shape)
    likelihood = jnp.exp(jax.vmap(log_pdf)(samples)).mean()
    print(likelihood)
