from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from attr import dataclass

from bnn_pref.alg.mcmc import build_hmc, build_mh, plot_samples, run_mcmc
from bnn_pref.data import create_pref_data
from bnn_pref.utils.type import Q1, Q2, Q2D, D, Q


@dataclass
class QueryWithResponse:
    queries: Q2D
    responses: Q1


class BradleyTerry:
    @staticmethod
    def logpdf(
        features: Q2D,
        response_Q1: Q1,
        weights: D,
        beta: float = 1.0,  # rationality constant
    ):
        returns_Q2 = beta * features @ weights  # scale by rationality constant
        returns_Q1 = jnp.take(returns_Q2, response_Q1)
        return returns_Q1 - jsp.special.logsumexp(returns_Q2, axis=1)

    @staticmethod
    def potential(params: D, data: QueryWithResponse):
        return BradleyTerry.logpdf(data.queries, data.responses, weights=params).sum()


if __name__ == "__main__":
    n_demos = 200  # check RLHF
    n_feats = 100
    n_queries = 800

    key = jr.key(0)
    key, key1, key2 = jr.split(key, 3)
    true_reward_D = jr.normal(key1, (n_feats,))
    demos_ND = jr.normal(key2, (n_demos, n_feats))
    returns_N = demos_ND @ true_reward_D
    returns_N = jnp.sort(returns_N)  # ascending

    key, subkey = jr.split(key)
    queries_idx_Q2, response_Q1, num_mislabels = create_pref_data(
        subkey,
        returns_N,
        n_queries,
    )
    features_Q2D = demos_ND[queries_idx_Q2]

    dist = BradleyTerry()
    data = QueryWithResponse(features_Q2D, response_Q1)
    init_samples = jnp.zeros_like(true_reward_D)
    logpdf = dist.logpdf(features_Q2D, response_Q1, weights=true_reward_D)

    sigma = 0.05
    alg = build_mh(partial(dist.potential, data=data), sigma)

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
    )  # (n_particles, n_features)

    mean_weight_D = samples.mean(axis=0)
    pred_response_Q1 = (features_Q2D @ mean_weight_D).argmax(axis=1, keepdims=True)
    acc = jnp.mean(pred_response_Q1 == response_Q1)
    print(f"Accuracy: {acc}")
