from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np

from bnn_pref.alg.mcmc import build_hmc, build_mh, plot_samples, run_mcmc
from bnn_pref.data import create_pref_data
from bnn_pref.utils.type import Q1, Q2, Q2D, D, Q


def get_gaussian_vector(key, dim: int, normalize: bool = True) -> D:
    vec = jr.normal(key, dim)
    if normalize:
        vec /= jnp.linalg.norm(vec)
    return vec


@dataclass
class QueryWithResponse:
    queries: Q2D
    responses: Q1


class BradleyTerry:
    @staticmethod
    def logpdf(
        features_Q2D: Q2D,
        response_Q1: Q1,
        weights_D: D,
        beta: float = 1.0,  # rationality constant
    ) -> Q1:
        returns_Q2 = beta * (features_Q2D @ weights_D)
        returns_Q1 = jnp.take(returns_Q2, response_Q1)
        return returns_Q1 - jsp.special.logsumexp(returns_Q2, axis=1)

    @staticmethod
    def potential(params: D, data: QueryWithResponse):
        ll = BradleyTerry.logpdf(
            features_Q2D=data.queries,
            response_Q1=data.responses,
            weights_D=params,
        )
        joint_ll = ll.sum()
        return joint_ll


def generate_pref_data(
    key,
    params_D: D,
    n_demos: int,
    n_feats: int,
    n_queries: int,
):
    key, key1, key2 = jr.split(key, 3)

    demos_ND = jr.normal(key1, (n_demos, n_feats))
    demos_ND /= jnp.linalg.norm(demos_ND, axis=1, keepdims=True)

    returns_N = demos_ND @ params_D
    returns_N = jnp.sort(returns_N)  # ascending

    queries_idx_Q2, response_Q1, num_mislabels = create_pref_data(
        key2,
        returns_N,
        n_queries,
    )
    features_Q2D = demos_ND[queries_idx_Q2]

    return features_Q2D, response_Q1


if __name__ == "__main__":
    # check RLHF paper
    data_kwargs = {
        "n_demos": 200,
        "n_feats": 5,
        "n_queries": 1000,
    }

    key = jr.key(0)
    key, key1, key2 = jr.split(key, 3)
    true_reward_D = get_gaussian_vector(key1, data_kwargs["n_feats"], normalize=True)
    features_Q2D, response_Q1 = generate_pref_data(key2, true_reward_D, **data_kwargs)

    dist = BradleyTerry()
    data = QueryWithResponse(features_Q2D, response_Q1)
    # logpdf = dist.logpdf(features_Q2D, response_Q1, weights=true_reward_D)

    key, subkey = jr.split(key)
    init_samples = get_gaussian_vector(subkey, len(true_reward_D), normalize=True)
    # init_samples = jnp.zeros_like(true_reward_D)
    sigma = 0.05
    alg = build_mh(partial(dist.potential, data=data), sigma)

    mcmc_kwargs = {
        "n_samples": 21000,
        "burn_in": 5000,
        "thinning": 2,
    }
    key, subkey = jr.split(key)
    samples_SD = run_mcmc(key=subkey, alg=alg, init_samples=init_samples, **mcmc_kwargs)
    samples_SD /= jnp.linalg.norm(samples_SD, axis=1, keepdims=True)

    mean_weight_D = samples_SD.mean(axis=0)
    pred_response_Q1 = (features_Q2D @ mean_weight_D).argmax(axis=1, keepdims=True)
    acc = jnp.mean(pred_response_Q1 == response_Q1)
    print(f"Accuracy: {acc:.2%}")
    print(f"Weight L2: {jnp.linalg.norm(true_reward_D - mean_weight_D):.2f}")
