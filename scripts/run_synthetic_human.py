import logging
from dataclasses import dataclass
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpl
import jax.random as jr
import jax.scipy as jsp
import numpy as np

from bnn_pref.alg.mcmc import build_hmc, build_mh, plot_samples, run_mcmc
from bnn_pref.data import create_pref_data
from bnn_pref.utils.type import Q1, Q2, Q2D, SD, D, Q

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


def alignment_metric(true_D: D, est_SD: SD):
    """
    assumes unit L2 norm!
    """
    m = (est_SD @ true_D) / (jnpl.norm(est_SD, axis=1) * jnpl.norm(true_D, axis=0))
    return jnp.mean(m)


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
        ll_Q = BradleyTerry.logpdf(
            features_Q2D=data.queries,
            response_Q1=data.responses,
            weights_D=params,
        )
        # prior = # just uniform log 1
        joint_ll = ll_Q.sum()
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
        ranked_returns=returns_N,
        n_queries=n_queries,
        noisy_prefs=False,
        bt_beta=1.0,
    )
    features_Q2D = demos_ND[queries_idx_Q2]

    return features_Q2D, response_Q1


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    # check RLHF paper
    data_kw = cfg["data"]
    mcmc_kw = cfg["mcmc"]

    # * generate true weights + preference data
    key = jr.key(0)
    key, key1, key2 = jr.split(key, 3)
    true_reward_D = get_gaussian_vector(key1, data_kw["n_feats"], normalize=True)
    features_Q2D, response_Q1 = generate_pref_data(key2, true_reward_D, **data_kw)
    data = QueryWithResponse(features_Q2D, response_Q1)

    # * build + run sampler
    dist = BradleyTerry()
    key, *subkeys = jr.split(key, 3)
    init_sample = get_gaussian_vector(subkeys[0], len(true_reward_D), normalize=True)
    # init_sample = jnp.zeros_like(true_reward_D)
    sigma = 0.05
    alg = build_mh(partial(dist.potential, data=data), sigma)

    samples_SD = run_mcmc(subkeys[1], alg=alg, init_sample=init_sample, **mcmc_kw)
    samples_SD /= jnp.linalg.norm(samples_SD, axis=1, keepdims=True)

    # * posterior check
    mean_weight_D = samples_SD.mean(axis=0)
    pred_response_Q1 = (features_Q2D @ mean_weight_D).argmax(axis=1, keepdims=True)
    acc = jnp.mean(pred_response_Q1 == response_Q1)

    print(f"N={data_kw['n_demos']}, Q={data_kw['n_queries']}, D={data_kw['n_feats']}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Weight L2: {jnp.linalg.norm(true_reward_D - mean_weight_D):.2f}")
    print(f"Cosine Sim: {alignment_metric(true_reward_D, samples_SD):.2f}")
    # logpdf = dist.logpdf(features_Q2D, response_Q1, weights=true_reward_D)


if __name__ == "__main__":
    main()
