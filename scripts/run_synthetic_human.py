import logging
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import arviz as az
import hydra
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpl
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np

from bnn_pref.alg.mcmc import build_hmc, build_mh, plot_samples, plot_trace, run_mcmc
from bnn_pref.data import create_pref_data
from bnn_pref.utils.type import Q1, Q2, Q2D, SD, D, Q
from bnn_pref.utils.utils import (
    alignment_metric,
    get_gaussian_vector,
    get_uniform_vector,
    tile_first_dim,
)

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


def print_summary(cfg, samples_SD, acc: float, align: float, seed: int):
    data_kw = cfg["data"]
    mcmc_kw = cfg["mcmc"]

    print(f"Seed: {seed}")
    print(f"N={data_kw['n_demos']}, Q={data_kw['n_queries']}, D={data_kw['n_feats']}")
    print(
        f"{mcmc_kw['n_samples']} samples w/ {mcmc_kw['burn_in']} burn-in, then {mcmc_kw['thinning']} thinning"
    )
    print(f"MCMC Samples: {samples_SD.shape}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Cosine Sim: {align:.2f}")


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
        returns_Q1 = jnp.take_along_axis(returns_Q2, response_Q1, axis=1)
        return returns_Q1 - jax.nn.logsumexp(returns_Q2, axis=1, keepdims=True)

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
    # demos_ND /= jnp.linalg.norm(demos_ND, axis=1, keepdims=True)

    returns_N = demos_ND @ params_D
    sorted_idx = jnp.argsort(returns_N)  # ascending
    returns_N = returns_N[sorted_idx]
    demos_ND = demos_ND[sorted_idx]

    queries_idx_Q2, response_Q1, num_mislabels = create_pref_data(
        key2,
        ranked_returns=returns_N,
        n_queries=n_queries,
        noisy_prefs=False,
        bt_beta=1.0,
    )
    features_Q2D = demos_ND[queries_idx_Q2]  # todo check?

    return features_Q2D, response_Q1


def compute_accuracy1(samples_SD, features_Q2D, response_Q1):
    # * approach 1: mean sample from posterior
    mean_weight_D = samples_SD.mean(axis=0)
    mean_weight_D /= jnpl.norm(mean_weight_D)
    probs_Q2 = jax.nn.softmax(features_Q2D @ mean_weight_D, axis=1)

    pred_response_Q = probs_Q2.argmax(axis=1)
    # pred_response_Q = jnp.exp(features_Q2D @ mean_weight_D).argmax(axis=1) # approach 0
    acc = jnp.mean(pred_response_Q == response_Q1.squeeze())
    return acc


def compute_accuracy2(samples_SD, features_Q2D, response_Q1):
    # * approach 2: mean predictive probability from posterior
    @partial(jax.vmap, in_axes=(None, 0))
    def compute_postpred_mean(params_SD, features_2D):
        returns_S2 = params_SD @ features_2D.T
        probs_S2 = jax.nn.softmax(returns_S2, axis=1)  # BT model
        postpred_mean_prob_2 = probs_S2.mean(0)
        return postpred_mean_prob_2

    samples_SD /= jnpl.norm(samples_SD, axis=1, keepdims=True)
    probs_Q2 = compute_postpred_mean(samples_SD, features_Q2D)

    pred_response_Q = probs_Q2.argmax(axis=1)
    acc = jnp.mean(pred_response_Q == response_Q1.squeeze())
    return acc


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    # check RLHF paper
    data_kw = cfg["data"]
    mcmc_kw = cfg["mcmc"]
    dist = BradleyTerry()

    seed = int(datetime.now().timestamp()) if cfg["seed"] == -1 else cfg["seed"]
    # seed = 1740514373 # weird ass multimodal stuff
    key = jax.random.key(seed=seed)

    # * generate true weights + preference data
    key, key1, key2 = jr.split(key, 3)
    true_reward_D = get_gaussian_vector(key1, dim=data_kw["n_feats"], normalize=True)
    # true_reward_D = get_uniform_vector(key1, dim=data_kw["n_feats"], normalize=True)
    features_Q2D, response_Q1 = generate_pref_data(key2, true_reward_D, **data_kw)
    data = QueryWithResponse(features_Q2D, response_Q1)

    # * build + run sampler
    key, key1, key2 = jr.split(key, 3)
    # init_sample = get_gaussian_vector(key2, len(true_reward_D), normalize=True)
    init_sample = jnp.zeros_like(true_reward_D)
    alg = build_mh(partial(dist.potential, data=data), sigma=mcmc_kw["sigma"])
    samples_SD, states, infos = run_mcmc(
        key1,
        alg=alg,
        init_sample=init_sample,
        **{k: mcmc_kw[k] for k in ["n_samples", "burn_in", "thinning", "normalize"]},
    )

    # * posterior check
    # two approaches to computing acc should now be equivalent
    acc = compute_accuracy2(samples_SD, features_Q2D, response_Q1)
    align = alignment_metric(true_reward_D, samples_SD)
    print_summary(cfg, samples_SD, acc, align, seed)
    # logpdf = dist.logpdf(features_Q2D, response_Q1, weights=true_reward_D)

    # * plotting
    all_samples = jnp.concat([init_sample[None, :], samples_SD], axis=0)
    bbox_dict = {
        "D": data_kw["n_feats"],
        "Q": data_kw["n_queries"],
        "Acc": acc,
        "m": align,
    }
    plot_trace(key, all_samples, true_reward_D, bbox_dict=bbox_dict)
    if cfg["show_fig"]:
        plt.show()
    if cfg["save_fig"]:
        plt.savefig(f"{cfg['paths']['output_dir']}/trace.png")


if __name__ == "__main__":
    main()
