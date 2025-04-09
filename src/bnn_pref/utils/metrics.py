from dataclasses import dataclass
from functools import partial
from typing import Callable

import distrax
import ipdb
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpl
import optax

from bnn_pref.alg.ekf_subspace import EKFBeliefState
from bnn_pref.data.pref_utils import QueryWithResponse
from bnn_pref.utils.type import NTD, SD, D


@dataclass()
class MeanStd:
    array: jnp.ndarray  # (n_seeds, nq_update)
    mean: jnp.ndarray = None
    std: jnp.ndarray = None

    def __post_init__(self):
        self.mean = self.array.mean(axis=0)
        self.std = self.array.std(axis=0)


def compute_reward_nn(fn: Callable, demos_NTD: NTD):
    """
    fn: (NTD) -> (N) logit/return of a single trajectory
    """
    rewards_N = fn(demos_NTD)
    return rewards_N


def compute_acc_nn(fn: Callable, data: QueryWithResponse):
    """
    fn: (N2TD) -> (N2) logits of both items in a pairwise query
    """
    features_Q2TD, response_Q1 = data.queries_Q2TD, data.responses_Q1
    logits_Q2 = fn(features_Q2TD)
    pred_response_Q = logits_Q2.argmax(axis=1)
    acc = jnp.mean(pred_response_Q == response_Q1.squeeze())
    return acc


def compute_acc_nn_bma(
    key, sub2full_fn: Callable, bel: EKFBeliefState, data: QueryWithResponse
):
    """
    sub2full_fn: (params, N2TD) -> (N2) logits of both items in a pairwise query
    """
    queries_Q2TD, labels_Q1 = data.queries_Q2TD, data.responses_Q1
    n_models = 10
    mean, cov = bel.mean, bel.cov
    dist = distrax.MultivariateNormalFullCovariance(mean, cov)
    ss_params = dist.sample(seed=key, sample_shape=(n_models,))

    sub2full_fn = jax.vmap(sub2full_fn, in_axes=(None, 0))  # input: param, context
    sub2full_fn = jax.vmap(sub2full_fn, in_axes=(0, None))
    logits_MQ2 = sub2full_fn(ss_params, queries_Q2TD)
    probs_MQ2 = jax.nn.softmax(logits_MQ2, axis=2)
    probs_Q2 = probs_MQ2.mean(0)

    pred_response_Q = probs_Q2.argmax(axis=1)
    acc = jnp.mean(pred_response_Q == labels_Q1.squeeze())
    return acc


def compute_logpdf_nn(fn: Callable, data: QueryWithResponse):
    """
    fn: (N2TD) -> (N2) logits of both items in a pairwise query
    """
    features_Q2TD, labels_Q1 = data.queries_Q2TD, data.responses_Q1
    logits_Q2 = fn(features_Q2TD)
    logits_Q1 = jnp.take_along_axis(logits_Q2, labels_Q1, axis=1)
    ll_Q1 = logits_Q1 - jax.nn.logsumexp(logits_Q2, axis=1, keepdims=True)
    avg_ll = ll_Q1.mean()

    # * CE is just Negative LL, so this should be equivalent
    # avg_ll2 = -optax.losses.softmax_cross_entropy(
    #     logits_Q2,
    #     jax.nn.one_hot(labels_Q1.squeeze(), 2),
    # ).mean()

    return avg_ll


def compute_logpdf_ensemble(fn: Callable, data: QueryWithResponse):
    """
    fn: (N2TD) -> (NM2) logits of both items in a pairwise query, across M models
    """
    features_Q2TD, labels_Q1 = data.queries_Q2TD, data.responses_Q1
    logits_QM2 = fn(features_Q2TD)
    probs_QM2 = jax.nn.softmax(logits_QM2, axis=2)
    probs_Q2 = probs_QM2.mean(1)

    llik_Q1 = jnp.log(jnp.take_along_axis(probs_Q2, labels_Q1, axis=1))
    avg_ll = llik_Q1.mean()

    return avg_ll


def compute_acc_ensemble(fn: Callable, data: QueryWithResponse):
    """
    fn: (N2TD) -> (NM2) logits of both items in a pairwise query, across M models
    """
    features_Q2TD, labels_Q1 = data.queries_Q2TD, data.responses_Q1
    logits_QM2 = fn(features_Q2TD)
    probs_QM2 = jax.nn.softmax(logits_QM2, axis=2)
    probs_Q2 = probs_QM2.mean(1)

    pred_response_Q = probs_Q2.argmax(axis=1)
    acc = jnp.mean(pred_response_Q == labels_Q1.squeeze())
    return acc


def compute_logpdf_nn_bel(
    key, sub2full_predict_logits: Callable, bel: EKFBeliefState, data: QueryWithResponse
):
    # todo: this is not used anywhere. need to debug
    n_models = 10
    mean, cov = bel.mean, bel.cov
    dist = distrax.MultivariateNormalFullCovariance(mean, cov)
    ss_params = dist.sample(seed=key, sample_shape=(n_models,))
    blah = jax.vmap(sub2full_predict_logits, in_axes=(None, 0))  # input: param, context
    blah = jax.vmap(blah, in_axes=(0, None))
    logits_MQ2 = blah(ss_params, data.queries_Q2TD)

    def compute_ll(logits_Q2):
        return -optax.losses.softmax_cross_entropy(
            logits_Q2,
            jax.nn.one_hot(data.responses_Q1.squeeze(), 2),
        ).mean()

    avg_ll = jax.vmap(compute_ll)(logits_MQ2).mean()
    return avg_ll


def compute_pref_ranking_acc(reward_predictor: Callable, data: QueryWithResponse):
    # todo why same output as compute_accuracy_nn?
    features_Q2TD, response_Q1 = data.queries_Q2TD, data.responses_Q1
    left_rewards_Q = reward_predictor(features_Q2TD[:, 0]).squeeze()
    right_rewards_Q = reward_predictor(features_Q2TD[:, 1]).squeeze()
    pred_prefs_Q = (left_rewards_Q < right_rewards_Q).astype(int)
    return jnp.mean(pred_prefs_Q == response_Q1.squeeze())


def compute_accuracy1_mcmc(samples_SD, data: QueryWithResponse, reward_fn: Callable):
    features_Q2TD, response_Q1 = data.queries_Q2TD, data.responses_Q1
    # * approach 1: mean sample from posterior
    mean_weight_D = samples_SD.mean(axis=0)
    mean_weight_D /= jnpl.norm(mean_weight_D)
    probs_Q2 = jax.nn.softmax(reward_fn(features_Q2TD, mean_weight_D), axis=1)

    pred_response_Q = probs_Q2.argmax(axis=1)
    # pred_response_Q = jnp.exp(reward_fn(features_Q2TD, mean_weight_D).argmax(axis=1) # approach 0
    acc = jnp.mean(pred_response_Q == response_Q1.squeeze())
    return acc


def compute_accuracy2_mcmc(samples_SD, data: QueryWithResponse, reward_fn: Callable):
    features_Q2TD, response_Q1 = data.queries_Q2TD, data.responses_Q1

    # * approach 2: mean predictive probability from posterior
    @partial(jax.vmap, in_axes=(None, 0))
    def compute_postpred_mean(params_SD, features_2TD):
        returns_S2 = reward_fn(features_2TD, params_SD.T).T  # todo make this robust
        probs_S2 = jax.nn.softmax(returns_S2, axis=1)  # BT model
        postpred_mean_prob_2 = probs_S2.mean(0)
        return postpred_mean_prob_2

    samples_SD /= jnpl.norm(samples_SD, axis=1, keepdims=True)
    probs_Q2 = compute_postpred_mean(samples_SD, features_Q2TD)

    pred_response_Q = probs_Q2.argmax(axis=1)
    acc = jnp.mean(pred_response_Q == response_Q1.squeeze())
    return acc


def alignment_metric(true_D: D, est_SD: SD):
    """
    Average cosine similarity of MCMC samples wrt true parameter.
    Assumes unit L2 norm!
    """
    m = (est_SD @ true_D) / (jnpl.norm(est_SD, axis=1) * jnpl.norm(true_D, axis=0))
    return jnp.mean(m)
