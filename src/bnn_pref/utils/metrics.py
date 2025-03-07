from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpl
import jax.random as jr
import jax.scipy as jsp

from bnn_pref.data.data import QueryWithResponse
from bnn_pref.utils.type import ND, Q1, Q2, Q2D, SD, D


def compute_accuracy_nn(pref_predictor: Callable, data: QueryWithResponse):
    features_Q2TD, response_Q1 = data.queries_Q2D, data.responses_Q1
    logits_Q2 = pref_predictor(features_Q2TD)
    pred_response_Q = logits_Q2.argmax(axis=1)
    acc = jnp.mean(pred_response_Q == response_Q1.squeeze())
    return acc


def compute_reward_nn(reward_predictor: Callable, demos_ND: ND):
    rewards_N = reward_predictor(demos_ND)
    return rewards_N


def compute_pref_ranking_acc(reward_predictor: Callable, data: QueryWithResponse):
    # todo why same output as compute_accuracy_nn?
    features_Q2TD, response_Q1 = data.queries_Q2D, data.responses_Q1
    left_rewards_Q = reward_predictor(features_Q2TD[:, 0]).squeeze()
    right_rewards_Q = reward_predictor(features_Q2TD[:, 1]).squeeze()
    pred_prefs_Q = (left_rewards_Q < right_rewards_Q).astype(int)
    return jnp.mean(pred_prefs_Q == response_Q1.squeeze())


def compute_accuracy1_mcmc(samples_SD, data: QueryWithResponse, reward_fn: Callable):
    features_Q2D, response_Q1 = data.queries_Q2D, data.responses_Q1
    # * approach 1: mean sample from posterior
    mean_weight_D = samples_SD.mean(axis=0)
    mean_weight_D /= jnpl.norm(mean_weight_D)
    probs_Q2 = jax.nn.softmax(reward_fn(features_Q2D, mean_weight_D), axis=1)

    pred_response_Q = probs_Q2.argmax(axis=1)
    # pred_response_Q = jnp.exp(reward_fn(features_Q2D, mean_weight_D).argmax(axis=1) # approach 0
    acc = jnp.mean(pred_response_Q == response_Q1.squeeze())
    return acc


def compute_accuracy2_mcmc(samples_SD, data: QueryWithResponse, reward_fn: Callable):
    features_Q2D, response_Q1 = data.queries_Q2D, data.responses_Q1

    # * approach 2: mean predictive probability from posterior
    @partial(jax.vmap, in_axes=(None, 0))
    def compute_postpred_mean(params_SD, features_2D):
        returns_S2 = reward_fn(features_2D, params_SD.T).T  # todo make this robust
        probs_S2 = jax.nn.softmax(returns_S2, axis=1)  # BT model
        postpred_mean_prob_2 = probs_S2.mean(0)
        return postpred_mean_prob_2

    samples_SD /= jnpl.norm(samples_SD, axis=1, keepdims=True)
    probs_Q2 = compute_postpred_mean(samples_SD, features_Q2D)

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
