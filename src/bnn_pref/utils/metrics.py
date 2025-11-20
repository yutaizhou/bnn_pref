from dataclasses import dataclass
from functools import partial
from typing import Callable

import distrax
import ipdb
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpl
from jax import Array
from jaxtyping import Scalar

from bnn_pref.alg.agent_ekf import EKFBeliefState
from bnn_pref.data.pref_utils import QueryFeaturesAndResponses
from bnn_pref.utils.type import NTD, SD, D


@dataclass()
class MeanStd:
    array: jnp.ndarray  # (n_seeds, nq_update)
    mean: jnp.ndarray = None
    std: jnp.ndarray = None

    def __post_init__(self):
        self.mean = self.array.mean(axis=0)
        self.std = self.array.std(axis=0)

    def get_stats(self):
        return {"mean": self.mean, "std": self.std}


def compute_logpdf_nn(
    fn: Callable, data: QueryFeaturesAndResponses, chunk_size: int = 32
):
    """
    fn: (2TD) -> (2,) logits of both items in a pairwise query
    """
    features_Q2TD, responses_Q1 = data.queries_Q2TD, data.responses_Q1
    # logits_Q2 = fn(features_Q2TD)
    logits_Q2 = jax.lax.map(fn, features_Q2TD, batch_size=chunk_size)
    logits_Q1 = jnp.take_along_axis(logits_Q2, responses_Q1, axis=1)
    llik_Q1 = logits_Q1 - jax.nn.logsumexp(logits_Q2, axis=1, keepdims=True)
    avg_ll = llik_Q1.mean()

    # * CE is just Negative LL, so this should be equivalent
    # avg_ll2 = -optax.losses.softmax_cross_entropy(
    #     logits_Q2,
    #     jax.nn.one_hot(labels_Q1.squeeze(), 2),
    # ).mean()

    return avg_ll


def compute_accuracy1_mcmc(
    samples_SD, data: QueryFeaturesAndResponses, reward_fn: Callable
):
    features_Q2TD, responses_Q1 = data.queries_Q2TD, data.responses_Q1
    # * approach 1: mean sample from posterior
    mean_weight_D = samples_SD.mean(axis=0)
    mean_weight_D /= jnpl.norm(mean_weight_D)
    # probs_Q2 = jax.nn.softmax(reward_fn(features_Q2TD, mean_weight_D), axis=1)
    probs_Q2 = jnp.exp(
        jax.nn.log_softmax(reward_fn(features_Q2TD, mean_weight_D), axis=1)
    )

    pred_response_Q = probs_Q2.argmax(axis=1)
    # pred_response_Q = jnp.exp(reward_fn(features_Q2TD, mean_weight_D).argmax(axis=1) # approach 0
    acc = jnp.mean(pred_response_Q == responses_Q1.squeeze())
    return acc


def compute_accuracy2_mcmc(
    samples_SD, data: QueryFeaturesAndResponses, reward_fn: Callable
):
    features_Q2TD, responses_Q1 = data.queries_Q2TD, data.responses_Q1

    # * approach 2: mean predictive probability from posterior
    @partial(jax.vmap, in_axes=(None, 0))
    def compute_postpred_mean(params_SD, features_2TD):
        returns_S2 = reward_fn(features_2TD, params_SD.T).T  # todo make this robust
        # probs_S2 = jax.nn.softmax(returns_S2, axis=1)  # BT model
        probs_S2 = jnp.exp(jax.nn.log_softmax(returns_S2, axis=1))
        postpred_mean_prob_2 = probs_S2.mean(0)
        return postpred_mean_prob_2

    samples_SD /= jnpl.norm(samples_SD, axis=1, keepdims=True)
    probs_Q2 = compute_postpred_mean(samples_SD, features_Q2TD)

    pred_response_Q = probs_Q2.argmax(axis=1)
    acc = jnp.mean(pred_response_Q == responses_Q1.squeeze())
    return acc


def compute_accuracy(probs_Q2: Array, labels_Q1: Array) -> Scalar:
    pred_Q = jnp.argmax(probs_Q2, axis=1)
    return jnp.mean(pred_Q == labels_Q1.squeeze())


def compute_logpdf(probs_Q2: Array, labels_Q1: Array) -> Scalar:
    prob_Q1 = jnp.take_along_axis(probs_Q2, labels_Q1, axis=1)
    return jnp.log(prob_Q1).mean()


def compute_ece(probs_Q2: Array, labels_Q1: Array, n_bins: int = 5) -> Scalar:
    """
    Compute Expected Calibration Error (ECE) for binary classification.
    JAX JIT/vmap friendly version.

    Args:
        probs_Q2: (Q, 2) predicted probabilities for each class
        labels_Q1: (Q,) true labels (0 or 1)
        n_bins: number of bins for calibration plot

    Returns:
        ece: scalar Expected Calibration Error
    """
    # Get confidence (max probability) and predictions
    conf_Q = jnp.max(probs_Q2, axis=1)
    pred_Q = jnp.argmax(probs_Q2, axis=1)
    correct_Q = pred_Q == labels_Q1.squeeze()

    # Create bins
    bin_boundaries = jnp.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    def compute_bin_contribution(bin_lower, bin_upper):
        """Compute ECE contribution for a single bin."""
        # Find samples in this bin
        bin_mask_Q = (bin_lower < conf_Q) & (conf_Q <= bin_upper)
        bin_count = jnp.sum(bin_mask_Q)
        bin_weight = bin_count / conf_Q.size

        # Avoid division by zero by using safe_count
        safe_count = jnp.maximum(bin_count, 1)

        # Compute accuracy and confidence for samples in this bin
        # Use jnp.where to handle empty bins gracefully
        acc_sum = jnp.sum(jnp.where(bin_mask_Q, correct_Q.astype(jnp.float32), 0.0))
        conf_sum = jnp.sum(jnp.where(bin_mask_Q, conf_Q, 0.0))

        avg_acc = acc_sum / safe_count
        avg_conf = conf_sum / safe_count

        # Return contribution (will be 0 for empty bins)
        return jnp.abs(avg_conf - avg_acc) * bin_weight

    # Vectorize over all bins
    bin_contributions = jax.vmap(compute_bin_contribution)(bin_lowers, bin_uppers)

    # Sum all contributions
    ece = jnp.sum(bin_contributions)

    return ece


def compute_brier_score(probs_Q2: Array, labels_Q1: Array) -> Scalar:
    """
    Compute Brier Score for binary classification.
    """
    labels_Q2 = jax.nn.one_hot(labels_Q1.squeeze(), 2)
    brier_score = jnp.mean(jnp.sum((probs_Q2 - labels_Q2) ** 2, axis=1))
    return brier_score


def compute_coverage_rate(
    probs_Q2: Array, labels_Q1: Array, alpha: float = 0.95
) -> Scalar:
    """
    Compute coverage rate for prediction regions.

    Args:
        alpha: confidence level (e.g., 0.95 for 95% coverage)

    Returns:
        coverage_rate: scalar coverage rate (should be close to alpha)
    """
    # Get the predicted class and its probability
    pred_Q = jnp.argmax(probs_Q2, axis=1)
    conf_Q = jnp.max(probs_Q2, axis=1)
    is_correct_Q = pred_Q == labels_Q1.squeeze()

    # For correct predictions, check if confidence is above threshold
    # For incorrect predictions, coverage is 0 (not covered)
    coverage_Q = jnp.where(is_correct_Q, conf_Q >= alpha, 0.0)

    coverage_rate = jnp.mean(coverage_Q)
    return coverage_rate


def compute_sharpness(probs_Q2: Array) -> Scalar:
    """
    Compute sharpness of predictions.

    Returns:
        sharpness: scalar sharpness of predictions
    """
    return jnp.var(probs_Q2, axis=1).mean()


def compute_alignment(true_D: D, est_SD: SD):
    """
    Average cosine similarity of MCMC samples wrt true parameter.
    Assumes unit L2 norm!
    """
    m = (est_SD @ true_D) / (jnpl.norm(est_SD, axis=1) * jnpl.norm(true_D, axis=0))
    return jnp.mean(m)
