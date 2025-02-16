import math
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from bnn_pref.utils.type import Q1, Q2, N


def bt_likelihood(return_i: float, return_j: float, beta: float = 1.0) -> float:
    """
    computes likelihood of preference tj > ti, given their rewards
    """
    exp_i = jnp.exp(return_i * beta)
    exp_j = jnp.exp(return_j * beta)
    return exp_j / (exp_i + exp_j)


def random_query_iterator(key, n: int, n_queries: int):
    """
    Note this does not return ti=tj, and migth return duplicates
    """
    for _ in range(n_queries):
        ti = jr.randint(key=key, shape=(), minval=0, maxval=n - 1)
        tj = jr.randint(key=key, shape=(), minval=ti + 1, maxval=n)
        yield ti, tj


def create_pref_data(
    key,
    ranked_returns: N,
    n_queries: int = -1,
    use_delta: bool = False,
    delta_rank: int = 1,
    delta_reward: float = 0,
    noisy_prefs: bool = False,
    bt_beta: float = 1.0,
    skip_threshold: float = -jnp.inf,
    mistake_prob: float = 0.0,
) -> Tuple[Q2, Q1, int]:
    """
    Args:
        num_queries (int): specifies the number of pairwise comparisons between trajectories
            to use in our training set
        delta_rank (int): recovers original (just that pairwise comps can't be the same)

    Outputs:
        queries_Q2
        labels_Q1
        reward_diffs (NTD): (num_queries, 1)

    Note: demonstrations and/or returns must be ranked by increasing reward.
    """
    queries = []
    labels = []
    n_demos = len(ranked_returns)

    if not isinstance(ranked_returns, jnp.ndarray):
        ranked_returns = jnp.asarray(ranked_returns)

    num_mislabels = 0
    n_queries = n_queries if n_queries != -1 else math.comb(n_demos, 2)

    for ti, tj in random_query_iterator(key, n_demos, n_queries):
        if use_delta:
            # skip if tj is not better than ti by delta_rank or delta_reward
            if delta_rank > 1:
                if (tj - ti) < delta_rank:
                    continue
            else:
                if (ranked_returns[tj] - ranked_returns[ti]) < delta_reward:
                    continue

        label = 1  # label=1 means tj > ti

        # * irrationality: skip if both are bad
        if max(ranked_returns[tj], ranked_returns[ti]) < skip_threshold:
            continue

        # * irrationality: noisily rational prefs (beta=1 in the Bradley-Terry model)
        if noisy_prefs:
            prob = bt_likelihood(ranked_returns[ti], ranked_returns[tj], bt_beta)

            if jr.uniform(key) > prob:
                num_mislabels += 1
                label = 0

        # * irrationality: label flip mistake
        label = 1 - label if (jr.uniform(key) < mistake_prob) else label

        queries.append((ti, tj))
        labels.append(label)

    queries_Q2 = jnp.array(queries).astype(jnp.int32)
    labels_Q1 = jnp.expand_dims(jnp.array(labels), 1).astype(jnp.int32)

    return queries_Q2, labels_Q1, num_mislabels
