import math
from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from bnn_pref.utils.type import NTD, Q1, Q2, Q2TD, D, N


@dataclass
class QueryWithResponse:
    queries_Q2TD: Q2TD
    responses_Q1: Q1


class BradleyTerry:
    @staticmethod
    def logpdf(
        params_D: D,
        data: QueryWithResponse,
        reward_fn: Callable,
        beta: float = 1.0,  # rationality constant
    ) -> Q1:
        features_Q2TD, response_Q1 = data.queries_Q2TD, data.responses_Q1
        returns_Q2 = beta * reward_fn(features_Q2TD, params_D)
        returns_Q1 = jnp.take_along_axis(returns_Q2, response_Q1, axis=1)
        return returns_Q1 - jax.nn.logsumexp(returns_Q2, axis=1, keepdims=True)

    @staticmethod
    def potential(params: D, reward_fn: Callable, data: QueryWithResponse) -> float:
        ll_Q = BradleyTerry.logpdf(
            params_D=params,
            data=data,
            reward_fn=reward_fn,
        )
        # prior = # just uniform log 1
        joint_ll = ll_Q.sum()
        return joint_ll


def bt_likelihood(return_i: float, return_j: float, beta: float = 1.0) -> float:
    """
    computes likelihood of preference tj > ti, given their rewards
    """
    a = jnp.exp(return_i * beta)
    b = jnp.exp(return_j * beta)
    return b / (a + b)


def random_query_iterator(key, n: int, n_queries: int):
    """
    Note this does not return ti=tj, and migth return duplicates
    """
    for _ in range(n_queries):
        key, key1, key2 = jr.split(key, 3)
        ti = jr.randint(key=key1, shape=(), minval=0, maxval=n - 1)
        tj = jr.randint(key=key2, shape=(), minval=ti + 1, maxval=n)
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
    n_demos = len(ranked_returns)
    n_queries = n_queries if n_queries != -1 else math.comb(n_demos, 2)

    queries = []
    labels = []
    num_mislabels = 0

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

            key, subkey = jr.split(key)
            if jr.uniform(subkey) > prob:
                num_mislabels += 1
                label = 0

        # * irrationality: label flip mistake
        key, subkey = jr.split(key)
        label = 1 - label if (jr.uniform(subkey) < mistake_prob) else label

        queries.append((ti, tj))
        labels.append(label)

    queries_Q2 = jnp.array(queries).astype(jnp.int32)
    labels_Q1 = jnp.expand_dims(jnp.array(labels), 1).astype(jnp.int32)

    return queries_Q2, labels_Q1, num_mislabels


def create_pref_data_jit(
    key,
    ranked_returns: N,
    traj_obs: NTD = None,
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
    Jit-compatible version of create_pref_data. (thanks Cursor)
    Instead of using Python lists and conditionals, uses jax arrays and jnp.where.

    Args and outputs are the same as create_pref_data.
    """
    n_demos = len(ranked_returns)
    n_queries = n_queries if n_queries != -1 else math.comb(n_demos, 2)

    # Pre-allocate arrays
    queries = jnp.zeros((n_queries, 2), dtype=jnp.int32)
    labels = jnp.ones(n_queries, dtype=jnp.int32)
    num_mislabels = jnp.array(0)

    def body_fn(i, state):
        key, queries, labels, num_mislabels = state

        # Generate random indices
        key, key1, key2 = jr.split(key, 3)
        ti = jr.randint(key=key1, shape=(), minval=0, maxval=n_demos - 1)
        tj = jr.randint(key=key2, shape=(), minval=ti + 1, maxval=n_demos)

        # Delta check
        delta_mask = jnp.where(
            use_delta,
            jnp.where(
                delta_rank > 1,
                (tj - ti) >= delta_rank,
                (ranked_returns[tj] - ranked_returns[ti]) >= delta_reward,
            ),
            1,
        )

        # Skip if both are bad
        skip_bad = jnp.where(
            jnp.maximum(ranked_returns[tj], ranked_returns[ti]) < skip_threshold, 0, 1
        )

        # Noisy preferences
        prob = bt_likelihood(ranked_returns[ti], ranked_returns[tj], bt_beta)
        key, subkey = jr.split(key)
        noisy_label = jnp.where(
            noisy_prefs, jnp.where(jr.uniform(subkey) > prob, 0, 1), 1
        )

        # Mistake flips
        key, subkey = jr.split(key)
        mistake_flip = jnp.where(jr.uniform(subkey) < mistake_prob, 1, 0)

        # Update label
        label = jnp.where(mistake_flip, 0, 1) * noisy_label * delta_mask * skip_bad

        # Update arrays
        queries = queries.at[i].set(jnp.array([ti, tj]))
        labels = labels.at[i].set(label)
        num_mislabels += (1 - noisy_label) * noisy_prefs + mistake_flip

        return key, queries, labels, num_mislabels

    # Run the loop
    init_state = (key, queries, labels, num_mislabels)
    key, queries_Q2, labels, num_mislabels = jax.lax.fori_loop(
        0, n_queries, body_fn, init_state
    )

    labels_Q1 = jnp.expand_dims(labels, 1)

    if traj_obs is not None:
        queries_Q2TD = traj_obs[queries_Q2]  # index into (N, T, D) using (Q, 2)
        pref_data = QueryWithResponse(queries_Q2TD, labels_Q1)
        return pref_data, num_mislabels
    else:
        return queries_Q2, labels_Q1, num_mislabels


if __name__ == "__main__":
    """Test that create_pref_data and create_pref_data_jit produce the same output."""
    from bnn_pref.utils.test_functions import test_functions_dict
    from bnn_pref.utils.utils import get_gaussian_vector, get_random_seed

    # Set up random key and config
    seed = get_random_seed()
    key = jr.key(seed)

    # Create a simple config similar to what's used in make_synthetic_data
    cfg = {
        "data": {
            "n_feats": 2,
            "n_demos": 10,
            "length": 5,
            "nq_train": 5,
            "nq_test": 5,
        },
        "f": "linear",  # Using linear reward function for simplicity
    }

    n_feats = cfg["data"]["n_feats"]
    demo_len = cfg["data"]["length"]
    n_demos = cfg["data"]["n_demos"]
    nq_train = cfg["data"]["nq_train"]
    nq_test = cfg["data"]["nq_test"]

    # Generate synthetic data using make_synthetic_data
    key, param_key, demo_key = jr.split(key, 3)
    true_param_D = get_gaussian_vector(param_key, dim=n_feats, normalize=True)
    true_reward_fn = test_functions_dict[cfg["f"]]
    demos_NTD = jr.normal(demo_key, (n_demos, demo_len, n_feats))
    returns_N = true_reward_fn(demos_NTD, true_param_D)

    # sort by increasing reward
    sorted_idx = jnp.argsort(returns_N)
    returns_N = returns_N[sorted_idx]
    demos_NTD = demos_NTD[sorted_idx]

    # Generate preference data using both functions
    key, pref_key = jr.split(key, 2)
    queries1_Q2, labels1_Q1, mislabels1 = create_pref_data(
        pref_key,
        ranked_returns=returns_N,
        n_queries=nq_train,
    )

    queries2_Q2, labels2_Q1, mislabels2 = create_pref_data_jit(
        pref_key,
        ranked_returns=returns_N,
        n_queries=nq_train,
    )

    # Compare outputs
    assert queries1_Q2.shape == queries2_Q2.shape, "Query shapes should match"
    assert labels1_Q1.shape == labels2_Q1.shape, "Label shapes should match"

    queried_returns1_Q2 = returns_N[queries1_Q2]
    queried_returns2_Q2 = returns_N[queries2_Q2]

    assert jnp.all(queried_returns1_Q2[:, 0] < queried_returns1_Q2[:, 1])
    assert jnp.all(queried_returns2_Q2[:, 0] < queried_returns2_Q2[:, 1])
