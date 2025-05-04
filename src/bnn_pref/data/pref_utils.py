import itertools as it
import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int

from bnn_pref.utils.type import ArrayDict, unpackable_dataclass


@unpackable_dataclass
class QueryFeaturesAndResponses:
    queries_Q2TD: Float[Array, "n_queries 2 n_steps n_feats"]
    responses_Q1: Int[Array, "n_queries 1"]
    n_mislabels: int


@unpackable_dataclass
class QueryIndexAndResponses:
    queries_Q2: Int[Array, "n_queries 2"]
    responses_Q1: Int[Array, "n_queries 1"]
    n_mislabels: int


def query_indices_to_features(
    queries: QueryIndexAndResponses,
    trajs: ArrayDict,
) -> QueryFeaturesAndResponses:
    """
    Convert a QueryIndexAndResponses object to a QueryFeaturesAndResponses object.
    """
    traj_obs = trajs["observations"]
    features_Q2TD = traj_obs[queries.queries_Q2]
    return QueryFeaturesAndResponses(
        features_Q2TD,
        queries.responses_Q1,
        queries.n_mislabels,
    )


class BradleyTerry:
    @staticmethod
    def logpdf(
        params_D: Float[Array, "n_feats"],
        data: QueryFeaturesAndResponses,
        reward_fn: Callable,
        beta: float = 1.0,  # rationality constant
    ) -> Float[Array, "n_queries 1"]:
        features_Q2TD, responses_Q1 = data.queries_Q2TD, data.responses_Q1
        returns_Q2 = beta * reward_fn(features_Q2TD, params_D)
        returns_Q1 = jnp.take_along_axis(returns_Q2, responses_Q1, axis=1)
        return returns_Q1 - jax.nn.logsumexp(returns_Q2, axis=1, keepdims=True)

    @staticmethod
    def potential(
        params_D: Float[Array, "n_feats"],
        reward_fn: Callable,
        data: QueryFeaturesAndResponses,
    ) -> float:
        ll_Q1 = BradleyTerry.logpdf(
            params_D=params_D,
            data=data,
            reward_fn=reward_fn,
        )
        # prior = # just uniform log 1
        joint_ll = ll_Q1.sum()
        return joint_ll


def bt_likelihood(r1: float, r2: float, beta: float = 1.0) -> float:
    """
    computes Bernoulli likelihood of the preference relation t2 > t1, given their rewards
    """

    # a = jnp.exp(return_i * beta)
    # b = jnp.exp(return_j * beta)
    # return b / (a + b)

    logits = jnp.asarray([r1 * beta, r2 * beta])
    return jnp.exp(jax.nn.log_softmax(logits))[1]


def random_query_iter_sample(key, n_trajs: int, n_queries: int):
    """
    Note this does not return ti=tj, and might return duplicates
    """
    for _ in range(n_queries):
        key, key1, key2 = jr.split(key, 3)
        ti = jr.randint(key=key1, shape=(), minval=0, maxval=n_trajs - 1)
        tj = jr.randint(key=key2, shape=(), minval=ti + 1, maxval=n_trajs)
        yield ti, tj


def random_query_iter_perm(key, n_trajs: int, n_queries: int):
    _, key_perm = jr.split(key)

    queries_gen = it.combinations(range(n_trajs), 2)
    queries = jnp.asarray(list(queries_gen))  # ((n choose 2), 2)
    queries = jr.permutation(key_perm, queries)
    queries = queries[:n_queries]  # (n_queries, 2)

    for query in queries:
        yield query


def create_pref_data(
    key,
    ranked_returns: Float[Array, "n_trajs"],
    n_queries: int = -1,
    skip_threshold: float = -jnp.inf,  # skip if both are bad
    use_delta: bool = False,  # skip by delta_rank or delta_reward
    delta_rank: int = 1,
    delta_reward: float = 0,
    noisy_label: bool = False,  # if False, equivalent to beta=inf in BT Likelihood
    bt_beta: float = 1.0,
    mistake_prob: float = 0.0,  # label flip mistake (not rly used)
) -> QueryIndexAndResponses:
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
    if noisy_label:
        # this allows for bt_beta=15 to get 5-10% error rate across all tasks
        mean = jnp.mean(ranked_returns)
        std = jnp.std(ranked_returns)
        ranked_returns = (ranked_returns - mean) / std

    n_demos = len(ranked_returns)
    n_queries = n_queries if n_queries != -1 else math.comb(n_demos, 2)

    queries = []
    labels = []
    n_mislabels = 0

    iterator = random_query_iter_perm if n_demos <= 1000 else random_query_iter_sample
    for ti, tj in iterator(key, n_demos, n_queries):
        label = 1  # label=1 means tj > ti

        # * skip if both are bad
        if max(ranked_returns[ti], ranked_returns[tj]) < skip_threshold:
            continue

        # * skip if tj is not better than ti by delta_rank or delta_reward
        if use_delta:
            if delta_rank > 1:
                if (tj - ti) < delta_rank:
                    continue
            else:
                if (ranked_returns[tj] - ranked_returns[ti]) < delta_reward:
                    continue

        # * noisily rational prefs (finite beta in the BT Likelihood)
        if noisy_label:
            prob = bt_likelihood(ranked_returns[ti], ranked_returns[tj], bt_beta)

            key, subkey = jr.split(key)
            if jr.uniform(subkey) > prob:
                n_mislabels += 1
                label = 0

        # * irrationality: label flip mistake
        key, subkey = jr.split(key)
        prob = jr.uniform(subkey)
        label = 1 - label if (prob < mistake_prob) else label

        queries.append((ti, tj))
        labels.append(label)

    queries_Q2 = jnp.asarray(queries).astype(jnp.int32)
    labels_Q1 = jnp.expand_dims(jnp.asarray(labels), 1).astype(jnp.int32)

    return QueryIndexAndResponses(queries_Q2, labels_Q1, n_mislabels)


if __name__ == "__main__":
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

    # Compare outputs
    queried_returns1_Q2 = returns_N[queries1_Q2]
    assert jnp.all(queried_returns1_Q2[:, 0] < queried_returns1_Q2[:, 1])
