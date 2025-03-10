from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from bnn_pref.data.utils import QueryWithResponse, demos_to_pref_data
from bnn_pref.utils.test_functions import test_functions_dict
from bnn_pref.utils.type import NTD, N
from bnn_pref.utils.utils import get_gaussian_vector


def make_synthetic_data(key, cfg) -> Tuple[NTD, N, QueryWithResponse]:
    data_kw = cfg["data"]
    n_feats = data_kw["n_feats"]
    n_demos = data_kw["n_demos"]
    demo_len = data_kw["length"]
    train_frac = data_kw["train_frac"]
    n_queries = data_kw["n_queries"]

    # * generate true params + trajectories
    key, key1, key2, key3, key4 = jr.split(key, 5)
    true_param_D = get_gaussian_vector(key1, dim=n_feats, normalize=True)
    true_reward_fn = test_functions_dict[cfg["f"]]
    demos_NTD = jr.normal(key2, (n_demos, demo_len, n_feats))

    # * preprocess trajectories
    # demos_NTD /= jnp.linalg.norm(demos_NTD, axis=2, keepdims=True)
    # demos_NTD = demos_NTD - jnp.mean(demos_NTD, axis=(0, 1))
    # demos_NTD = demos_NTD / jnp.std(demos_NTD, axis=(0, 1))

    # * split into train/test demos, and generate preference data for each split
    n_train_demos = int(n_demos * train_frac)
    train_demos_NTD = demos_NTD[:n_train_demos]
    test_demos_NTD = demos_NTD[n_train_demos:]

    train_returns_N, train_pref_data = demos_to_pref_data(
        key3,
        demos=train_demos_NTD,
        returns_N=true_reward_fn(train_demos_NTD, true_param_D),
        n_queries=n_queries,
    )
    test_returns_N, test_pref_data = demos_to_pref_data(
        key4,
        demos=test_demos_NTD,
        returns_N=true_reward_fn(test_demos_NTD, true_param_D),
        n_queries=-1,
    )
    output = {
        # true reward fn + params
        "true_param": true_param_D,
        "true_reward_fn": true_reward_fn,
        # train data
        "train_demos": train_demos_NTD,
        "train_returns": train_returns_N,
        "train_prefs": train_pref_data,
        # test data
        "test_demos": test_demos_NTD,
        "test_returns": test_returns_N,
        "test_prefs": test_pref_data,
    }
    return output
