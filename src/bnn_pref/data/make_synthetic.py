from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from bnn_pref.data.pref_utils import QueryWithResponse, create_pref_data_jit
from bnn_pref.utils.test_functions import test_functions_dict
from bnn_pref.utils.type import NTD, N
from bnn_pref.utils.utils import get_gaussian_vector


def make_synthetic_data(key, cfg) -> Tuple[NTD, N, QueryWithResponse]:
    data_kw = cfg["data"]
    task_kw = cfg["task"]
    n_feats = task_kw["n_feats"]
    demo_len = task_kw["length"]
    n_demos = data_kw["n_demos"]
    demo_train_frac = data_kw["demo_train_frac"]
    nq_train, nq_test = data_kw["nq_train"], data_kw["nq_test"]

    # * generate true params + trajectories
    key, key1, key2, key3, key4 = jr.split(key, 5)
    true_param_D = get_gaussian_vector(key1, dim=n_feats, normalize=True)
    true_reward_fn = test_functions_dict[task_kw["f"]]
    train_trajs, test_trajs = generate_synthetic_trajs(
        key2,
        traj_shape=(n_demos, demo_len, n_feats),
        true_param=true_param_D,
        true_reward_fn=true_reward_fn,
        train_frac=demo_train_frac,
    )

    train_prefs, _ = create_pref_data_jit(
        key3,
        ranked_returns=train_trajs["returns"],
        traj_obs=train_trajs["observations"],
        n_queries=nq_train,
    )

    test_prefs, _ = create_pref_data_jit(
        key4,
        ranked_returns=test_trajs["returns"],
        traj_obs=test_trajs["observations"],
        n_queries=nq_test,
    )

    output = {
        # true reward fn + params
        "true_param": true_param_D,
        "true_reward_fn": true_reward_fn,
        # train data
        "train_trajs": train_trajs,
        "train_prefs": train_prefs,
        # test data
        "test_trajs": test_trajs,
        "test_prefs": test_prefs,
    }
    return output


def generate_synthetic_trajs(
    key,
    traj_shape: Tuple[int, int, int],  # (N, T, D)
    true_param: N,
    true_reward_fn: Callable,
    train_frac: float = 0.8,
):
    # * generate trajectories
    obs_NTD = jr.normal(key, traj_shape)
    returns_N = true_reward_fn(obs_NTD, true_param)
    trajs = {"observations": obs_NTD, "returns": returns_N}

    # * split into train/test
    n_demos = len(returns_N)
    idxs = jnp.arange(n_demos)
    n_train = int(n_demos * train_frac)
    train_idxs, test_idxs = idxs[:n_train], idxs[n_train:]
    train_trajs = jax.tree.map(lambda x: x[train_idxs], trajs)
    test_trajs = jax.tree.map(lambda x: x[test_idxs], trajs)

    # * rank by return
    train_sorted_idxes = jnp.argsort(train_trajs["returns"])
    test_sorted_idxes = jnp.argsort(test_trajs["returns"])
    train_trajs = jax.tree.map(lambda x: x[train_sorted_idxes], train_trajs)
    test_trajs = jax.tree.map(lambda x: x[test_sorted_idxes], test_trajs)

    return train_trajs, test_trajs
