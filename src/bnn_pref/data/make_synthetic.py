from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from bnn_pref.data.pref_utils import QueryIndexAndResponses, create_pref_data
from bnn_pref.utils.test_functions import test_functions_dict
from bnn_pref.utils.type import ArrayDict, N
from bnn_pref.utils.utils import get_gaussian_vector


def make_synthetic_data(key, cfg) -> ArrayDict:
    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    n_feats = task_cfg["n_feats"]
    demo_len = task_cfg["length"]
    n_demos = data_cfg["n_demos"]
    demo_train_frac = data_cfg["demo_train_frac"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]

    # * generate true params + trajectories, split into train/test, rank by return
    key, key_param, key_traj = jr.split(key, 3)
    true_param_D = get_gaussian_vector(key_param, dim=n_feats, normalize=True)
    true_reward_fn = test_functions_dict[task_cfg["f"]]
    train_trajs, test_trajs = generate_synthetic_trajs(
        key_traj,
        traj_shape=(n_demos, demo_len, n_feats),
        true_param=true_param_D,
        true_reward_fn=true_reward_fn,
        train_frac=demo_train_frac,
    )

    # * create preference data
    key, key_train, key_test = jr.split(key, 3)
    train_prefs: QueryIndexAndResponses = create_pref_data(
        key_train,
        ranked_returns=train_trajs["returns"],
        n_queries=nq_train,
        noisy_label=data_cfg["noisy_label"],
        bt_beta=data_cfg["bt_beta"],
    )

    test_prefs: QueryIndexAndResponses = create_pref_data(
        key_test,
        ranked_returns=test_trajs["returns"],
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
