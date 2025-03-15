from typing import Dict

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
from tensordict import TensorDict

from bnn_pref.data.pref_utils import create_pref_data_jit
from bnn_pref.utils.utils import get_random_seed


def make_prefcc_data(key, cfg) -> Dict:
    task_cfg = cfg["task"]
    data_cfg = cfg["data"]
    path = task_cfg["tensordict_path"]
    n_queries = data_cfg["n_queries"]
    demo_train_frac = data_cfg["demo_train_frac"]
    n_train_queries = int(n_queries * data_cfg["train_frac"])
    n_test_queries = n_queries - n_train_queries

    td = torch.load(path, weights_only=False)
    ds = process_prefcc_data(td)

    key, key_split, key_train, key_test = jr.split(key, 4)
    train_trajs, test_trajs = split_dataset(key_split, ds, demo_train_frac)

    train_prefs, _ = create_pref_data_jit(
        key_train,
        ranked_returns=train_trajs["returns"],
        traj_obs=train_trajs["observations"],
        n_queries=n_train_queries,
    )
    test_prefs, _ = create_pref_data_jit(
        key_test,
        ranked_returns=test_trajs["returns"],
        traj_obs=test_trajs["observations"],
        n_queries=n_test_queries,
    )

    return {
        "train_trajs": train_trajs,
        "train_prefs": train_prefs,
        "test_trajs": test_trajs,
        "test_prefs": test_prefs,
    }


def process_prefcc_data(
    td: TensorDict,
    rank: bool = False,
) -> Dict[str, jnp.ndarray]:
    """
    Tensordict only contains obs, act, rew, and are already sorted by returns.
    """
    ds = {
        "observations": jnp.array(td["obs"]),
        "actions": jnp.array(td["actions"]),
        "rewards": jnp.array(td["rewards"]),
        "returns": jnp.array(td["rewards"]).sum(axis=-1),
    }

    # * sort trajectories by return (ascending)
    if rank:
        sorted_idxes = jnp.argsort(ds["returns"])
        ds = jax.tree.map(lambda x: x[sorted_idxes], ds)

    return ds


def split_dataset(key, ds, train_frac=0.8):
    n = len(jax.tree_util.tree_leaves(ds)[0])  # Get length from first array
    idxs = jr.permutation(key, n)
    n_train = int(n * train_frac)
    train_idxs, test_idxs = idxs[:n_train], idxs[n_train:]
    train_ds = jax.tree.map(lambda x: x[train_idxs], ds)
    test_ds = jax.tree.map(lambda x: x[test_idxs], ds)

    # sort by return (ascending)
    train_sorted_idxes = jnp.argsort(train_ds["returns"])
    test_sorted_idxes = jnp.argsort(test_ds["returns"])
    train_ds = jax.tree.map(lambda x: x[train_sorted_idxes], train_ds)
    test_ds = jax.tree.map(lambda x: x[test_sorted_idxes], test_ds)

    return train_ds, test_ds


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../../cfg"):
        cfg = compose(config_name="config", overrides=["data=lunar"])

    key = jr.key(get_random_seed())
    output = make_prefcc_data(key, cfg)
    train_data, test_data = output["train_prefs"], output["test_prefs"]
    print(train_data.queries_Q2TD.shape, train_data.responses_Q1.shape)
    print(test_data.queries_Q2TD.shape, test_data.responses_Q1.shape)
    print()

    demos_NTD = output["train_trajs"]["observations"]
    demos_NTD = demos_NTD - jnp.mean(demos_NTD, axis=(0, 1))
    demos_NTD = demos_NTD / jnp.std(demos_NTD, axis=(0, 1))
    demos2 = jax.nn.standardize(demos_NTD, axis=(0, 1))
