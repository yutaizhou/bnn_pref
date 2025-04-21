from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
from tensordict import TensorDict

from bnn_pref.data.pref_utils import QueryData, create_pref_data
from bnn_pref.data.traj_utils import rebalance
from bnn_pref.utils.type import ArrayDict
from bnn_pref.utils.utils import get_random_seed


def make_prefcc_data(key, cfg) -> ArrayDict:
    task_cfg = cfg["task"]
    data_cfg = cfg["data"]
    path = task_cfg["tensordict_path"]
    demo_train_frac = data_cfg["demo_train_frac"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]

    # * load trajectory data, sort by return
    td = torch.load(path, weights_only=False)
    ds = process_prefcc_data(td)

    # * optionally normalize observations
    if task_cfg["name"] not in [
        "Reacher-v4",
    ]:
        obs = ds["observations"]  # (N, T, D)
        mean = jnp.mean(obs, axis=(0, 1), keepdims=True)
        std = jnp.std(obs, axis=(0, 1), keepdims=True)
        obs = (obs - mean) / std
        ds.update({"observations": obs})

    # * optional pruning
    ds = rebalance(
        key,
        task_cfg["name"],
        ds=ds,
        n_bins=data_cfg["n_bins"],
        max_count_per_bin=data_cfg["max_count_per_bin"],
        tokeep=data_cfg["tokeep"],
    )

    # * split into train/test
    key, key_split = jr.split(key, 2)
    train_trajs, test_trajs = split_dataset(key_split, ds, demo_train_frac)

    # * turn train/test trajs into preference data
    key, key_train, key_test = jr.split(key_split, 3)
    train_prefs: QueryData = create_pref_data(
        key_train,
        ranked_returns=train_trajs["returns"],
        n_queries=nq_train,
        noisy_label=data_cfg["noisy_label"],
        bt_beta=data_cfg["bt_beta"],
    )
    test_prefs: QueryData = create_pref_data(
        key_test,
        ranked_returns=test_trajs["returns"],
        n_queries=nq_test,
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
) -> ArrayDict:
    """
    Tensordict only contains obs, act, rew, and are already sorted by returns.
    """
    rewards = jnp.asarray(td["rewards"])  # (N, T)
    returns = rewards.sum(axis=-1)  # (N,)
    ds = {
        "observations": jnp.asarray(td["obs"]),
        # "actions": jnp.asarray(td["actions"]),
        "rewards": rewards,
        "returns": returns,
    }

    # * sort trajectories by return (ascending)
    if rank:
        sorted_idxes = jnp.argsort(ds["returns"])
        ds = jax.tree.map(lambda x: x[sorted_idxes], ds)

    return ds


def split_dataset(
    key,
    ds: ArrayDict,
    train_frac=0.8,
) -> Tuple[ArrayDict, ArrayDict]:
    """
    take (optionally ranked) ds, split into train/test, each sorted by return (ascending)

    ds = {
        "observations": (N, T, D),
        "actions": (N, T, A),
        "rewards": (N, T),
        "returns": (N,),
    }
    """
    n = len(ds["returns"])
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
