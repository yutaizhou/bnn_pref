from typing import Dict

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import ogbench
from bnn_pref.data.pref_utils import create_pref_data_jit
from bnn_pref.utils.utils import get_random_seed


def make_ogbench_data(key, cfg):
    data_kw = cfg["data"]
    task_kw = cfg["task"]
    n_queries = data_kw["n_queries"]
    train_frac = data_kw["train_frac"]
    n_train_queries = int(n_queries * train_frac)
    n_test_queries = n_queries - n_train_queries

    # * load data
    _, train_trajs, test_trajs = ogbench.make_env_and_datasets(
        task_kw["name"],
        compact_dataset=False,
    )
    # * data normalization
    # train_trajs, val_trajs = standardize_traj(train_trajs, val_trajs)
    # train_trajs, val_trajs = minmax_scale_traj(train_trajs, val_trajs)

    # * separate trajs, filter by low return, sort by increasing return
    train_trajs = process_ogbench(train_trajs, rank=True)
    test_trajs = process_ogbench(test_trajs, rank=True)

    # * create preference data
    key, key1, key2 = jr.split(key, 3)
    train_prefs, _ = create_pref_data_jit(
        key1,
        ranked_returns=train_trajs["returns"],
        traj_obs=train_trajs["observations"],
        n_queries=n_train_queries,
    )

    test_prefs, _ = create_pref_data_jit(
        key2,
        ranked_returns=test_trajs["returns"],
        traj_obs=test_trajs["observations"],
        n_queries=n_test_queries,
    )

    output = {
        "train_trajs": train_trajs,
        "train_prefs": train_prefs,
        "test_trajs": test_trajs,
        "test_prefs": test_prefs,
    }

    return output


def process_ogbench(
    ds: Dict[str, np.ndarray],
    rank: bool = False,
    thin: int = 1,
) -> Dict[str, jnp.ndarray]:
    """
    observations (5000000, 29) -> (10000, 500, 29)
    actions (5000000, 8) -> (10000, 500, 8)
    terminals (5000000,) -> (10000, 500)
    next_observations (5000000, 29) -> (10000, 500, 29)
    rewards (5000000,) -> (10000, 500)
    masks (5000000,) -> (10000, 500)
    returns (5000000,) -> (10000,)

    Number of trajectories: 10000
    """
    # * seperate trajectories via terminals field
    starts = jnp.where(ds["terminals"])[0]
    ends = jnp.concatenate([jnp.array([-1]), starts[:-1]])
    separator_fn = lambda x: jnp.array([x[s + 1 : e + 1] for s, e in zip(ends, starts)])
    ds = jax.tree.map(separator_fn, ds)

    # * thin out trajectories?
    ds = jax.tree.map(lambda x: x[:, ::thin], ds)

    # * sum rewards to get returns
    ds["returns"] = ds["rewards"].sum(axis=-1)
    # ds = {k: ds[k] for k in ["observations", "actions", "returns"]}

    # * filter out low return trajectories
    n_traj, traj_len, _ = ds["observations"].shape
    ds = jax.tree.map(lambda x: x[ds["returns"] > -traj_len], ds)

    # * sort trajectories by return (ascending)
    if rank:
        sorted_idxes = jnp.argsort(ds["returns"])
        ds = jax.tree.map(lambda x: x[sorted_idxes], ds)
    return ds


def standardize_traj(train_trajs, val_trajs):
    train_obs = train_trajs["observations"]
    mean = jnp.mean(train_obs, axis=(0, 1), keepdims=True)
    std = jnp.std(train_obs, axis=(0, 1), keepdims=True)

    train_trajs["observations"] = (train_trajs["observations"] - mean) / std
    val_trajs["observations"] = (val_trajs["observations"] - mean) / std
    return train_trajs, val_trajs


def minmax_scale_traj(train_trajs, val_trajs):
    train_obs = train_trajs["observations"]
    min_val = jnp.min(train_obs, axis=(0, 1), keepdims=True)
    max_val = jnp.max(train_obs, axis=(0, 1), keepdims=True)
    range = max_val - min_val

    train_trajs["observations"] = (train_trajs["observations"] - min_val) / range
    val_trajs["observations"] = (val_trajs["observations"] - min_val) / range
    return train_trajs, val_trajs


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../../cfg"):
        cfg = compose(config_name="config", overrides=["data=ogbench"])

    key = jr.key(get_random_seed())
    output = make_ogbench_data(key, cfg)
    train_data, test_data = output["train_prefs"], output["test_prefs"]
    print(train_data.queries_Q2TD.shape, train_data.responses_Q1.shape)
    print(test_data.queries_Q2TD.shape, test_data.responses_Q1.shape)
    print()

    demos_NTD = output["train_trajs"]["observations"]
    demos_NTD = demos_NTD - jnp.mean(demos_NTD, axis=(0, 1))
    demos_NTD = demos_NTD / jnp.std(demos_NTD, axis=(0, 1))
    demos2 = jax.nn.standardize(demos_NTD, axis=(0, 1))
