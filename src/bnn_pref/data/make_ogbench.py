import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import ogbench
from bnn_pref.data.pref_utils import QueryIndexAndResponses, create_pref_data
from bnn_pref.utils.type import ArrayDict
from bnn_pref.utils.utils import get_random_seed


def make_ogbench_data(key, cfg) -> ArrayDict:
    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]

    # * load data
    _, train_trajs, test_trajs = ogbench.make_env_and_datasets(
        task_cfg["name"],
        compact_dataset=False,
    )
    # * data normalization
    # train_trajs, val_trajs = standardize_traj(train_trajs, val_trajs)
    # train_trajs, val_trajs = minmax_scale_traj(train_trajs, val_trajs)

    # * separate trajs, filter by low return, sort by increasing return
    train_trajs = process_ogbench(train_trajs, rank=True)
    test_trajs = process_ogbench(test_trajs, rank=True)

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
        "train_trajs": train_trajs,
        "train_prefs": train_prefs,
        "test_trajs": test_trajs,
        "test_prefs": test_prefs,
    }

    return output


def process_ogbench(
    ds: ArrayDict,
    rank: bool = False,
) -> ArrayDict:
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
    ends = jnp.where(ds["terminals"])[0]
    bgns = jnp.concatenate([jnp.array([-1]), ends[:-1]])
    separator_fn = lambda x: jnp.array([x[s + 1 : e + 1] for s, e in zip(bgns, ends)])
    ds = jax.tree.map(separator_fn, ds)

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
