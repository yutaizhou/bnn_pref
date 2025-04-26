import os
import warnings

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import d4rl
import gym
import jax
import jax.numpy as jnp
import jax.random as jr

from bnn_pref.data.pref_utils import QueryIndexAndResponses, create_pref_data
from bnn_pref.data.traj_utils import (
    normalize_NTD,
    rebalance,
    segment_traj,
    split_dataset,
)
from bnn_pref.utils.type import ArrayDict


def make_d4rl_data(key, cfg) -> ArrayDict:
    """
    d4rl returns dict with keys:
        "observations": (N, T, D)
        "actions": (N, T, A)
        "next_observations": (N, T, D)
        "rewards": (N, T)
        "terminals": (N, T)
    """
    task_cfg = cfg["task"]
    data_cfg = cfg["data"]
    demo_train_frac = data_cfg["demo_train_frac"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]

    ds = gym.make(task_cfg["name"]).get_dataset()
    ds = process_d4rl_data(ds)

    # * optionally normalize observations
    ds.update({"observations": normalize_NTD(ds["observations"])})

    # * optional pruning
    key, key_rebalance = jr.split(key, 2)
    ds = rebalance(
        key_rebalance,
        task_cfg["name"],
        ds=ds,
        n_bins=data_cfg["n_bins"],
        max_count_per_bin=data_cfg["max_count_per_bin"],
        tokeep=data_cfg["tokeep"],
    )

    if data_cfg["segment_size"] != -1:
        ds = jax.tree.map(lambda x: segment_traj(x, data_cfg["segment_size"]), ds)

    # * split into train/test
    key, key_split = jr.split(key, 2)
    train_trajs, test_trajs = split_dataset(key_split, ds, demo_train_frac)

    # * turn train/test trajs into preference data
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

    return {
        "train_trajs": train_trajs,
        "train_prefs": train_prefs,
        "test_trajs": test_trajs,
        "test_prefs": test_prefs,
    }


def process_d4rl_data(ds: ArrayDict, rank: bool = False) -> ArrayDict:
    """
    Convert d4rl dataset to ArrayDict
    Inputs
      observations: (n_samples, observation_dim)
      actions: (n_samples, action_dim)
      next_observations: (n_samples, observation_dim)
      rewards: (n_samples,)
      terminals: (n_samples,)
      timeouts: (n_samples,)
    """
    # * seperate trajectories via timeouts field
    ends = jnp.where(ds["timeouts"] | ds["terminals"])[0]
    bgns = jnp.concatenate([jnp.array([-1]), ends[:-1]])
    separator_fn = lambda x: jnp.array([x[s + 1 : e + 1] for s, e in zip(bgns, ends)])
    ds = jax.tree.map(separator_fn, ds)

    output = {
        "observations": jnp.asarray(ds["observations"]),
        # "actions": jnp.asarray(ds["actions"]),
        "rewards": jnp.asarray(ds["rewards"]),
    }
    output["returns"] = output["rewards"].sum(axis=1)
    if rank:
        sorted_idxes = jnp.argsort(output["returns"])
        output = jax.tree.map(lambda x: x[sorted_idxes], output)
    return output


if __name__ == "__main__":
    from hydra import compose, initialize

    from bnn_pref.utils.utils import get_random_seed

    with initialize(version_base=None, config_path="../../cfg"):
        cfg = compose(config_name="config", overrides=["task=cheetah_medexp"])

    key = jr.key(get_random_seed())
    data = make_d4rl_data(key, cfg)
    train_trajs, test_trajs = data["train_trajs"], data["test_trajs"]
    train_prefs, test_prefs = data["train_prefs"], data["test_prefs"]
    import ipdb

    ipdb.set_trace()
