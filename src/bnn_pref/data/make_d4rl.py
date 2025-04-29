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
from einops import rearrange
from jaxtyping import Array, Float

from bnn_pref.data.pref_utils import QueryIndexAndResponses, create_pref_data
from bnn_pref.data.traj_utils import (
    normalize,
    rebalance,
    segment_traj_masked,
    split_dataset,
)
from bnn_pref.utils.type import ArrayDict


def segment_arraydict_masked(trajs: ArrayDict, sz: int) -> ArrayDict:
    """
    segment each traj in traj dict into chunks of size sz
    """
    assert sz > 0, f"segment size {sz=} must be positive"
    # print(f"  Whole trajs: {trajs['observations'].shape}")
    seg_fn = lambda x: segment_traj_masked(x, trajs["masks"], sz)
    trajs["observations"] = seg_fn(trajs["observations"])
    trajs["rewards"] = rearrange(seg_fn(trajs["rewards"]), "S sz 1-> S sz")
    trajs["returns"] = trajs["rewards"].sum(1)
    # print(
    #     f"  Segments:    {trajs['observations'].shape} (avg whole traj length={trajs['masks'].sum(1).mean()})"
    # )
    return trajs


def make_d4rl_data(key, cfg) -> ArrayDict:
    """
    d4rl returns dict with keys, where S = num_transitions
        observations: (S, O)
        actions: (S, A)
        next_observations: (S, O)
        rewards: (S,)
        terminals: (S,)
        timeouts: (S,)

    output:
        observations: (N, T, D)
        rewards: (N, T)
        masks: (N, T)
        returns: (N,)
    """
    task_cfg = cfg["task"]
    data_cfg = cfg["data"]
    demo_train_frac = data_cfg["demo_train_frac"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    sz = data_cfg["segment_size"]

    # * transitions -> trajs, pad to max length w/ masks, filter out short traj length
    ds = gym.make(task_cfg["name"]).get_dataset()
    trajs = process_d4rl_data(ds, min_traj_len=data_cfg["min_traj_len"])

    # * optionally normalize observations
    trajs.update({"observations": normalize(trajs["observations"], axis=(0, 1))})

    # * optional pruning
    key, key_rebalance = jr.split(key, 2)
    trajs = rebalance(
        key_rebalance,
        task_cfg["name"],
        ds=trajs,
        n_bins=data_cfg["n_bins"],
        max_count_per_bin=data_cfg["max_count_per_bin"],
        tokeep=data_cfg["tokeep"],
    )

    # * optionally segment trajectories: (N, T, ...) -> (N * n_chunks, sz, ...)
    # * can be done before or after splitting into train/test
    if sz != -1:
        trajs = segment_arraydict_masked(trajs, sz)

    # * split into train/test
    key, key_split = jr.split(key, 2)
    train_trajs, test_trajs = split_dataset(key_split, trajs, demo_train_frac)

    # if sz != -1:
    #     train_trajs = segment_arraydict_masked(train_trajs, sz)
    #     test_trajs = segment_arraydict_masked(test_trajs, sz)

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


def process_d4rl_data(
    ds: ArrayDict, rank: bool = False, min_traj_len: int = 50
) -> ArrayDict:
    """
    Convert d4rl dataset to ArrayDict, where trajs are padded to the max traj length
    found in the dataset.

    Inputs:
      observations: (S, O)
      actions: (S, A)
      next_observations: (S, O)
      rewards: (S,)
      terminals: (S,)
      timeouts: (S,)
    Outputs:
      observations: (N, T, D)
      rewards: (N, T)
      masks: (N, T)
      returns: (N,)
    """
    # * get traj boundaries via timeouts & terminals
    end_N = jnp.where(ds["timeouts"] | ds["terminals"])[0]
    bgn_N = jnp.concatenate([jnp.array([-1]), end_N[:-1]])
    length_N = end_N - bgn_N
    max_traj_len = jnp.max(length_N)

    # * optionally filter out trajectories shorter than required length
    if min_traj_len > 0:
        length_mask_N = length_N > min_traj_len
        bgn_N = bgn_N[length_mask_N]
        end_N = end_N[length_mask_N]
        length_N = length_N[length_mask_N]

    # * create valid mask
    valid_mask_NT = jnp.array(
        [
            jnp.pad(jnp.ones(traj_len), (0, max_traj_len - traj_len))
            for traj_len in length_N
        ]
    ).astype(jnp.bool)

    # * transitions -> trajs and pad to max traj length
    def pad_fn(
        x: Float[Array, "S ..."],
    ) -> Float[Array, "N T ..."]:
        def pad_traj(x, bgn: int, end: int):
            traj = x[bgn + 1 : end + 1]
            traj_len = traj.shape[0]
            pad_size = max_traj_len - traj_len
            pad_width = [(0, pad_size)] + [(0, 0)] * (traj.ndim - 1)
            return jnp.pad(traj, pad_width)

        return jnp.array([pad_traj(x, bgn, end) for bgn, end in zip(bgn_N, end_N)])

    output = {
        "observations": pad_fn(ds["observations"]),
        # "actions": pad_fn(ds["actions"]),
        "rewards": pad_fn(ds["rewards"]),
        "masks": valid_mask_NT,
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
