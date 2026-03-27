import os
import re
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict

import h5py
import ipdb
import numpy as np
from tqdm import tqdm

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
    _sort_by_return,
    _subsample,
    normalize,
    scale_image,
    segment_arraydict_masked,
    split_subsample_rank_ds,
)
from bnn_pref.utils.type import ArrayDict


def load_unirlhf_data(pref_dir: Path) -> ArrayDict:
    """
    Load unirlhf data from directory.
    Outputs:
        inds: (Q, 2) query indices: where in d4rl dataset the query items are located
        y: (Q,) preference labels
    """
    dirp = Path(pref_dir)
    y_fp = [p for p in dirp.iterdir() if p.is_file() and "human_label" in p.name][0]
    y_Q = np.load(y_fp, allow_pickle=True)  # (Q,)

    inds1_fp = [p for p in dirp.iterdir() if p.is_file() and "indices_1" in p.name][0]
    inds1 = np.load(inds1_fp, allow_pickle=True)  # (Q,)
    inds2_fp = [p for p in dirp.iterdir() if p.is_file() and "indices_2" in p.name][0]
    inds2 = np.load(inds2_fp, allow_pickle=True)  # (Q,)
    inds_Q2 = np.stack([inds1, inds2], axis=-1)  # (Q, 2)

    return inds_Q2, y_Q


def unirlhf_converter(d4rl_ds, queries, labels, sz: int):
    """
    convert unirlhf (d4rl) dataset to format for PreferenceEnv

    Args:
        d4rl_ds: (S, D) how d4rl loads the dataset by default
        queries: (Q, 2) query indices, indicates starting index of the query items in the d4rl dataset
        labels: (Q,) preference labels
            equality pref queries are removed
            0: prefer item 1
            1: prefer item 2
            -1: equality pref
        sz: int, how many samples to segment the dataset into
    """
    # remove equality pref queries
    mask = labels != -1
    queries_bgn_Q2 = queries[mask]  # (Q', 2)
    labels = labels[mask]  # (Q',)

    # construct item mapping
    bgns = jnp.unique(queries_bgn_Q2)  # (N,) unique item indices
    q2b = {q: b.item() for q, b in enumerate(bgns)}  # (0, ..., N-1) -> (bgn_0...)
    b2q = {b: q for q, b in q2b.items()}  # (bgn_0...) -> (0, ..., N-1)

    item_inds = jnp.array([jnp.arange(b, b + sz) for b in bgns])  # (N, T)
    trajs = {
        "observations": d4rl_ds["observations"][item_inds],  # (N, T, D)
        "rewards": d4rl_ds["rewards"][item_inds],  # (N, T)
        "returns": d4rl_ds["rewards"][item_inds].sum(axis=1),  # (N,)
    }

    query_Q2 = jnp.array([[b2q[b] for b in bgns_2] for bgns_2 in queries_bgn_Q2])
    labels_Q1 = jnp.expand_dims(labels, 1)

    return trajs, query_Q2, labels_Q1


def make_unirlhf_d4rl_data(key, cfg) -> ArrayDict:
    """
    d4rl returns dict with keys, where S = num_transitions
        observations: (S, O) or (S, H, W, C)
        actions: (S, A)
        next_observations: (S, O)
        rewards: (S,)
        terminals: (S,)
        timeouts: (S,)

    output:
        train_trajs / test_trajs:
            observations: (N, T, D)
            rewards: (N, T)
            returns: (N,)
        train_prefs / test_prefs:
            queries_Q2: (Q, 2)
            responses_Q1: (Q, 1)
    """
    task_cfg = cfg["task"]
    data_cfg = cfg["data"]

    # load unirlhf data, convert into our format
    pref_dir = task_cfg["dataset_dir"]
    queries_bgn_Q2, labels = load_unirlhf_data(pref_dir)
    ds = gym.make(task_cfg["name"]).get_dataset()
    sz = task_cfg["segment_size"]
    trajs, queries_Q2, labels_Q1 = unirlhf_converter(ds, queries_bgn_Q2, labels, sz)

    # split into train/test
    nq = len(queries_Q2)
    nq_train = int(nq * data_cfg["nq_train_frac"])
    train_trajs, test_trajs = trajs, deepcopy(trajs)

    key, key1 = jr.split(key, 2)
    inds = jr.permutation(key1, jnp.arange(nq))
    queries_Q2, labels_Q1 = queries_Q2[inds], labels_Q1[inds]

    train_queries, train_labels = queries_Q2[:nq_train], labels_Q1[:nq_train]
    test_queries, test_labels = queries_Q2[nq_train:], labels_Q1[nq_train:]
    train_prefs = QueryIndexAndResponses(train_queries, train_labels, 0)
    test_prefs = QueryIndexAndResponses(test_queries, test_labels, 0)

    # * normalize observations
    mean = jnp.mean(ds["observations"], axis=0).reshape(1, 1, -1)
    std = jnp.std(ds["observations"], axis=0).reshape(1, 1, -1)
    stats = (mean, std)
    train_trajs.update(
        {"observations": normalize(train_trajs["observations"], stats=stats)}
    )
    test_trajs.update(
        {"observations": normalize(test_trajs["observations"], stats=stats)}
    )
    # elif ds_type == "vd4rl":
    #     train_trajs.update({"observations": scale_image(train_trajs["observations"])})
    #     test_trajs.update({"observations": scale_image(test_trajs["observations"])})

    return {
        "train_trajs": train_trajs,
        "train_prefs": train_prefs,
        "test_trajs": test_trajs,
        "test_prefs": test_prefs,
    }


def make_d4rl_data(key, cfg) -> ArrayDict:
    """
    d4rl returns dict with keys, where S = num_transitions
        observations: (S, O) or (S, H, W, C)
        actions: (S, A)
        next_observations: (S, O)
        rewards: (S,)
        terminals: (S,)
        timeouts: (S,)

    output:
        train_trajs / test_trajs:
            observations: (N, T, D)
            rewards: (N, T)
            returns: (N,)
        train_prefs / test_prefs:
            queries_Q2: (Q, 2)
            responses_Q1: (Q, 1)
    """
    task_cfg = cfg["task"]
    data_cfg = cfg["data"]

    ds_type = task_cfg["ds_type"]
    demo_train_frac = data_cfg["demo_train_frac"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]

    ns_train, ns_test = data_cfg["n_segments_train"], data_cfg["n_segments_test"]

    # * d4rl transitions -> trajs, pad to max length w/ masks, filter out short traj length
    if ds_type == "d4rl":
        ds = gym.make(task_cfg["name"]).get_dataset()
        trajs = process_d4rl_data(ds, min_traj_len=data_cfg["min_traj_len"])
        sz = data_cfg["segment_size"]

    elif ds_type == "vd4rl":
        ds_dir = task_cfg["dataset_dir"]
        trajs = process_vd4rl_data(
            Path(ds_dir),
            min_traj_len=data_cfg["min_traj_len"],
            vd4rl_64=data_cfg["vd4rl_64"],
        )
        sz = data_cfg["vd4rl_segment_size"]
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

    # * prune trajectories based on return distribution
    key, key_subsample = jr.split(key, 2)
    trajs = _subsample(key_subsample, trajs, data_cfg["tokeep"])
    trajs = _sort_by_return(trajs)

    # * segment trajectories: (N, T, ...) -> (N * n_chunks, sz, ...)
    # example obs: (150, 501, HWC), 0.86G ->  (7650, 10, HWC), 0.87G
    if sz != -1:
        trajs = segment_arraydict_masked(trajs, sz)
        del trajs["masks"]

    # * split segments into train/test, each sorted by ascending return
    key, key_split = jr.split(key, 2)
    train_trajs, test_trajs = split_subsample_rank_ds(
        key_split, trajs, demo_train_frac, ns_train, ns_test
    )

    # * normalize observations
    if ds_type == "d4rl":
        mean = jnp.mean(ds["observations"], axis=0).reshape(1, 1, -1)
        std = jnp.std(ds["observations"], axis=0).reshape(1, 1, -1)
        stats = (mean, std)
        train_trajs.update(
            {"observations": normalize(train_trajs["observations"], stats=stats)}
        )
        test_trajs.update(
            {"observations": normalize(test_trajs["observations"], stats=stats)}
        )
    elif ds_type == "vd4rl":
        train_trajs.update({"observations": scale_image(train_trajs["observations"])})
        test_trajs.update({"observations": scale_image(test_trajs["observations"])})

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

    S: number of transitions
    N: number of trajectories
    T: max traj length

    Inputs:
      observations: (S, O)
      actions: (S, A)
      next_observations: (S, O)
      rewards: (S,)
      terminals: (S,)
      timeouts: (S,)
    Outputs:
      observations: (N, T, O)
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
            jnp.pad(
                array=jnp.ones(traj_len, dtype=jnp.bool_),
                pad_width=(0, max_traj_len - traj_len),
            )
            for traj_len in length_N
        ],
        dtype=jnp.bool_,
    )

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


def process_vd4rl_data(
    offline_dir: Path, min_traj_len: int = 50, vd4rl_64: bool = True
):
    def converter(ds: Dict):
        """
        Convert vd4rl dictionary to d4rl dictionary.

        Input vd4rl dictionary:
        - observation: (S, C, H, W), uint8
        - action: (S, A), float32
        - reward: (S,) float32
        - discount: (S,) float32
        - step_type: (S,) int32

        Output d4rl dictionary:
        - observations: (S, H, W, C), uint8
        - actions: (S, A), float32
        - rewards: (S,) float32
        - terminals: (S,) bool
        - timeouts: (S,) bool
        """

        # vd4rl only uses terminals. jax CNN uses (H, W, C)
        obs = rearrange(ds["observation"], "S C H W-> S H W C")
        if vd4rl_64:
            shape = (len(obs), 64, 64, 3)
            obs = jax.image.resize(obs, shape, "lanczos3").astype(jnp.uint8)
        timeouts = jnp.zeros_like(ds["step_type"], dtype=jnp.bool_)

        out = {
            "observations": obs,
            "actions": ds["action"],
            "rewards": ds["reward"],
            "terminals": ds["step_type"] == 2,
            "timeouts": timeouts,
        }
        return out

    filenames = sorted(offline_dir.glob("*.hdf5"))
    num_steps = 0
    traj_list = []
    for i, filename in tqdm(
        enumerate(filenames),
        desc="Loading offline dataset into replay buffer",
        unit="file",
    ):
        try:
            episodes = h5py.File(filename, "r")
            episodes = {k: jnp.array(episodes[k][:]) for k in episodes.keys()}
            trajs = process_d4rl_data(converter(episodes), min_traj_len=min_traj_len)
            traj_list.append(trajs)
            length = episodes["reward"].shape[0]
            num_steps += length
        except Exception as e:
            print(f"Could not load episode {str(filename)}: {e}")
            continue
        # if num_steps >= replay_buffer_size:
        #     break
        # if i == 2:
        #     print("early break!")
        #     break
    print(f"Finished, loaded {num_steps} offline timesteps.")
    trajs = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *traj_list)
    return trajs


if __name__ == "__main__":
    from hydra import compose, initialize

    from bnn_pref.utils.utils import get_random_seed

    with initialize(version_base=None, config_path="../../cfg"):
        cfg = compose(config_name="config", overrides=["task=cheetahMediumReplay"])
    key = jr.key(get_random_seed())
    data = make_d4rl_data(key, cfg)
    train_trajs, test_trajs = data["train_trajs"], data["test_trajs"]
    train_prefs, test_prefs = data["train_prefs"], data["test_prefs"]
    print(f"{train_trajs['observations'].shape=}")
    print(f"{test_trajs['observations'].shape=}")
    print(f"{train_prefs.queries_Q2.shape=}")
    print(f"{test_prefs.queries_Q2.shape=}")

    with initialize(version_base=None, config_path="../../cfg"):
        cfg = compose(config_name="config", overrides=["task=vcheetahMediumExpert"])

    key = jr.key(get_random_seed())
    data = make_d4rl_data(key, cfg)
    train_trajs, test_trajs = data["train_trajs"], data["test_trajs"]
    train_prefs, test_prefs = data["train_prefs"], data["test_prefs"]
    print(f"{train_trajs['observations'].shape=}")
    print(f"{test_trajs['observations'].shape=}")
    print(f"{train_prefs.queries_Q2.shape=}")
    print(f"{test_prefs.queries_Q2.shape=}")
