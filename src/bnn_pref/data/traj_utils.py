from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float

from bnn_pref.utils.type import ArrayDict

tasks_to_rebalance = [
    "LunarLander-v2",
    "HalfCheetah-v4",
    "acrobot-swingup-v0",
    "ball_in_cup-catch-v0",
    "cartpole-swingup-v0",
    "cheetah-run-v0",
    "hopper-hop-v0",
    "pendulum-swingup-v0",
    "reacher-easy-v0",
    "reacher-hard-v0",
    "walker-walk-v0",
]


def split_dataset(
    key,
    ds: ArrayDict,
    train_frac: float = 0.8,
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


def normalize_NTD(x: Float[Array, "N T D"]) -> Float[Array, "N T D"]:
    mean = jnp.mean(x, axis=(0, 1), keepdims=True)
    std = jnp.std(x, axis=(0, 1), keepdims=True)
    return (x - mean) / std


def segment_traj(traj: Float[Array, "N T ..."], seg_size: int = 50):
    """
    traj: (N, T, ...)
    return array of shape (N * n_chunks, seg_size, ...)
        where n_chunks = T // seg_size
        T must be divisible by seg_size
    """
    N, T, *_ = traj.shape
    n_chunks = T // seg_size
    splits = jnp.split(traj, n_chunks, axis=1)  # List[(N, chunk_size, ...)]
    return rearrange(
        splits,
        "n_chunks N seg ... -> (N n_chunks) seg ...",
        n_chunks=n_chunks,
        N=N,
        seg=seg_size,
    )


def rebalance(
    key,
    task_name: str,
    ds: ArrayDict,
    n_bins: int,
    max_count_per_bin: int,
    tokeep: int,
):
    """
    For tasks with skewed return distributions, prune trajectories (by bins) to maintain
    a more balanced distribution of returns. Then subsample to keep only `tokeep` trajectories.
    """

    return_bins = jnp.histogram_bin_edges(ds["returns"], bins=n_bins)

    # prune by bin for tasks with skewed returns
    if task_name in tasks_to_rebalance:
        key, key_prune = jr.split(key, 2)
        ds = prune_bin(key_prune, ds, return_bins, max_count_per_bin)

    # subsample to keep only `tokeep` trajectories
    key, key_subsample = jr.split(key, 2)
    ds = subsample(key_subsample, ds, tokeep)

    # sort (again) by return (ascending)
    sorted_idxes = jnp.argsort(ds["returns"])
    ds = jax.tree.map(lambda x: x[sorted_idxes], ds)
    return ds


def prune_bin(key, ds: ArrayDict, bins: jax.Array, max_count_per_bin: int):
    """
    Prune trajectories to maintain a more balanced distribution of returns.
    For any histogram bin that exceeds max_count_per_bin, randomly select only max_count_per_bin trajectories.

    Args:
        key: JAX random key
        task_name: Name of the task
        ds: Dictionary containing trajectory data
        max_count_per_bin: Maximum number of trajectories to keep in any histogram bin
    """
    # Get bin assignments for each trajectory
    bin_indices = jnp.digitize(ds["returns"], bins) - 1

    # Initialize mask for trajectories to keep
    keep_mask = jnp.zeros_like(ds["returns"], dtype=bool)

    # For each bin, randomly select trajectories if count exceeds max_count_per_bin
    for i in range(len(bins) - 1):
        bin_mask = bin_indices == i
        bin_count = jnp.sum(bin_mask)

        if bin_count <= max_count_per_bin:
            # Keep all trajectories in this bin
            keep_mask = keep_mask | bin_mask
        else:
            # Randomly select max_count_per_bin trajectories
            bin_idxs = jnp.where(bin_mask)[0]
            selected_idxs = jr.permutation(key, bin_idxs)[:max_count_per_bin]
            keep_mask = keep_mask | jnp.isin(
                jnp.arange(len(ds["returns"])), selected_idxs
            )

    # Apply mask to all elements in the dataset
    pruned_ds = jax.tree.map(lambda x: x[keep_mask], ds)

    return pruned_ds


def subsample(key, ds: ArrayDict, tokeep=300):
    """
    Subsample ds to keep only `tokeep` trajectories.
    """
    idxs = jr.permutation(key, len(ds["returns"]))[:tokeep]
    return jax.tree.map(lambda x: x[idxs], ds)
