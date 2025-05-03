from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Bool, Float

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


def normalize(
    x: Float[Array, "... D"],
    axis: Tuple[int, ...],
) -> Float[Array, "... D"]:
    """
    Designed to work with both trajectoy data (N, T, D) where axis=(0, 1)
    and samples data (N, D) where axis=(0,)
    """
    eps = 1e-8
    mean = jnp.mean(x, axis=axis, keepdims=True)
    std = jnp.std(x, axis=axis, keepdims=True)
    std = jnp.maximum(std, eps)
    return (x - mean) / std


def process_rewards(
    r: Float[Array, "N"],
    norm: bool = True,
    clip: bool = True,
    clip_value: float = 10.0,
) -> Float[Array, "N"]:
    """
    Normalize and clip rewards
    """
    if norm:
        r = normalize(r, axis=(0,))
    if clip:
        r = jnp.clip(r, -clip_value, clip_value)
    return r


def segment_traj(
    traj: Float[Array, "N T ..."],
    segment_size: int = 50,
):
    """
    traj: (N, T, ...)
    return array of shape (N * n_chunks, seg_size, ...)
        where n_chunks = T // seg_size
        T must be divisible by seg_size
    """
    assert traj.ndim >= 2
    traj = jnp.atleast_3d(traj)  # for rewards (N, T) -> (N, T, 1)
    N, T, *D = traj.shape
    assert T >= segment_size > 0, f"{segment_size=} must be <= {T=} and positive"
    assert T % segment_size == 0, f"{T=} must be divisible by {segment_size=}"
    n_chunks = T // segment_size
    splits = jnp.split(traj, n_chunks, axis=1)  # List[(N, chunk_size, ...)]
    return rearrange(
        splits,
        "n_chunks N seg ... -> (N n_chunks) seg ...",
        n_chunks=n_chunks,
        N=N,
        seg=segment_size,
    )


def segment_traj_masked(
    traj: Float[Array, "N T ..."],
    mask: Bool[Array, "N T"],
    segment_size: int = 50,
):
    """
    Segment trajectory into non-overlapping chunks of size segment_size.
    The last segment for each trajectory will be anchored at its last valid timestep.

    Args:
        traj: (N, T, ...) trajectory array, may be padded
        mask: (N, T) boolean mask indicating valid timesteps
        segment_size: size of each chunk

    Returns:
        segments_STD: (S, segment_size, ...) array of segments, where S is the total
                     number of valid segments across all trajectories
    """
    assert traj.ndim >= 2, "traj must have at least 2 dimensions"
    traj = jnp.atleast_3d(traj)  # for rewards (N, T) -> (N, T, 1)
    N, T, *D = traj.shape
    assert T >= segment_size > 0, f"{segment_size=} must be <= {T=} and positive"

    # Calculate number of full non-overlapping segments per trajectory
    n_full_segments = T // segment_size

    # Create segment start indices for non-overlapping segments [0, sz, 2*sz, ...]
    full_segment_bgns = jnp.arange(n_full_segments) * segment_size

    # Initialize lists to store valid segments and their masks
    all_valid_segments = []

    # Process each trajectory separately
    for i in range(N):
        traj_i = traj[i]  # (T, ...)
        mask_i = mask[i]  # (T,)

        # Find last valid index for this trajectory
        last_valid_idx = jnp.max(jnp.where(mask_i, jnp.arange(T), -1))

        # Calculate last segment bgn for this trajectory
        # Ensure it captures the last valid timestep
        last_segment_bgn = jnp.maximum(0, last_valid_idx - segment_size + 1)

        # Determine segment beginnings for this trajectory
        if last_segment_bgn > full_segment_bgns[-1]:
            # Add last segment if it's beyond the full segments
            segment_bgns = jnp.concatenate(
                [full_segment_bgns, jnp.array([last_segment_bgn])]
            )
        else:
            segment_bgns = full_segment_bgns

        # Create indices for all segments in this trajectory
        segment_indices = segment_bgns[:, None] + jnp.arange(segment_size)[None, :]

        # Extract segments
        segments = traj_i[segment_indices]  # (n_segments, segment_size, ...)

        # Check which segments are fully valid
        segment_masks = jnp.all(mask_i[segment_indices], axis=1)  # (n_segments,)

        # Only keep valid segments
        valid_segments = segments[segment_masks]

        if len(valid_segments) > 0:
            all_valid_segments.append(valid_segments)

    # Combine valid segments from all trajectories
    if all_valid_segments:
        valid_segments_STD = jnp.concatenate(all_valid_segments, axis=0)
    else:
        # Return empty array with correct shape if no valid segments
        valid_segments_STD = jnp.zeros((0, segment_size, *D))

    return valid_segments_STD


# def compute_pad_size(T: int, segment_size: int) -> int:
#     """
#     calculate pad_width required to pad up to next divisor of segment_size. Good for jnp.split
#     T % segment_size == 0. e.g. compute_pad_size(138, 50) -> 12
#     """
#     assert T > segment_size > 0, f"{segment_size=} must be <= {T=} and positive"
#     return (T + segment_size - 1) // segment_size * segment_size - T


# def segment_traj_masked(
#     traj: Float[Array, "N T ..."],
#     mask: Bool[Array, "N T"],
#     segment_size: int = 50,
# ):
#     """
#     segment traj into chunks of size segment_size, but only include chunks where mask is True

#     traj: (N, T, ...) may already be padded
#     mask: (N, T) indicates valid (unpadded) steps in traj
#     """
#     # * pad traj & mask to ensure T is divisible by segment_size
#     N, T, *_ = traj.shape
#     assert T > segment_size > 0, f"{segment_size=} must be <= {T=} and positive"
#     pad_sz = compute_pad_size(T, segment_size)
#     traj = jnp.pad(traj, ((0, 0), (0, pad_sz), *((0, 0) for _ in range(traj.ndim - 2))))
#     mask = jnp.pad(mask, ((0, 0), (0, pad_sz)))
#     assert traj.shape[1] % segment_size == 0, "T must be divisible by segment_size"

#     # * split into chunks
#     n_chunks = traj.shape[1] // segment_size
#     splits = jnp.split(traj, n_chunks, axis=1)  # List[(N, chunk_size, ...)]
#     return rearrange(
#         splits,
#         "n_chunks N seg ... -> (N n_chunks) seg ...",
#         n_chunks=n_chunks,
#         N=N,
#     )


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
    a more balanced distribution of returns. Then uniformly subsample to keep only `tokeep` trajectories.
    """

    return_bins = jnp.histogram_bin_edges(ds["returns"], bins=n_bins)

    # prune by bin for tasks with skewed returns
    if task_name in tasks_to_rebalance:
        key, key_prune = jr.split(key, 2)
        ds = prune_bin(key_prune, ds, return_bins, max_count_per_bin)

    # uniformly subsample to keep only `tokeep` trajectories
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
