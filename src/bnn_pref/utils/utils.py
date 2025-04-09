from datetime import datetime

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Num

from bnn_pref.utils.type import D


def get_random_seed() -> int:
    return int(datetime.now().timestamp())


def get_gaussian_vector(key, dim: int, normalize: bool = True) -> D:
    vec = jr.normal(key, dim)
    if normalize:
        vec /= jnp.linalg.norm(vec)
    return vec


def get_uniform_vector(key, dim: int, normalize: bool = True) -> D:
    vec = jr.uniform(key, dim)
    if normalize:
        vec /= jnp.linalg.norm(vec)
    return vec


def tile_first_dim(x: jnp.ndarray, reps: int):
    expanded = x[None, ...]
    tile_seq = (reps,) + (1,) * x.ndim
    return jnp.tile(expanded, tile_seq)


def vmap_chunked(fn, arr: Num[Array, "N *"], size: int, fout_shape: tuple):
    """
    fn: assumed to be vmapped over first dimension of arr
    apply vmapped fn over first dimension of arr, but do so in chunks of `size` to avoid OOM
    """

    N = arr.shape[0]
    if N <= size:
        values = fn(arr)
    else:
        values = jnp.empty((N, *fout_shape))
        for i in range(0, N, size):
            arr_chunk = arr[i : i + size]
            values_chunk = fn(arr_chunk)
            values = values.at[i : i + size].set(values_chunk)

    return values
