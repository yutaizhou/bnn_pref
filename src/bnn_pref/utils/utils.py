from collections import defaultdict
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Num

from bnn_pref.utils.type import D


def nested_defaultdict():
    """module level defaultdict of defaultdicts, pickleable"""
    return defaultdict(nested_defaultdict)


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
