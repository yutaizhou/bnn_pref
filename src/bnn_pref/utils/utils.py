from datetime import datetime
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpl
import jax.random as jr
import jax.scipy as jsp

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


def print_mcmc_summary(
    cfg,
    train_acc: float,
    test_acc: float,
    test_logpdf: float,
    align: float,
    seed: int,
):
    data_kw = cfg["data"]
    mcmc_kw = cfg["mcmc"]

    print(f"Seed: {seed}")
    print(f"N={data_kw['n_demos']}, Q={data_kw['n_queries']}, D={data_kw['n_feats']}")
    print(
        f"{mcmc_kw['n_samples']} samples w/ {mcmc_kw['burn_in']} burn-in, then {mcmc_kw['thinning']} thinning"
    )
    print(f"Train acc: {train_acc:.2%}")
    print(f"Test acc:  {test_acc:.2%}")
    print(f"Test logpdf: {test_logpdf:.2f}")
    print(f"Cosine Sim: {align:.2f}")
