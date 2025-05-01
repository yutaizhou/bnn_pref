import os
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


def get_random_seed(seed: int = -1) -> int:
    """
    If seed is -1, use the current time as the seed. Otherwise, use the seed provided.
    """
    if seed == -1:
        return int(datetime.now().timestamp())
    else:
        return seed


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


def slurm_auto_scancel():
    """
    Call at the end of scripts to prevent completed jobs from hanging on slurm.
    """
    is_slurm = bool(os.environ.get("SLURM_JOB_ID")) or bool(
        os.environ.get("SLURM_ARRAY_JOB_ID")
    )

    if is_slurm:
        if os.environ.get("SLURM_ARRAY_JOB_ID"):
            slurm_job_id = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
        else:
            slurm_job_id = os.environ["SLURM_JOB_ID"]
        os.system(f"scancel {slurm_job_id}")
    else:
        pass


def get_cuda_visible_devices():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    local_device_ids = [int(i) for i in cuda_visible_devices.split(",")]
    return local_device_ids
