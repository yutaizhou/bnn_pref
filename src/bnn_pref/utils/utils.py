import os
import time
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


class _TimerContextManager:
    def __init__(self, timer: "Timer", key: str):
        self.timer = timer
        self.key = key

    def __enter__(self):
        self.timer.tick(self.key)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.timer.tock(self.key)


class Timer:
    """
    Usage:
    timer = Timer()

    timer.tick("some_stuff")
    # do stuff
    timer.tock("some_stuff")

    print(timer.get_average_times())


    Taken from: https://github.com/nakamotoo/V-GPS/blob/main/jaxrl_m/utils/timer_utils.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = defaultdict(int)
        self.times = defaultdict(float)
        self.start_times = {}

    def tick(self, key):
        if key in self.start_times:
            raise ValueError(f"Timer is already ticking for key: {key}")
        self.start_times[key] = time.time()

    def tock(self, key):
        if key not in self.start_times:
            raise ValueError(f"Timer is not ticking for key: {key}")
        self.counts[key] += 1
        self.times[key] += time.time() - self.start_times[key]
        del self.start_times[key]

    def context(self, key):
        """
        Use this like:

        with timer.context("key"):
            # do stuff

        Then timer.tock("key") will be called automatically.
        """
        return _TimerContextManager(self, key)

    def get_average_times(self, reset=True):
        ret = {key: self.times[key] / self.counts[key] for key in self.counts}
        if reset:
            self.reset()
        return ret

    def get_total_times(self, reset=True):
        ret = {key: self.times[key] for key in self.times}
        if reset:
            self.reset()
        return ret
