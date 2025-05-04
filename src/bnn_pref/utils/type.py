import dataclasses
from typing import Dict, NamedTuple

import flax
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, Int, Scalar

# Data types
ArrayDict = Dict[str, jax.Array]
"""
    For trajectories
    ds = {
        "observations": (N, T, D),
        "actions": (N, T, A),
        "rewards": (N, T),
        "returns": (N,),
    }
"""


class QueryData(NamedTuple):
    contexts: Float[Array, "Q 2 T D"]
    labels: Float[Array, "Q 2"]  # one hot

    def add_leading_batch_dim(self):
        return QueryData(
            contexts=rearrange(self.contexts, "K T D -> 1 K T D", K=2),
            labels=rearrange(self.labels, "K -> 1 K", K=2),
        )


# demonstrations
N = Float[Array, "N"]
D = Float[Array, "D"]
NTD = Float[Array, "N T D"]
TD = Float[Array, "T D"]
N1 = Float[Array, "N 1"]

# queries
Q = Float[Array, "Q"]
Q1 = Float[Array, "Q 1"]
Q2 = Float[Array, "Q 2"]
Q2TD = Float[Array, "Q 2 T D"]

# MCMC samples
SD = Float[Array, "S features"]


def unpackable_dataclass(cls=None, **kwargs):
    """Custom decorator extending flax.struct.dataclass to support unpacking."""

    def wrap(cls):
        # Apply flax.struct.dataclass to make the class JAX-compatible
        cls = flax.struct.dataclass(cls, **kwargs)

        # Add an __iter__ method to allow unpacking
        def __iter__(self):
            return iter(getattr(self, field.name) for field in dataclasses.fields(self))

        cls.__iter__ = __iter__
        return cls

    # Support usage with or without parentheses
    if cls is None:
        return wrap
    return wrap(cls)
