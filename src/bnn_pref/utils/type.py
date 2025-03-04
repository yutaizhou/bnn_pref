import dataclasses
from typing import Tuple

import flax
from jaxtyping import Array, Float, Int, Scalar

# demonstrations
N = Float[Array, "N"]
D = Float[Array, "D"]
TwoD = Float[Array, "2 D"]
ND = Float[Array, "N features"]
N1 = Float[Array, "N 1"]

# queries
Q = Float[Array, "Q"]
Q1 = Float[Array, "Q 1"]
Q2 = Float[Array, "Q 2"]
Q2D = Float[Array, "Q 2 D"]

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


# EKF
@unpackable_dataclass
class BeliefState:
    mean: Float[Array, "system_dim"]
    cov: Float[Array, "system_dim system_dim"]
    t: int


@unpackable_dataclass
class CAR:
    """Context, Action, Reward"""

    contexts: Float[Array, "n n_features"]
    actions: Int[Array, "n"]
    rewards: Float[Array, "n"]


@unpackable_dataclass
class CARL:
    """Context, Action, Reward, Label (one-hot)"""

    contexts: Float[Array, "n n_features"]
    actions: Int[Array, "n"]
    rewards: Float[Array, "n"]
    labels: Float[Array, "n n_actions"]
