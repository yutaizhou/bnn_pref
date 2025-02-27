from typing import Tuple

import flax
from jaxtyping import Array, Float, Int

# demonstrations
N = Float[Array, "N"]
D = Float[Array, "D"]
ND = Float[Array, "N features"]
N1 = Float[Array, "N 1"]

# queries
Q = Float[Array, "Q"]
Q1 = Float[Array, "Q 1"]
Q2 = Float[Array, "Q 2"]
Q2D = Float[Array, "Q 2 D"]

# MCMC samples
SD = Float[Array, "S features"]


# EKF
@flax.struct.dataclass
class BeliefState:
    mean: Float[Array, "system_dim"]
    cov: Float[Array, "system_dim system_dim"]
    t: int


@flax.struct.dataclass
class CAR:
    """Context, Action, Reward"""

    contexts: Float[Array, "n n_features"]
    actions: Int[Array, "n"]
    rewards: Float[Array, "n"]


@flax.struct.dataclass
class CARL:
    """Context, Action, Reward, Label (one-hot)"""

    contexts: Float[Array, "n n_features"]
    actions: Int[Array, "n"]
    rewards: Float[Array, "n"]
    labels: Float[Array, "n n_actions"]
