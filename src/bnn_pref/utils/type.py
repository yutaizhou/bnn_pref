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
