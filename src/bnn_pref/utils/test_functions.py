import jax.numpy as jnp

from bnn_pref.utils.type import TD

"""
T can be 1 for traj-level embedding rather than state-level
sum(-1) for summing across T dimension, in case first dimension is batch
"""


def linear_reward_fn(features: TD, param_D):
    return (features @ param_D).sum(-1)


def polynomial_reward_fn(features: TD, param_D):
    return (features**2 @ param_D**2).sum(-1)


def sinusoidal_reward_fn(features: TD, param_D):
    return jnp.sin(1.2 * jnp.pi * (features @ param_D)).sum(-1)


test_functions_dict = {
    "linear": linear_reward_fn,
    "poly": polynomial_reward_fn,
    "sin": sinusoidal_reward_fn,
}
