import jax.numpy as jnp

from bnn_pref.utils.type import Q1, Q2, Q2D, SD, D, Q


def linear_reward_fn(features_D: D, param_D: D) -> D:
    return features_D @ param_D


def polynomial_reward_fn(features_D, param_D):
    return features_D**2 @ param_D**2


def sinusoidal_reward_fn(features_D, param_D):
    return jnp.sin(features_D @ param_D)


test_functions_dict = {
    "linear": linear_reward_fn,
    "poly": polynomial_reward_fn,
    "sin": sinusoidal_reward_fn,
}
