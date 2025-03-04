from typing import Callable

import jax.numpy as jnp
import jax.random as jr
from flax.training.train_state import TrainState
from jax import jit, value_and_grad
from jax.flatten_util import ravel_pytree
from jax.lax import scan
from jaxtyping import Array, Float


def subspace2full_params(
    params_subspace: Float[Array, "sub_dim"],
    proj_matrix: Float[Array, "sub_dim full_dim"],
    params_full: Float[Array, "full_dim"],
) -> Float[Array, "full_dim"]:
    params = params_subspace @ proj_matrix + params_full
    return params


def generate_random_basis(key, d: int, D: int):
    """
    return projection matrix P: fixed but random Gaussian matrix
    with columns normalized to 1,
    """
    P = jr.normal(key, shape=(d, D))
    P = P / jnp.linalg.norm(P, axis=-1, keepdims=True)
    return P


def train_sgd(
    ts: TrainState,
    loss_fn: Callable,
    n_epochs: int = 300,
    has_aux: bool = True,
):
    @jit
    def step(state, _):
        grad_fn = value_and_grad(loss_fn, has_aux=has_aux)
        val, grads = grad_fn(state.params)
        loss = val[0] if has_aux else val
        state = state.apply_gradients(grads=grads)
        flat_params, _ = ravel_pytree(state.params)
        return state, {"loss": loss, "params": flat_params}

    ts, metrics = scan(step, init=ts, xs=jnp.empty(n_epochs))

    return ts, metrics
