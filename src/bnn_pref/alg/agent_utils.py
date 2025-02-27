import jax.numpy as jnp
import jax.random as jr
from jax import jit, value_and_grad
from jax.flatten_util import ravel_pytree
from jax.lax import scan


def convert_params_from_subspace_to_full(
    params_subspace,
    projection_matrix,
    params_full,
):
    params = jnp.matmul(params_subspace, projection_matrix) + params_full
    return params


def generate_random_basis(key, d: int, D: int):
    """
    return projection matrix P: fixed but random Gaussian matrix
    with columns normalized to 1,
    """
    P = jr.normal(key, shape=(d, D))
    P = P / jnp.linalg.norm(P, axis=-1, keepdims=True)
    return P


def train(state, loss_fn, nepochs=300, has_aux=True):
    @jit
    def step(state, _):
        grad_fn = value_and_grad(loss_fn, has_aux=has_aux)
        val, grads = grad_fn(state.params)
        loss = val[0] if has_aux else val
        state = state.apply_gradients(grads=grads)
        flat_params, _ = ravel_pytree(state.params)
        return state, {"loss": loss, "params": flat_params}

    state, metrics = scan(step, state, jnp.empty(nepochs))

    return state, metrics
