from typing import Dict, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int


def generate_random_basis(
    key,
    d: int,
    D: int,
    type: str = "dense",
    # * sparse projection kwargs
    sparsity: float = None,
    k: int = None,
):
    if type == "dense":
        return generate_dense_random_basis(key, d, D)
    elif type == "sparse":
        return generate_sparse_random_basis(key, d, D, sparsity, k)
    else:
        raise ValueError(f"Invalid projection type: {type}")


def sub2full_params_flat(
    params_subspace: Float[Array, "sub_dim"],
    proj_matrix: Union[Float[Array, "sub_dim full_dim"], Dict],
    params_full: Float[Array, "full_dim"],
    type: str = "dense",
) -> Float[Array, "full_dim"]:
    """
    Project from subspace to full space.
    Handles both regular projection matrix (array) and Fast Food parameters (dict).
    """
    assert type in ["dense", "fastfood", "sparse"], "Invalid projection type"
    if type == "dense":
        return params_subspace @ proj_matrix + params_full
    elif type == "sparse":
        return _project_sparse_random_basis(params_subspace, proj_matrix, params_full)
    else:
        raise ValueError(f"Invalid projection type: {type}")


def generate_dense_random_basis(key, d: int, D: int):
    """
    return projection matrix P: fixed but random Gaussian matrix
    with rows normalized to 1,
    """
    P = jr.normal(key, shape=(d, D))
    P = P / jnp.linalg.norm(P, axis=-1, keepdims=True)
    return P


def generate_sparse_random_basis(
    key, d: int, D: int, sparsity: float = None, k: int = None
):
    """
    Generate sparse random projection matrix.

    Args:
        key: JAX random key
        d: subspace dimension
        D: full dimension
        sparsity: fraction of non-zero entries (e.g., 0.01 = 1% non-zero)
        k: number of non-zero entries per row (alternative to sparsity)

    Returns:
        Sparse matrix representation (could use JAX BCOO or custom format)
    """
    if k is None:
        if sparsity is None:
            # Default: sqrt(D) non-zeros per row (common choice)
            k = int(jnp.sqrt(D))
        else:
            k = int(sparsity * D)

    k = max(1, min(k, D))  # Ensure 1 <= k <= D

    # Generate random column indices for each row
    keys = jr.split(key, d)

    # Use vmap to generate k random integers per key
    def _randint_row(key_row):
        return jr.randint(key_row, (k,), 0, D)

    col_indices = jax.vmap(_randint_row)(keys)  # (d, k)

    # Generate random values (Gaussian or Rademacher)
    # Use vmap to generate k random values per key
    def _normal_row(key_row):
        return jr.normal(key_row, (k,))

    values = jax.vmap(_normal_row)(keys)  # (d, k) Gaussian
    # Or: values = jax.vmap(lambda k_row: (jr.randint(k_row, (k,), 0, 2) * 2 - 1).astype(jnp.float32))(keys)  # Rademacher

    # Normalize each row
    row_norms = jnp.linalg.norm(values, axis=1, keepdims=True)  # (d, 1)
    values = values / (row_norms + 1e-10)

    return {
        "col_indices": col_indices,  # (d, k)
        "values": values,  # (d, k)
        "d": d,
        "D": D,
        "k": k,
    }


def _project_sparse_random_basis(
    params_subspace: Float[Array, "sub_dim"],
    sparse_params: Dict,
    params_full: Float[Array, "full_dim"],
) -> Float[Array, "full_dim"]:
    """Project using sparse matrix: sum_i params_subspace[i] * sparse_row_i"""
    col_indices = sparse_params["col_indices"]  # (d, k)
    values = sparse_params["values"]  # (d, k)
    d = sparse_params["d"]
    D = sparse_params["D"]

    # Vectorized version: flatten all indices and values, then scatter
    # Each row i contributes k values at positions col_indices[i] with values params_subspace[i] * values[i]
    flat_indices = col_indices.reshape(-1)  # (d * k,)
    flat_values = (params_subspace[:, None] * values).reshape(-1)  # (d * k,)

    # Use scatter_add to accumulate all contributions at once
    result = jnp.zeros(D)
    result = result.at[flat_indices].add(flat_values)

    return result + params_full
