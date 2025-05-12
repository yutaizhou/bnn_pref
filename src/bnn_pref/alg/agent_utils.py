from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, value_and_grad
from jax.flatten_util import ravel_pytree
from jax.lax import scan
from jaxtyping import Array, Float, Int

from bnn_pref.data.data_env import BatchIndexManager, PreferenceEnv, retrieve
from bnn_pref.utils.type import QueryData


class Agent(ABC):
    @abstractmethod
    def init_bel(self, key, warmup_data: QueryData):
        pass

    @abstractmethod
    def update_bel(self, bel, batch: QueryData):
        pass

    @abstractmethod
    def acquire_next_query(
        self, key, bel, env: PreferenceEnv, pool_idxes_Q: Int[Array, "Q"]
    ) -> int:
        pass

    @abstractmethod
    def compute_predictive(
        self,
        key,
        bel,
        items_NTD: Float[Array, "N T D"],
        query_idxs_Q2: Float[Array, "Q 2"],
    ) -> Float[Array, "Q 2"]:
        pass

    @staticmethod
    @abstractmethod
    def get_hydra_config(cls_cfg: Dict):
        pass


def sub2full_params_flat(
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


def bt_loss_fn(params, ts: TrainState, batch: QueryData, l2_reg: float = 0.0):
    contexts_B2TD, labels_B2 = batch.contexts, batch.labels
    logits_B2 = ts.apply_fn({"params": params}, contexts_B2TD)
    loss = optax.softmax_cross_entropy(logits_B2, labels_B2).mean()
    params_flat, _ = ravel_pytree(params)
    l2_loss = l2_reg * (params_flat**2).sum()
    return loss + l2_loss, logits_B2


def run_gradient_descent(
    key,
    ts: TrainState,
    loss_fn: Callable,
    has_aux: bool,
    dataset: QueryData,
    niters: int,
    batch_size: int = -1,
    l2_reg: float = 0.0,
):
    """
    Run GD training for exactly niters steps.
    If batch_size == -1, run full-batch GD. Otherwise, run mini-batch SGD.
    """

    contexts, labels = dataset.contexts, dataset.labels
    N = contexts.shape[0]

    @jit
    def train_step(ts: TrainState, idxs_B: Int[Array, "B"]) -> Tuple[TrainState, Dict]:
        # retrieve batch. If full batch (bs==-1), use all data
        contexts_B2TD = contexts if batch_size == -1 else retrieve(contexts, idxs_B)
        labels_B2 = labels if batch_size == -1 else retrieve(labels, idxs_B)  # one-hot
        batch = QueryData(contexts_B2TD, labels_B2)

        # loss, grad, update
        grad_fn = value_and_grad(loss_fn, has_aux=has_aux)
        val, grads = grad_fn(ts.params, ts, batch, l2_reg)
        loss = val[0] if has_aux else val

        ts = ts.apply_gradients(grads=grads)
        flat_params, _ = ravel_pytree(ts.params)
        return ts, {"loss": loss, "params": flat_params}

    # Create batch manager and get all batches upfront
    batch_manager = BatchIndexManager(key, N, batch_size)
    batch_idxs = batch_manager.get_n_batches(niters)  # (niters, batch_size)

    # Run `niters` steps
    ts, metrics = scan(train_step, init=ts, xs=batch_idxs)
    return ts, metrics


class JaxPCA:
    """JAX-based PCA implementation with scikit-learn-like API."""

    def __init__(self, n_components: Union[int, float] = 0.9999):
        """
        Parameters
        ----------
        n_components : Union[int, float]
            Number of components to keep:
            - if int, number of components to keep
            - if float between 0 and 1, select the number of components such that
              the amount of variance explained is greater than the percentage specified
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None

    def fit(self, X: Float[Array, "N D"]):
        """Fit the model with X."""
        # Validate n_components
        if isinstance(self.n_components, float):
            if not 0 <= self.n_components <= 1.0:
                raise ValueError("n_components must be between 0 and 1")
        else:
            if not 1 <= self.n_components <= min(X.shape[0], X.shape[1]):
                raise ValueError(
                    "n_components must be between 1 and min(n_samples, n_features)"
                )

        # Center the data
        self.mean_ = jnp.mean(X, axis=0, keepdims=True)
        X_centered = X - self.mean_

        # Use SVD instead of eigendecomposition of covariance matrix
        U, S, Vt = jnp.linalg.svd(X_centered, full_matrices=False)

        # Compute explained variance and ratio directly from singular values
        n_samples = X.shape[0]
        self.singular_values_ = S
        self.explained_variance_ = (S**2) / (n_samples - 1)
        total_var = jnp.sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        # Determine number of components
        if isinstance(self.n_components, float):
            cumsum = jnp.cumsum(self.explained_variance_ratio_)
            # self.n_components_ = jnp.sum(cumsum <= self.n_components) + 1
            self.n_components_ = jnp.sum(cumsum < self.n_components) + 1
            self.n_components_ = min(self.n_components_, X.shape[1])
        else:
            # self.n_components_ = self.n_components
            self.n_components_ = min(self.n_components, X.shape[1])

        # Store components (right singular vectors)
        self.components_ = Vt[: self.n_components_]
        return self

    def transform(self, X: Float[Array, "N D"]) -> Float[Array, "N K"]:
        """Apply dimensionality reduction to X."""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: Float[Array, "N D"]) -> Float[Array, "N K"]:
        """Fit the model with X and apply dimensionality reduction to X."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: Float[Array, "N K"]) -> Float[Array, "N D"]:
        """Transform data back to its original space."""
        return X @ self.components_ + self.mean_
