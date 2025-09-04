from typing import Union

import jax.numpy as jnp
from jaxtyping import Array, Float


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
