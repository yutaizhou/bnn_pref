from datetime import date

import blackjax
import distrax
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np


class BimodalGaussian:
    def __init__(self, weights, means, variance):
        """
        Args:
            weights: Array-like (2,) mixture coefficients (sum to 1)
            means: Array-like (2,) component means
            variance: Shared variance (σ²) for both components
        """
        self.weights = jnp.asarray(weights)
        self.means = jnp.asarray(means)
        self.variance = jnp.asarray(variance)
        self.std = jnp.sqrt(variance)

        # Create TFP distribution using MixtureSameFamily
        self._dist = distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(probs=self.weights),
            components_distribution=distrax.Normal(
                loc=self.means,  # Shape [2]
                scale=self.std,  # Shared scalar -> broadcast to [2]
            ),
        )

    def log_prob(self, x):
        """Log probability density function"""
        return self._dist.log_prob(x)

    def sample(self, key, sample_shape=()):
        """Generate samples using JAX PRNG key"""
        return self._dist.sample(sample_shape=sample_shape, seed=key)

    def sample_and_log_prob(self, key, sample_shape=()):
        """Joint sampling and log probability calculation"""
        return self._dist.experimental_sample_and_log_prob(
            sample_shape=sample_shape, seed=key
        )

    @property
    def component_distributions(self):
        """Access component distributions"""
        return self._dist.components_distribution


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    gmm = BimodalGaussian(
        weights=jnp.array([0.6, 0.4]), means=jnp.array([-0.5, 0.5]), variance=1.0
    )

    # Sampling and evaluation
    samples = gmm.sample(key, sample_shape=(1000,))
    log_probs = gmm.log_prob(samples)

    # Joint sampling + log_prob (more efficient)
    samples, log_probs = gmm.sample_and_log_prob(key, (1000,))

    print(samples.shape, log_probs.shape)
