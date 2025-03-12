from typing import Callable, Tuple

import jax.numpy as jnp
import jax.numpy.linalg as jnpl
from einops import rearrange


def plot_reward_heatmap(
    ax,
    reward_fn: Callable,
    bounds: Tuple[float, float],
    title=None,
    plot_3d: bool = False,
):
    """
    reward_fn: Callable, takes in a (100,100,1,2) feature array and returns a (100,100) reward array
        may need to be vmapped over the first two dimensions
    bounds: has options
        - tuple of (min, max), which generates a square grid
        - tuple of ((min, max), (min, max)), which generates a rectangular grid
    """
    if not isinstance(bounds[0], tuple):
        feat_min, feat_max = bounds
        X, Y = jnp.mgrid[feat_min:feat_max:100j, feat_min:feat_max:100j]
    else:
        feat1_min, feat1_max = bounds[0]
        feat2_min, feat2_max = bounds[1]
        X, Y = jnp.mgrid[feat1_min:feat1_max:100j, feat2_min:feat2_max:100j]
    inputs = jnp.stack([X, Y], axis=-1)  # 100,100,2
    inputs = rearrange(inputs, "H W D -> H W 1 D", D=2)  # for time dim
    Z = reward_fn(inputs).squeeze()

    if plot_3d:
        surface = ax.plot_surface(X, Y, Z, cmap="viridis")
        fig = ax.get_figure()
        fig.colorbar(surface, ax=ax)
        ax.set_zlabel("Reward")
    else:
        contour = ax.contourf(X, Y, Z, levels=10)
        fig = ax.get_figure()
        fig.colorbar(contour, ax=ax)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    if title is not None:
        ax.set_title(title)


def plot_logpdf(
    ax,
    potential_fn,
    bounds,
    true_param_D=None,
    samples_SD=None,
    title=None,
):
    """
    potential_fn: Callable, takes in a (100,100,2) parameter array and returns a (100,100) logpdf array
        may need to be vmapped over the first two dimensions
    """
    param_min, param_max = bounds
    X, Y = jnp.mgrid[param_min:param_max:100j, param_min:param_max:100j]
    Z = potential_fn(jnp.stack([X, Y], axis=-1))
    contour = ax.contourf(X, Y, Z, levels=10)
    fig = ax.get_figure()
    fig.colorbar(contour, ax=ax)
    ax.set_xlabel("Param 1")
    ax.set_ylabel("Param 2")

    if title is not None:
        ax.set_title(title)

    if true_param_D is not None:
        ax.scatter(*true_param_D, color="r", marker="*", label="True")

    if samples_SD is not None:
        sample_param = samples_SD.mean(axis=0)
        sample_param /= jnpl.norm(sample_param)
        ax.scatter(*sample_param, color="b", marker=".", label="Posterior Mean")

        # add a few MCMC iterates
        indices = jnp.linspace(0, samples_SD.shape[0] - 1, num=30, dtype=jnp.int32)
        iterates = samples_SD[indices.tolist(), :]
        ax.scatter(
            iterates[:, 0],
            iterates[:, 1],
            color="black",
            marker="x",
            alpha=0.1,
            s=5,
            label="MCMC Iterates",
        )
        # Add confidence ellipses
        cov = jnp.cov(samples_SD.T)
        eigvals, eigvecs = jnpl.eigh(cov)
        theta = jnp.linspace(0, 2 * jnp.pi, 100)

        for n_std in [
            1,
        ]:
            ellipse_x = (
                sample_param[0]
                + n_std * jnp.sqrt(eigvals[0]) * jnp.cos(theta) * eigvecs[0, 0]
                + n_std * jnp.sqrt(eigvals[1]) * jnp.sin(theta) * eigvecs[0, 1]
            )
            ellipse_y = (
                sample_param[1]
                + n_std * jnp.sqrt(eigvals[0]) * jnp.cos(theta) * eigvecs[1, 0]
                + n_std * jnp.sqrt(eigvals[1]) * jnp.sin(theta) * eigvecs[1, 1]
            )
            ax.plot(
                ellipse_x, ellipse_y, "b--", alpha=0.1, label=f"{n_std}σ confidence"
            )
