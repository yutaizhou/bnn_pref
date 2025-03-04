import logging
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import arviz as az
import hydra
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpl
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np

from bnn_pref.alg.mcmc import build_hmc, build_mh, plot_samples, plot_trace, run_mcmc
from bnn_pref.data import BradleyTerry, QueryWithResponse, generate_pref_data
from bnn_pref.utils.utils import (
    alignment_metric,
    compute_accuracy2,
    get_gaussian_vector,
    print_mcmc_summary,
    tile_first_dim,
)

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    # check RLHF paper
    data_kw = cfg["data"]
    mcmc_kw = cfg["mcmc"]
    dist = BradleyTerry()
    n_feats = data_kw["n_feats"]

    seed = int(datetime.now().timestamp()) if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed=seed)

    # * generate true weights + preference data
    key, key1, key2 = jr.split(key, 3)
    true_reward_D = get_gaussian_vector(key1, dim=n_feats, normalize=True)
    features_Q2D, response_Q1 = generate_pref_data(key2, true_reward_D, **data_kw)
    data = QueryWithResponse(features_Q2D, response_Q1)

    # * build + run sampler
    key, key1, key2 = jr.split(key, 3)
    init_sample = jnp.zeros_like(true_reward_D)
    alg = build_mh(partial(dist.potential, data=data), sigma=mcmc_kw["sigma"])
    samples_SD, states, infos = run_mcmc(
        key1,
        alg=alg,
        init_sample=init_sample,
        **{k: mcmc_kw[k] for k in ["n_samples", "burn_in", "thinning", "normalize"]},
    )

    # * posterior check
    acc = compute_accuracy2(samples_SD, features_Q2D, response_Q1)
    align = alignment_metric(true_reward_D, samples_SD)
    print_mcmc_summary(cfg, samples_SD, acc, align, seed)

    # * arviz - post processing: label switch
    names = [f"weight_{i}" for i in range(n_feats)]
    true_reward_D = jnp.sort(true_reward_D)
    idx = jnp.argsort(samples_SD[-1, :])
    samples_SD = samples_SD[:, idx]

    samples = [samples_SD[:, i] for i in range(n_feats)]
    posterior_data = {k: tile_first_dim(v, reps=1) for k, v in zip(names, samples)}
    idata = az.from_dict(posterior=posterior_data)

    # summary_stats = az.summary(idata, hdi_prob=0.94)
    # print(f"True: {true_reward_D[:10]}")
    # print(summary_stats)

    # * arviz - posterior
    # axs = az.plot_posterior(idata, ref_val=true_reward_D.tolist())
    # for ax in axs:
    #     ax.set_xlim(-1.1, 1.1)
    # plt.tight_layout()
    # plt.show()

    # * arviz - trace
    if cfg["show_fig"]:
        axs = az.plot_trace(idata)
        for i in range(n_feats):
            axs[i, 0].axvline(true_reward_D[i], color="red", label="True", lw=0.5)
            axs[i, 0].set_xlim(-1.1, 1.1)
            axs[i, 1].axhline(true_reward_D[i], color="red", label="True", lw=0.5)
            axs[i, 1].set_ylim(-1.1, 1.1)
        plt.tight_layout()
        plt.show()

    # * plotting
    # all_samples = jnp.concat([init_sample[None, :], samples_SD], axis=0)
    bbox_dict = {
        "D": data_kw["n_feats"],
        "Q": data_kw["n_queries"],
        "Acc": acc,
        "m": align,
    }
    # plot_trace(key, all_samples, true_reward_D, bbox_dict=bbox_dict)
    # if cfg["show_fig"]:
    #     plt.show()
    # if cfg["save_fig"]:
    #     plt.savefig(f"{cfg['paths']['output_dir']}/trace.png")

    linear_reward_fn = lambda features_D, param_D: features_D @ param_D

    def plot_reward_heatmap(ax, reward_fn, title=None):
        feat_min = features_Q2D.min()
        feat_max = features_Q2D.max()
        X, Y = jnp.mgrid[feat_min:feat_max:100j, feat_min:feat_max:100j]
        Z = jax.vmap(jax.vmap(reward_fn))(jnp.stack([X, Y], axis=-1))
        # ax.pcolormesh(X, Y, Z)
        ax.contourf(X, Y, Z, levels=10)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        if title is not None:
            ax.set_title(title)

    def plot_logpdf(ax, potential, true_param_D=None, samples_SD=None, title=None):
        param_min = -1.1
        param_max = 1.1
        X, Y = jnp.mgrid[param_min:param_max:100j, param_min:param_max:100j]
        Z = jax.vmap(jax.vmap(potential))(jnp.stack([X, Y], axis=-1))
        ax.contourf(X, Y, Z, levels=10)
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
            iterates = samples_SD[jnp.arange(0, samples_SD.shape[0], 30).tolist(), :]
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

    if data_kw["n_feats"] == 2:
        fig, axs = plt.subplots(1, 3, figsize=(12, 5))

        ax = axs[0]
        plot_reward_heatmap(
            ax,
            reward_fn=partial(linear_reward_fn, param_D=true_reward_D),
            title=f"True Reward {true_reward_D}",
        )

        ax = axs[1]
        sample_param = samples_SD.mean(axis=0)
        sample_param /= jnpl.norm(sample_param)
        plot_reward_heatmap(
            ax,
            reward_fn=partial(linear_reward_fn, param_D=sample_param),
            title=f"Posterior Mean {sample_param}",
        )

        ax = axs[2]
        plot_logpdf(
            ax,
            potential=partial(dist.potential, data=data),
            true_param_D=true_reward_D,
            samples_SD=samples_SD,
            title="Logpdf",
        )

        ax.legend()

        text_content = "\n".join([f"{k}: {v:.2f}" for k, v in bbox_dict.items()])
        fig.text(
            x=0.05,  # Right edge alignment
            y=0.98,  # Top edge alignment
            s=text_content,
            fontfamily="monospace",
            fontsize=8,
            linespacing=1.0,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.9,
                edgecolor="black",
                pad=0.8,
            ),
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.80)

        plt.show()


if __name__ == "__main__":
    main()
