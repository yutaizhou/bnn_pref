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
from bnn_pref.utils.plotting import plot_logpdf, plot_reward_heatmap
from bnn_pref.utils.utils import (
    alignment_metric,
    compute_accuracy2_mcmc,
    get_gaussian_vector,
    linear_reward_fn,
    print_mcmc_summary,
    tile_first_dim,
)

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)

# todo try other reward function classes


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
    true_param_D = get_gaussian_vector(key1, dim=n_feats, normalize=True)
    true_reward_fn = linear_reward_fn
    features_Q2D, response_Q1 = generate_pref_data(
        key2, reward_fn=true_reward_fn, params_D=true_param_D, **data_kw
    )
    data = QueryWithResponse(features_Q2D, response_Q1)

    # * build + run sampler
    key, key1, key2 = jr.split(key, 3)
    init_sample = jnp.zeros_like(true_param_D)
    alg = build_mh(
        partial(dist.potential, data=data, reward_fn=true_reward_fn),
        sigma=mcmc_kw["sigma"],
    )
    samples_SD, states, infos = run_mcmc(
        key1,
        alg=alg,
        init_sample=init_sample,
        **{k: mcmc_kw[k] for k in ["n_samples", "burn_in", "thinning", "normalize"]},
    )

    # * posterior check
    acc = compute_accuracy2_mcmc(samples_SD, features_Q2D, response_Q1, true_reward_fn)
    align = alignment_metric(true_param_D, samples_SD)
    print_mcmc_summary(cfg, samples_SD, acc, align, seed)

    # * arviz - post processing: label switch
    names = [f"weight_{i}" for i in range(n_feats)]
    true_param_D = jnp.sort(true_param_D)
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
            axs[i, 0].axvline(true_param_D[i], color="red", label="True", lw=0.5)
            axs[i, 0].set_xlim(-1.1, 1.1)
            axs[i, 1].axhline(true_param_D[i], color="red", label="True", lw=0.5)
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

    if data_kw["n_feats"] == 2:
        fig, axs = plt.subplots(1, 3, figsize=(12, 5))

        ax = axs[0]
        true_utility_fn = partial(linear_reward_fn, param_D=true_param_D)
        true_utility_fn = jax.vmap(jax.vmap(true_utility_fn))
        plot_reward_heatmap(
            ax,
            reward_fn=true_utility_fn,
            bounds=(features_Q2D.min(), features_Q2D.max()),
            title=f"True Reward {true_param_D}",
        )

        ax = axs[1]
        sample_param = samples_SD.mean(axis=0)
        sample_param /= jnpl.norm(sample_param)
        posterior_utility_fn = partial(linear_reward_fn, param_D=sample_param)
        posterior_utility_fn = jax.vmap(jax.vmap(posterior_utility_fn))
        plot_reward_heatmap(
            ax,
            reward_fn=posterior_utility_fn,
            bounds=(features_Q2D.min(), features_Q2D.max()),
            title=f"Posterior Predictive Reward {sample_param}",
        )

        ax = axs[2]
        potential = partial(dist.potential, data=data, reward_fn=true_reward_fn)
        potential = jax.vmap(jax.vmap(potential))
        plot_logpdf(
            ax,
            potential_fn=potential,
            bounds=(-1.1, 1.1),
            true_param_D=true_param_D,
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
