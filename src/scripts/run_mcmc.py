import os

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
import logging
from functools import partial

import arviz as az
import hydra
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpl
import jax.random as jr
import matplotlib.pyplot as plt

from bnn_pref.alg.mcmc import build_hmc, build_mh, plot_samples, plot_trace, run_mcmc
from bnn_pref.data import dataset_creators
from bnn_pref.data.pref_utils import BradleyTerry
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import alignment_metric, compute_accuracy2_mcmc
from bnn_pref.utils.plotting import plot_logpdf, plot_reward_heatmap
from bnn_pref.utils.print_utils import print_mcmc_cfg
from bnn_pref.utils.test_functions import test_functions_dict
from bnn_pref.utils.utils import get_random_seed, tile_first_dim

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    # check RLHF paper
    data_cfg = cfg["data"]
    mcmc_cfg = cfg["mcmc"]
    task_cfg = cfg["task"]
    dist = BradleyTerry()

    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)

    # * generate true params + preference data
    output = dataset_creators[task_cfg["ds_type"]](key, cfg)
    train_prefs, test_prefs = output["train_prefs"], output["test_prefs"]
    learned_reward_fn = test_functions_dict[cfg["fhat"]]
    _, _, T, D = train_prefs.queries_Q2TD.shape
    print_mcmc_cfg(seed, cfg, length=T, n_feats=D)

    # * build + run sampler
    key, key1, key2 = jr.split(key, 3)
    init_sample = jnp.zeros(D)
    alg = build_mh(
        partial(dist.potential, data=train_prefs, reward_fn=learned_reward_fn),
        sigma=mcmc_cfg["sigma"],
    )
    samples_SD, states, infos = run_mcmc(
        key1,
        alg=alg,
        init_sample=init_sample,
        **{k: mcmc_cfg[k] for k in ["n_samples", "burn_in", "thinning", "normalize"]},
    )

    # * posterior check
    train_acc = compute_accuracy2_mcmc(samples_SD, train_prefs, learned_reward_fn)
    test_acc = compute_accuracy2_mcmc(samples_SD, test_prefs, learned_reward_fn)
    sample_D = samples_SD.mean(axis=0)
    sample_D /= jnpl.norm(sample_D)
    test_logpdf = dist.logpdf(sample_D, test_prefs, learned_reward_fn).mean()
    if task_cfg["ds_type"] == "synthetic":
        true_param_D, true_reward_fn = output["true_param"], output["true_reward_fn"]
        align = alignment_metric(true_param_D, samples_SD)
    else:
        align = None
    print(f"Train acc: {train_acc:.2%}")
    print(f"Test acc:  {test_acc:.2%}")
    print(f"Test logpdf: {test_logpdf:.2f}")
    if align is not None:
        print(f"Cosine Sim: {align:.2f}")

    # * arviz - post processing: label switch
    # names = [f"weight_{i}" for i in range(n_feats)]
    # true_param_D = jnp.sort(true_param_D)
    # idx = jnp.argsort(samples_SD[-1, :])
    # samples_SD = samples_SD[:, idx]

    # samples = [samples_SD[:, i] for i in range(n_feats)]
    # posterior_data = {k: tile_first_dim(v, reps=1) for k, v in zip(names, samples)}
    # idata = az.from_dict(posterior=posterior_data)

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
    # if cfg["show_fig"]:
    #     axs = az.plot_trace(idata)
    #     for i in range(n_feats):
    #         axs[i, 0].axvline(true_param_D[i], color="red", label="True", lw=0.5)
    #         axs[i, 0].set_xlim(-1.1, 1.1)
    #         axs[i, 1].axhline(true_param_D[i], color="red", label="True", lw=0.5)
    #         axs[i, 1].set_ylim(-1.1, 1.1)
    #     plt.tight_layout()
    #     plt.show()

    # * plotting
    # all_samples = jnp.concat([init_sample[None, :], samples_SD], axis=0)
    # bbox_dict = {
    #     "D": task_cfg["n_feats"],
    #     "Q": data_cfg["nq_train"],
    #     "Train Acc": train_acc,
    #     "Test Acc": test_acc,
    #     "m": align,
    # }
    # plot_trace(key, all_samples, true_reward_D, bbox_dict=bbox_dict)
    # if cfg["show_fig"]:
    #     plt.show()
    # if cfg["save_fig"]:
    #     plt.savefig(f"{cfg['paths']['output_dir']}/trace.png")

    if D == 2 and T == 1:
        nrows, ncols = 2, 3
        fig = plt.figure(figsize=(12, 5))
        feature_bounds = (
            train_prefs.queries_Q2TD.min(),
            train_prefs.queries_Q2TD.max(),
        )

        true_utility_fn = partial(true_reward_fn, param_D=true_param_D)
        true_utility_fn = jax.vmap(jax.vmap(true_utility_fn))
        title = f"True Reward {true_param_D}"
        true_plotkw = {"reward_fn": true_utility_fn, "bounds": feature_bounds}
        ax = fig.add_subplot(nrows, ncols, 1, projection="3d")
        plot_reward_heatmap(ax, **true_plotkw, title=title, plot_3d=True)
        ax = fig.add_subplot(nrows, ncols, 4)
        plot_reward_heatmap(ax, **true_plotkw, title=title, plot_3d=False)

        sample_param = samples_SD.mean(axis=0)
        sample_param /= jnpl.norm(sample_param)
        learned_utility_fn = partial(learned_reward_fn, param_D=sample_param)
        learned_utility_fn = jax.vmap(jax.vmap(learned_utility_fn))
        title = f"Posterior Predictive Reward {sample_param}"
        learned_plotkw = {"reward_fn": learned_utility_fn, "bounds": feature_bounds}
        ax = fig.add_subplot(nrows, ncols, 2, projection="3d")
        plot_reward_heatmap(ax, **learned_plotkw, title=title, plot_3d=True)
        ax = fig.add_subplot(nrows, ncols, 5)
        plot_reward_heatmap(ax, **learned_plotkw, title=title, plot_3d=False)

        potential = partial(dist.potential, data=train_prefs, reward_fn=true_reward_fn)
        potential = jax.vmap(jax.vmap(potential))
        title = f"True Logpdf {true_param_D}"
        logpdf_plotkw = {"potential_fn": potential, "bounds": (-3, 3)}
        ax = fig.add_subplot(nrows, ncols, 3)
        plot_logpdf(
            ax,
            **logpdf_plotkw,
            title=title,
            true_param_D=true_param_D,
            samples_SD=samples_SD,
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
