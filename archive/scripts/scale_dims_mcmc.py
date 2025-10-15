import logging
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

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
from bnn_pref.data import make_synthetic_data
from bnn_pref.data.pref_utils import BradleyTerry
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import MeanStd, alignment_metric, compute_accuracy2_mcmc
from bnn_pref.utils.test_functions import test_functions_dict
from bnn_pref.utils.utils import get_random_seed


def run_experiment(key, cfg):
    # check RLHF paper
    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    mcmc_cfg = cfg["mcmc"]
    dist = BradleyTerry()

    # * generate true params + preference data
    output = make_synthetic_data(key, cfg)
    train_prefs, test_prefs = output["train_prefs"], output["test_prefs"]
    true_param_D, true_reward_fn = output["true_param"], output["true_reward_fn"]
    learned_reward_fn = test_functions_dict[task_cfg["fhat"]]

    # * build + run sampler
    key, key1, key2 = jr.split(key, 3)
    init_sample = jnp.zeros_like(true_param_D)
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

    test_accs = compute_accuracy2_mcmc(samples_SD, test_prefs, learned_reward_fn)
    aligns = alignment_metric(true_param_D, samples_SD)
    sample_D = samples_SD.mean(axis=0)
    sample_D /= jnpl.norm(sample_D)
    test_logpdf = dist.logpdf(sample_D, test_prefs, learned_reward_fn).mean()

    results = {"test_accs": test_accs, "aligns": aligns, "test_logpdf": test_logpdf}
    metadata = {"true_reward": true_param_D}

    return results, metadata


@hydra.main(version_base=None, config_name="configScaling", config_path="../cfg")
def run_dimensinality_exp(cfg):
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)

    # n_feats_list = [2, 3]
    n_feats_list = [3, 10, 30, 50, 100, 150, 300, 500, 1000]
    stats = []
    for n_feats in n_feats_list:
        new_cfg = hydra.compose(
            "config",
            overrides=["task=synthetic", f"task.n_feats={n_feats}"],
        )
        key, *subkeys = jr.split(key, 1 + cfg["seeds"])  # m = 1 + n_seeds

        start_time = datetime.now()
        vmap_run_experiment = jax.vmap(run_experiment, in_axes=(0, None))
        res_m, metadata_m = vmap_run_experiment(jnp.array(subkeys), new_cfg)
        duration = (datetime.now() - start_time).total_seconds()

        stat = {
            "n_feats": n_feats,
            "test_acc": MeanStd(res_m["test_accs"]),
            "aligns": MeanStd(res_m["aligns"]),
            "test_logpdf": MeanStd(res_m["test_logpdf"]),
        }
        stats.append(stat)

        print(
            f"n_feats={n_feats:4}, acc = {stat['test_acc'].mean:.2%} ± {stat['test_acc'].std:.1%}, "
            f"align = {stat['aligns'].mean:.2f} ± {stat['aligns'].std:.1f}, "
            f"avg_ll = {stat['test_logpdf'].mean:.2f} ± {stat['test_logpdf'].std:.1f}, "
            f"Time: {duration:.1f} seconds"
        )

    # Create single plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot accuracy and alignment on primary y-axis
    color1 = "tab:blue"
    ax1.errorbar(
        [stat["n_feats"] for stat in stats],
        [stat["test_acc"].mean for stat in stats],
        yerr=[stat["test_acc"].std for stat in stats],
        label="Accuracy",
        marker="o",
        markersize=3,
        color=color1,
    )
    ax1.errorbar(
        [stat["n_feats"] for stat in stats],
        [stat["aligns"].mean for stat in stats],
        yerr=[stat["aligns"].std for stat in stats],
        label="Alignment",
        marker="o",
        markersize=3,
        color="green",
    )
    ax1.set_xlabel("Num Dimensions")
    ax1.set_ylabel("Accuracy / Alignment")
    ax1.tick_params(axis="y")
    ax1.set_ylim(0, 1)

    # Create secondary y-axis for log-likelihood
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.errorbar(
        [stat["n_feats"] for stat in stats],
        [stat["test_logpdf"].mean for stat in stats],
        yerr=[stat["test_logpdf"].std for stat in stats],
        label="Log-Likelihood",
        marker="o",
        markersize=3,
        color=color2,
    )

    ax2.set_ylabel("Log-Likelihood", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("MCMC Sweep")

    # Save the figure
    fp = Path(cfg.paths.output_dir) / "mcmc_sweep.png"
    plt.savefig(fp)
    plt.show()


if __name__ == "__main__":
    run_dimensinality_exp()
