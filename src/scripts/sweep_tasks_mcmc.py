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
from bnn_pref.data import dataset_creators
from bnn_pref.data.pref_utils import BradleyTerry
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import MeanStd, compute_accuracy2_mcmc
from bnn_pref.utils.test_functions import test_functions_dict
from bnn_pref.utils.utils import get_random_seed


def run_experiment(key, cfg):
    # check RLHF paper
    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    mcmc_cfg = cfg["mcmc"]
    dist = BradleyTerry()

    # * generate true params + preference data
    output = dataset_creators[task_cfg["ds_type"]](key, cfg)
    train_prefs, test_prefs = output["train_prefs"], output["test_prefs"]
    learned_reward_fn = test_functions_dict[cfg["fhat"]]
    queries_Q2TD = train_prefs.queries_Q2TD
    _, _, T, D = queries_Q2TD.shape

    # * build + run sampler
    key, key1, key2 = jr.split(key, 3)
    init_sample = jnp.zeros((D,))
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
    sample_D = samples_SD.mean(axis=0)
    sample_D /= jnpl.norm(sample_D)
    test_logpdf = dist.logpdf(sample_D, test_prefs, learned_reward_fn).mean()

    results = {"test_accs": test_accs, "test_logpdf": test_logpdf}
    metadata = None

    return results, metadata


@hydra.main(version_base=None, config_name="configPref", config_path="../cfg")
def run_dimensinality_exp(cfg):
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)

    tasks = [
        "reacher",
        "lunar",
        "cheetah",
        "acrobot",
        "ball",
        "cartpoleSwing",
        "cheetahDMC",
        "hopperHop",
        "pendulum",
        "reacherEasy",
        "reacherHard",
        "walkerWalk",
    ]
    stats = []
    for task in tasks:
        new_cfg = hydra.compose(
            "config",
            overrides=[f"task={task}"],
        )
        key, *subkeys = jr.split(key, 1 + cfg["seeds"])  # m = 1 + n_seeds

        start_time = datetime.now()
        vmap_run_experiment = jax.vmap(run_experiment, in_axes=(0, None))
        res_m, metadata_m = vmap_run_experiment(jnp.array(subkeys), new_cfg)
        duration = (datetime.now() - start_time).total_seconds()

        stat = {
            "test_accs": MeanStd(res_m["test_accs"]),
            "test_logpdf": MeanStd(res_m["test_logpdf"]),
        }
        stats.append(stat)

        print(
            f"{task:10}: "
            f"acc = {stat['test_accs'].mean:.2%} ± {stat['test_accs'].std:.1%}, "
            f"avg_ll = {stat['test_logpdf'].mean:.2f} ± {stat['test_logpdf'].std:.1f}, "
            f"Time: {duration:.1f} seconds"
        )


if __name__ == "__main__":
    run_dimensinality_exp()
