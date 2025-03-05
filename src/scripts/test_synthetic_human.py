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
from bnn_pref.data import create_pref_data
from bnn_pref.utils.metrics import alignment_metric
from bnn_pref.utils.type import Q1, Q2, Q2D, SD, D, Q
from bnn_pref.utils.utils import get_gaussian_vector, tile_first_dim
from scripts.run_mcmc import BradleyTerry, QueryWithResponse, generate_pref_data


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    # check RLHF paper
    data_kw = cfg["data"]
    mcmc_kw = cfg["mcmc"]
    dist = BradleyTerry()

    # * generate true params + preference data
    seed = int(datetime.now().timestamp()) if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed)
    key, key1, key2 = jr.split(key, 3)
    true_reward_D = get_gaussian_vector(key1, dim=data_kw["n_feats"], normalize=True)
    features_Q2D, response_Q1 = generate_pref_data(key2, true_reward_D, **data_kw)
    data = QueryWithResponse(features_Q2D, response_Q1)
    potential = partial(dist.potential, data=data)

    # generate a 2D pdf grid of bradley terry
    lim = 3
    X_RR, Y_RR = jnp.mgrid[-lim:lim:100j, -lim:lim:100j]
    pos_RR2 = jnp.stack([X_RR, Y_RR], axis=-1)
    Z_RR = jax.vmap(jax.vmap(potential))(pos_RR2)
    print(pos_RR2.shape)
    print(Z_RR.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(X_RR, Y_RR, Z_RR)
    plt.colorbar()
    plt.scatter(true_reward_D[0], true_reward_D[1], color="red", label="True Reward")

    plt.xlabel("Param 1")
    plt.ylabel("Param 2")
    plt.title("Bradley-Terry logpdf")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
