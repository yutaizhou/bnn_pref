import logging
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import arviz as az
import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from bnn_pref.data.make_synthetic import generate_synthetic_trajs, make_synthetic_data
from bnn_pref.data.pref_utils import BradleyTerry
from bnn_pref.utils.utils import get_gaussian_vector, get_random_seed


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    # check RLHF paper
    data_kw = cfg["data"]
    task_kw = cfg["task"]
    mcmc_kw = cfg["mcmc"]
    dist = BradleyTerry()

    # * generate true params + preference data
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)
    key, key_data = jr.split(key)
    data = make_synthetic_data(key_data, cfg)
    train_prefs, test_prefs = data["train_prefs"], data["test_prefs"]
    true_param_D, true_reward_fn = data["true_param"], data["true_reward_fn"]
    potential = partial(dist.potential, data=train_prefs)

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
    plt.scatter(true_param_D[0], true_param_D[1], color="red", label="True Reward")

    plt.xlabel("Param 1")
    plt.ylabel("Param 2")
    plt.title("Bradley-Terry logpdf")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
