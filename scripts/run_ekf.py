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

from bnn_pref.alg.bandit_env import BanditEnvironment
from bnn_pref.alg.ekf_subspace import SubspaceNeuralBandit
from bnn_pref.alg.train_utils import bandit_pipeline, summarize_results
from bnn_pref.data import BradleyTerry, QueryWithResponse, generate_pref_data
from bnn_pref.utils.type import Q1, Q2, Q2D, SD, D, Q
from bnn_pref.utils.utils import (
    alignment_metric,
    get_gaussian_vector,
    get_uniform_vector,
    tile_first_dim,
)

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    # check RLHF paper
    data_kw = cfg["data"]
    bandit_kw = cfg["bandit"]
    dist = BradleyTerry()
    n_feats = data_kw["n_feats"]

    seed = int(datetime.now().timestamp()) if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed=seed)

    # * generate true weights + preference data
    key, key1, key2 = jr.split(key, 3)
    true_reward_D = get_gaussian_vector(key1, dim=n_feats, normalize=True)
    features_Q2D, response_Q1 = generate_pref_data(key2, true_reward_D, **data_kw)
    # data = QueryWithResponse(features_Q2D, response_Q1)

    # * build + run bandit alg
    key, key1, key2 = jr.split(key, 3)

    env = BanditEnvironment(
        key1,
        X=features_Q2D,
        Y=jax.nn.one_hot(response_Q1.squeeze(), num_classes=2),
    )

    rewards_info, bel, bandit = bandit_pipeline(
        key2,
        SubspaceNeuralBandit,
        env,
        n_warmup_obs=bandit_kw["n_warmup_obs"],
        n_trials=bandit_kw["n_trials"],
        bandit_kw=bandit_kw,
    )
    warmup_rewards, rewards_trace, opt_rewards = rewards_info
    rtotal, rstd = summarize_results(warmup_rewards, rewards_trace)

    key, key1 = jr.split(key, 2)
    # todo n_trials beliefs..
    fn = jax.vmap(partial(bandit.predict_rewards, bel.mean[0]))
    logits = fn(features_Q2D)
    probs = jax.nn.softmax(logits, axis=1)
    pred_response_Q = probs.argmax(axis=1)
    acc = jnp.mean(pred_response_Q == response_Q1.squeeze())
    print(acc)


if __name__ == "__main__":
    main()
