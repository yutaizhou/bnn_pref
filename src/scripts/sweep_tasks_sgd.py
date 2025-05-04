import logging
from collections import defaultdict
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
import optax
from flax.training.train_state import TrainState

from bnn_pref.alg.agent_utils import bt_loss_fn, run_gradient_descent
from bnn_pref.data import dataset_creators
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import MeanStd, compute_acc_nn, compute_logpdf_nn
from bnn_pref.utils.network import RewardNet, count_params
from bnn_pref.utils.type import QueryData
from bnn_pref.utils.utils import get_random_seed


def run_sgd(key, cfg, data_dict, env):
    # check RLHF paper
    data_cfg = cfg["data"]
    sgd_cfg = cfg["sgd"]

    niters = sgd_cfg["cls"]["niters"]
    batch_size = sgd_cfg["cls"]["bs"]
    lr = sgd_cfg["learning_rate"]

    train_prefs, test_prefs = data_dict["train_prefs"], data_dict["test_prefs"]

    # Initialize RewardNet
    key, model_key = jr.split(key)
    model = RewardNet(sgd_cfg["hidden_sizes"], sgd_cfg["n_splits"])
    dummy_input = train_prefs.queries_Q2TD[:1]
    params = model.init(model_key, dummy_input)["params"]

    # Create optimizer and training state
    optimizer = optax.adam(lr)
    ts = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    train_data = QueryData(
        train_prefs.queries_Q2TD,
        jax.nn.one_hot(train_prefs.responses_Q1, num_classes=2),
    )

    def pref_predictor(params, queries_Q2TD):
        return model.apply({"params": params}, queries_Q2TD)

    # Run training
    key, train_key = jr.split(key)
    final_ts, metrics = run_gradient_descent(
        train_key,
        ts,
        loss_fn=bt_loss_fn,
        has_aux=True,
        dataset=train_data,
        niters=niters,
        batch_size=batch_size,
    )

    # Evaluate on test set
    pref_predictor = partial(pref_predictor, final_ts.params)
    test_acc = compute_acc_nn(pref_predictor, test_prefs)
    test_logpdf = compute_logpdf_nn(pref_predictor, test_prefs)
    losses = metrics["loss"]
    results = {
        "test_acc": test_acc,
        "test_logpdf": test_logpdf,
        "train_losses": losses,
    }
    metadata = {
        "param_count": count_params(final_ts.params),
    }
    return results, metadata


@hydra.main(version_base=None, config_name="configPref", config_path="../cfg")
def main(cfg):
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
    stats = defaultdict(lambda: defaultdict(dict))
    data_cfg = cfg["data"]
    sgd_cfg = cfg["sgd"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    batch_size = sgd_cfg["bs"]
    niters = sgd_cfg["niters"]
    print(
        f"Seed: {seed} x {cfg['seeds']}\n"
        f"Data:\n"
        f"  Train/Test: {nq_train}/{nq_test}\n"
        f"SGD:\n"
        f"  init: bs={batch_size}, niters={niters}\n"
    )

    for task in tasks:
        print(f"{task}: ")
        key, key_data, *key_seeds = jr.split(key, 2 + cfg["seeds"])
        data_dict = dataset_creators[cfg["task"]["ds_type"]](key_data, cfg)
        train_prefs = data_dict["train_prefs"]
        env = PreferenceEnv(
            X=train_prefs.queries_Q2TD,
            Y=jax.nn.one_hot(train_prefs.responses_Q1.squeeze(), num_classes=2),
        )
        for is_al in [False, True]:
            # * update cfg
            new_cfg = hydra.compose(
                "config",
                overrides=[f"task={task}", f"sgd.active={is_al}"],
            )

            # * run
            seeds = jnp.array(key_seeds)
            vmap_run_experiment = jax.vmap(run_sgd, in_axes=(0, None, None, None))
            start_time = datetime.now()
            res_m, metadata_m = vmap_run_experiment(seeds, new_cfg, data_dict, env)
            duration = (datetime.now() - start_time).total_seconds()

            stat = {
                "test_acc": MeanStd(res_m["test_acc"]),
                "test_logpdf": MeanStd(res_m["test_logpdf"]),
            }
            stats[task][is_al] = stat

            print(
                f"  acc: {stat['test_acc'].mean:.2%} ± {stat['test_acc'].std:.1%}, "
                f"logpdf: {stat['test_logpdf'].mean:.2f} ± {stat['test_logpdf'].std:.1f}, "
                f"({metadata_m['param_count'][0]:,d}) "
                f"({duration:.1f}s)"
            )


if __name__ == "__main__":
    main()
