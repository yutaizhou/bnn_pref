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
import optax
from flax.training.train_state import TrainState

from bnn_pref.alg.agent_utils import run_gradient_descent
from bnn_pref.data import dataset_creators
from bnn_pref.data.ekf_env import retrieve
from bnn_pref.utils.metrics import compute_accuracy_nn, compute_logpdf_nn
from bnn_pref.utils.network import RewardNet, count_params
from bnn_pref.utils.utils import get_random_seed


def run_experiment(key, cfg):
    # check RLHF paper
    data_kw = cfg["data"]
    ekf_kw = cfg["ekf"]
    task_kw = cfg["task"]

    hidden_sizes = ekf_kw["hidden_sizes"]
    n_iterates = ekf_kw["cls"]["n_iterates"]
    batch_size = ekf_kw["cls"]["batch_size"]
    lr = ekf_kw["learning_rate"]

    # * generate true params + preference data
    output = dataset_creators[task_kw["ds_type"]](key, cfg)
    train_prefs, test_prefs = output["train_prefs"], output["test_prefs"]
    Q, _, T, D = train_prefs.queries_Q2TD.shape

    # Initialize RewardNet
    key, model_key = jr.split(key)
    reward_net = RewardNet(hidden_sizes)
    dummy_input = train_prefs.queries_Q2TD[:1]
    params = reward_net.init(model_key, dummy_input)["params"]

    # Create optimizer and training state
    optimizer = optax.adam(lr)
    ts = TrainState.create(
        apply_fn=reward_net.apply,
        params=params,
        tx=optimizer,
    )

    # Define loss function for preference learning
    def loss_fn(params, batch_idx):
        # Retrieve batch data using the retrieve function
        contexts_batch = retrieve(train_prefs.queries_Q2TD, batch_idx)
        labels_batch = retrieve(train_prefs.responses_Q1, batch_idx)
        logits_B2 = reward_net.apply({"params": params}, contexts_batch)
        labels_B2 = jax.nn.one_hot(labels_batch, num_classes=2)
        loss = optax.softmax_cross_entropy(logits_B2, labels_B2).mean()
        return loss, logits_B2

    def pref_predictor(params, queries_Q2TD):
        return reward_net.apply({"params": params}, queries_Q2TD)

    # Run training
    key, train_key = jr.split(key)
    final_ts, metrics = run_gradient_descent(
        train_key,
        ts,
        loss_fn,
        n_iterates=n_iterates,
        data_size=Q,
        batch_size=batch_size,
        has_aux=True,
    )

    # Print training metrics

    # Evaluate on test set
    pref_predictor = partial(pref_predictor, final_ts.params)
    test_acc = compute_accuracy_nn(pref_predictor, test_prefs)
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


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def run_dimensinality_exp(cfg):
    seed = get_random_seed() if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed)

    tasks = [
        "ogbench",
        "lunar",
        "reacher",
        "cheetah",
        "acrobot",
        "ball",
        "cartpoleSwing",
        "cheetahDMC",
        "hopperHop",
        "lunar",
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
        results, metadata = vmap_run_experiment(jnp.array(subkeys), new_cfg)
        duration = (datetime.now() - start_time).total_seconds()

        stats.append(
            {
                "acc_mean": results["test_acc"].mean(),
                "acc_std": results["test_acc"].std(),
                "logpdf_mean": results["test_logpdf"].mean(),
                "logpdf_std": results["test_logpdf"].std(),
            }
        )

        print(
            f"{task:10}: "
            f"acc = {results['test_acc'].mean():.2%} ± {results['test_acc'].std():.1%}, "
            f"logpdf = {results['test_logpdf'].mean():.2f} ± {results['test_logpdf'].std():.1f}, "
            f"param_count = {metadata['param_count'][0]}, "
            f"Time: {duration:.1f} seconds"
        )


if __name__ == "__main__":
    run_dimensinality_exp()
