import os

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
import logging
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from flax.training.train_state import TrainState

from bnn_pref.alg.agent_utils import run_gradient_descent
from bnn_pref.data import dataset_creators
from bnn_pref.data.ekf_env import retrieve
from bnn_pref.utils.metrics import compute_accuracy_nn, compute_logpdf_nn
from bnn_pref.utils.network import RewardNet
from bnn_pref.utils.utils import get_random_seed

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    seed = get_random_seed() if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed)
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
    print_sgd_cfg(seed, cfg, length=T, n_feats=D)

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
    losses = metrics["loss"]
    print(f"Final training loss: {losses[-1]:.4f}")

    # Evaluate on test set
    pref_predictor = partial(pref_predictor, final_ts.params)
    test_acc = compute_accuracy_nn(pref_predictor, test_prefs)
    test_logpdf = compute_logpdf_nn(pref_predictor, test_prefs)
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Test avg_ll: {test_logpdf:.2f}")


def print_sgd_cfg(seed, cfg, length=None, n_feats=None):
    data_kw = cfg["data"]
    task_kw = cfg["task"]
    ekf_kw = cfg["ekf"]

    n_queries = data_kw["n_queries"]
    length = length if length is not None else task_kw["length"]
    n_feats = n_feats if n_feats is not None else task_kw["n_feats"]

    n_iterates = ekf_kw["cls"]["n_iterates"]
    batch_size = ekf_kw["cls"]["batch_size"]
    lr = ekf_kw["learning_rate"]

    if task_kw["ds_type"] == "synthetic":
        # todo fix this fhat thing
        task_str = f"{task_kw['ds_type']}: f={task_kw['f']}, fhat={task_kw['fhat']} (fhat ignored for ekf runs)"
    else:
        task_str = f"{task_kw['ds_type']}: {task_kw['name']}"

    print(
        f"Seed: {seed}\n"
        f"Data:\n"
        f"  {task_str}\n"
        f"  N={data_kw['n_demos']}, Q={n_queries}, T={length}, D={n_feats}\n"
        f"SGD:\n"
        f"  n_iterates={n_iterates}, batch_size={batch_size}, lr={lr}"
    )


if __name__ == "__main__":
    main()
