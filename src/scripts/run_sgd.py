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

from bnn_pref.alg.agent_utils import bt_loss_fn, run_gradient_descent
from bnn_pref.data import dataset_creators
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import compute_acc_nn, compute_logpdf_nn
from bnn_pref.utils.network import RewardNet
from bnn_pref.utils.print_utils import print_sgd_cfg
from bnn_pref.utils.type import QueryData
from bnn_pref.utils.utils import get_random_seed

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)
    data_cfg = cfg["data"]
    ekf_cfg = cfg["ekf"]
    task_cfg = cfg["task"]

    niters = ekf_cfg["cls"]["niters"]
    batch_size = ekf_cfg["cls"]["batch_size"]
    lr = ekf_cfg["learning_rate"]

    # * generate true params + preference data
    output = dataset_creators[task_cfg["ds_type"]](key, cfg)
    train_prefs, test_prefs = output["train_prefs"], output["test_prefs"]
    Q, _, T, D = train_prefs.queries_Q2TD.shape
    print_sgd_cfg(seed, cfg, length=T, n_feats=D)

    # Initialize RewardNet
    key, model_key = jr.split(key)
    model = RewardNet(ekf_cfg["hidden_sizes"], ekf_cfg["n_splits"])
    dummy_input = train_prefs.queries_Q2TD[:1]
    params = model.init(model_key, dummy_input)["params"]

    # Create optimizer and training state
    optimizer = optax.adam(lr)
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
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

    # Print training metrics
    losses = metrics["loss"]
    print(f"Final training loss: {losses[-1]:.4f}")

    # Evaluate on test set
    pref_predictor = partial(pref_predictor, final_ts.params)
    test_acc = compute_acc_nn(pref_predictor, test_prefs)
    test_logpdf = compute_logpdf_nn(pref_predictor, test_prefs)
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Test avg_ll: {test_logpdf:.2f}")


if __name__ == "__main__":
    main()
