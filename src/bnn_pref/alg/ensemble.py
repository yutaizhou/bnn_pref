from functools import partial
from typing import Tuple

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from einops import rearrange
from flax import linen as nn
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Scalar
from sklearn.decomposition import PCA
from tensorflow_probability.substrates import jax as tfp

from bnn_pref.alg.agent_utils import Agent, bt_loss_fn, run_gradient_descent
from bnn_pref.data.ekf_env import retrieve
from bnn_pref.utils.network import RewardNet, count_params
from bnn_pref.utils.type import CARL, unpackable_dataclass
from bnn_pref.utils.utils import vmap_chunked

# # EKF
# @unpackable_dataclass
# class EnsembleBeliefState:
#     t: int


def init_model(key, model, input_shape, tx):
    dummy_input = jnp.ones((1, *input_shape))
    params = model.init(key, dummy_input)["params"]
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return ts


class DeepEnsemble(Agent):
    def __init__(
        self,
        model: nn.Module,
        opt: optax.GradientTransformation,
        n_models: int,
        l2_reg: float = 0.0,
        niters: int = 1000,
        batch_size: int = 32,
    ):
        self.n_models = n_models
        self.model = model
        self.opt = opt
        self.l2_reg = l2_reg
        self.niters = niters
        self.batch_size = batch_size

    def init_bel(self, key, warmup_data: CARL) -> TrainState:
        key, *keys_init = jr.split(key, 1 + self.n_models)
        ts = jax.vmap(init_model, in_axes=(0, None, None, None))(
            jnp.array(keys_init), self.model, warmup_data.contexts.shape[1:], self.opt
        )

        run_sgd_fn = partial(
            run_gradient_descent,
            loss_fn=bt_loss_fn,
            has_aux=True,
            dataset=warmup_data,
            niters=self.niters,
            batch_size=self.batch_size,
            l2_reg=self.l2_reg,
        )

        # same datastream for all models
        key, key_sgd = jr.split(key)
        run_sgd_fn = jax.vmap(run_sgd_fn, in_axes=(None, 0))  # vmap (key, ts)
        warm_ts, warm_metrics = run_sgd_fn(key_sgd, ts)

        # different datastreams for each model
        # key, *key_sgd = jr.split(key, 1 + self.n_models)
        # run_sgd_fn = jax.vmap(run_sgd_fn, in_axes=(0, 0))  # vmap (key, ts)
        # warm_ts, warm_metrics = run_sgd_fn(jnp.array(key_sgd), ts)

        return warm_ts

    def update_bel(self, ts: TrainState, batch: CARL) -> TrainState:
        """
        flax nn needs batch of shape (1, N, ...), hence expand_dims
        """
        batch = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), batch)
        batch = (batch.contexts, batch.labels)

        def train_step(ts, batch):
            grad_fn = jax.value_and_grad(bt_loss_fn, has_aux=True)
            (loss, _), grads = grad_fn(ts.params, ts, batch, self.l2_reg)
            return ts.apply_gradients(grads=grads), loss

        grad_fn = jax.vmap(train_step, in_axes=(0, None))
        ts, loss = grad_fn(ts, batch)
        return ts

    def acquire_next_query(self, key, ts: TrainState, contexts_N2TD) -> int:
        """
        active learning: greedily compute query that maximizes ensemble prediction var
        """
        chunk_size = 32

        def predict_fn(ts: TrainState, contexts):
            contexts = rearrange(contexts, "K T D -> 1 K T D", K=2)
            logits_2 = ts.apply_fn({"params": ts.params}, contexts).squeeze()
            return logits_2

        fn = jax.vmap(predict_fn, in_axes=(0, None))  # over ts
        logits_NM2 = vmap_chunked(
            jax.vmap(partial(fn, ts)),
            contexts_N2TD,
            size=chunk_size,
            fout_shape=(self.n_models, 2),
        )

        probs_NM2 = jax.nn.softmax(logits_NM2, axis=2)
        pred_NM = jnp.argmax(probs_NM2, axis=2)
        values_N = jnp.var(pred_NM, axis=1)
        query_idx = jnp.argmax(values_N)
        return query_idx


if __name__ == "__main__":
    import ipdb

    def init_model(key, model, input_shape):
        dummy_input = jnp.ones((1, *input_shape))
        params = model.init(key, dummy_input)["params"]
        ts = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.adam(3e-4),
        )
        return ts

    def train_step(ts: TrainState, batch) -> Tuple[TrainState, Scalar]:
        """Forward pass and loss computation vectorized across models."""
        contexts_N2TD, labels_N2 = batch

        def loss_fn(params) -> Tuple[Scalar, Float[Array, "N 2"]]:
            logits_N2 = model.apply({"params": params}, contexts_N2TD)
            loss = optax.softmax_cross_entropy(logits_N2, labels_N2).mean()
            return loss, logits_N2

        # Vectorize the loss computation across models
        vgrad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, _), grads = vgrad_fn(ts.params)

        ts = ts.apply_gradients(grads=grads)
        return ts, loss

    def predict_step(ts, batch):
        contexts_Q2TD, labels_Q2 = batch
        logits_Q2 = model.apply({"params": ts.params}, contexts_Q2TD)
        return logits_Q2

    # * Model definition
    key = jr.key(0)
    n_models = 4
    model = RewardNet(hidden_sizes=[32, 32])
    Q, K, T, D = 100, 2, 6, 3
    input_shape = (K, T, D)

    # * data def
    key, key_data = jr.split(key)
    contexts_Q2TD = jr.normal(key_data, shape=(Q, *input_shape))
    labels_Q2 = jax.nn.one_hot(jnp.ones((Q,)), num_classes=K)
    batch = (contexts_Q2TD, labels_Q2)

    # * Model initialization
    key, *keys_model = jr.split(key, 1 + n_models)
    keys_model = jnp.array(keys_model)
    ts = jax.vmap(init_model, in_axes=(0, None, None))(keys_model, model, input_shape)
    print(jax.tree.map(lambda x: x.shape, ts.params))
    train_step_vj = jax.jit(jax.vmap(train_step, in_axes=(0, None)))

    # * Model training
    n_iters = 100
    # for i in range(n_iters):
    #     ts, loss = train_step_vj(ts, batch)
    #     print(loss)

    def scan_step(ts, _):
        ts, loss = train_step_vj(ts, batch)
        return ts, (loss, ts)

    _, (loss, ts) = jax.lax.scan(scan_step, init=ts, length=n_iters)
    print(loss)
    ipdb.set_trace()

    # * Model prediction
    predict_step_vj = jax.jit(jax.vmap(predict_step, in_axes=(0, None)))
    logits_Q2 = predict_step_vj(ts, batch)
    print(logits_Q2.shape)
