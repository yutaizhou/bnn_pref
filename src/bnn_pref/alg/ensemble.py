from functools import partial
from typing import NamedTuple, Tuple

import flax
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from einops import rearrange
from flax import linen as nn
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, Int, Scalar

from bnn_pref.alg.agent_utils import Agent, bt_loss_fn, run_gradient_descent
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.network import RewardNet, count_params
from bnn_pref.utils.type import QueryData, unpackable_dataclass


@unpackable_dataclass
class QueryBufferState:
    contexts: Float[Array, "Q 2 T D"]
    labels: Float[Array, "Q 2"]
    ptr: int = 0
    max_size: int = 2000

    def __len__(self) -> int:
        return self.ptr


class QueryBuffer:
    """
    Store all queries received so far, for sgd training
    todo - see flax for ways to make this more jax native
    """

    @staticmethod
    def create_buffer(max_size: int, traj_shape: Tuple[int, ...]) -> QueryBufferState:
        buffer = QueryBufferState(
            contexts=jnp.empty((max_size, 2, *traj_shape)),
            labels=jnp.empty((max_size, 2)),
            ptr=0,
            max_size=max_size,
        )
        return buffer

    @staticmethod
    def update(state: QueryBufferState, new: QueryData) -> QueryBufferState:
        """Update the buffer with new query data."""
        assert new.contexts.ndim == state.contexts.ndim, "contexts must have same ndim"
        assert new.labels.ndim == state.labels.ndim, "labels must have same ndim"
        n_new = new.contexts.shape[0]
        new_contexts = jax.lax.dynamic_update_slice(
            state.contexts, new.contexts, (state.ptr, 0, 0, 0)
        )
        new_labels = jax.lax.dynamic_update_slice(
            state.labels, new.labels, (state.ptr, 0)
        )
        new_ptr = state.ptr + n_new

        state = state.replace(
            contexts=new_contexts,
            labels=new_labels,
            ptr=new_ptr,
        )
        # Optionally, add a runtime check (outside jit) for overflow
        # assert state.ptr <= state.max_size, "buffer overflow"
        return state

    @staticmethod
    def get_all(state: QueryBufferState) -> QueryData:
        """Get all queries from the buffer."""
        return QueryData(
            contexts=state.contexts[: state.ptr],
            labels=state.labels[: state.ptr],
        )

    @staticmethod
    def get_recent_n(state: QueryBufferState, n: int) -> QueryData:
        """Get the last n queries from the buffer."""
        return QueryData(
            contexts=state.contexts[state.ptr - n : state.ptr],
            labels=state.labels[state.ptr - n : state.ptr],
        )


@unpackable_dataclass
class EnsembleBeliefState:
    ts: TrainState
    buffer: QueryBufferState


def init_model(
    key,
    model: nn.Module,
    tx: optax.GradientTransformation,
    traj_shape: Tuple[int, ...],
) -> TrainState:
    dummy_input = jnp.ones((1, 2, *traj_shape))
    params = model.init(key, dummy_input)["params"]
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return ts


class DeepEnsemble(Agent):
    def __init__(
        self,
        model: nn.Module,
        opt: optax.GradientTransformation,
        n_models: int,
        traj_shape: Tuple[int, ...],
        max_buffer_size: int = 2000,
        l2_reg: float = 0.0,
        niters: int = 1000,
        batch_size: int = 32,
        chunk_size: int = 64,
        use_vmap: bool = True,  # for training update_bel in {init,update}_bel
    ):
        self.n_models = n_models
        self.model = model
        self.opt = opt
        self.l2_reg = l2_reg
        self.niters = niters
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.use_vmap = use_vmap
        self.max_buffer_size = max_buffer_size
        self.traj_shape = traj_shape

    @staticmethod
    def get_hydra_config(sgd_cfg):
        # follow sgd.yaml config
        return {
            # training
            "niters": sgd_cfg["niters"],
            "batch_size": sgd_cfg["bs"],
            "l2_reg": sgd_cfg["l2_reg"],
            # ensembling
            "n_models": sgd_cfg["M"],
            "chunk_size": sgd_cfg["chunk_size"],
            "use_vmap": sgd_cfg["use_vmap"],
        }

    @partial(jax.jit, static_argnames=["self"])
    def init_bel(self, key, warmup_data: QueryData) -> EnsembleBeliefState:
        # todo precompute logits for all items
        key, *keys_init = jr.split(key, 1 + self.n_models)
        ts = jax.vmap(init_model, in_axes=(0, None, None, None))(
            jnp.array(keys_init),
            self.model,
            self.opt,
            self.traj_shape,
        )
        self.ensemble_param_count = count_params(ts.params)
        self.param_count = self.ensemble_param_count // self.n_models

        buffer = QueryBuffer.create_buffer(self.max_buffer_size, self.traj_shape)
        buffer = QueryBuffer.update(buffer, warmup_data)
        state = EnsembleBeliefState(ts=ts, buffer=buffer)

        sgd_fn = partial(
            run_gradient_descent,
            loss_fn=bt_loss_fn,
            has_aux=True,
            dataset=QueryBuffer.get_all(state.buffer),
            niters=self.niters,
            batch_size=self.batch_size,
            l2_reg=self.l2_reg,
        )  # remaining args: (key, ts)

        # same datastream for all models
        key, key_sgd = jr.split(key)
        if self.use_vmap:
            sgd_fn = jax.vmap(sgd_fn, in_axes=(None, 0))  # vmap over ts
            warm_ts, warm_metrics = sgd_fn(key_sgd, state.ts)
        else:
            sgd_fn = partial(sgd_fn, key_sgd)
            warm_ts, warm_metrics = jax.lax.map(sgd_fn, state.ts)

        # different datastreams for each model
        # key, *key_sgd = jr.split(key, 1 + self.n_models)
        # run_sgd_fn = jax.vmap(run_sgd_fn, in_axes=(0, 0))  # vmap (key, ts)
        # warm_ts, warm_metrics = run_sgd_fn(jnp.array(key_sgd), ts)

        state = state.replace(ts=warm_ts)

        return state

    @partial(jax.jit, static_argnames=["self"])
    def update_bel(
        self, state: EnsembleBeliefState, batch: QueryData
    ) -> EnsembleBeliefState:
        """
        Training cases
        1 sgd step 1 query:      niters=1, batch_size=1
        1 sgd epoch Q queries:   niters=Q, batch_size=1
        M SGD epochs, Q queries: niters=Q * M, batch_size=1
        """
        # batch = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), batch)
        new_buffer = QueryBuffer.update(state.buffer, batch)
        state = state.replace(buffer=new_buffer)

        def train_step(ts: TrainState, batch: QueryData):
            grad_fn = jax.value_and_grad(bt_loss_fn, has_aux=True)
            (loss, _), grads = grad_fn(ts.params, ts, batch, self.l2_reg)
            return ts.apply_gradients(grads=grads), loss

        if self.use_vmap:
            grad_fn = jax.vmap(train_step, in_axes=(0, None))  # vmap over ts
            ts, loss = grad_fn(state.ts, batch)
        else:
            grad_fn = partial(train_step, batch=batch)
            ts, loss = jax.lax.map(grad_fn, state.ts)
        state = state.replace(ts=ts)
        return state

    @partial(jax.jit, static_argnames=["self", "env"])
    def acquire_next_query(
        self,
        key,
        bel: EnsembleBeliefState,
        env: PreferenceEnv,
        pool_idxes_Q: Int[Array, "Q"],
    ) -> int:
        """
        active learning: greedily compute query that maximizes ensemble prediction var
        """

        def predict_fn(ts: TrainState, x: Float[Array, "1 T D"]) -> Float[Array, " "]:
            """unbatched ts, and output"""
            ret = ts.apply_fn(
                {"params": ts.params}, x, method=self.model.predict_traj_return
            ).squeeze(0)
            return ret

        # * precompute logits for all items
        if self.use_vmap:
            # vmap over ts, run over items sequentially
            fn = jax.vmap(predict_fn, in_axes=(0, None))
            items_N1TD = rearrange(env.items_NTD, "N T D -> N 1 T D")
            fn = partial(fn, bel.ts)
            logits_NM = jax.lax.map(fn, items_N1TD, batch_size=self.chunk_size)
        else:
            # run over ts sequentially, run over items sequentially
            items_N1TD = rearrange(env.items_NTD, "N T D -> N 1 T D")

            def scan_ts(_, ts_single):
                fn = partial(predict_fn, ts_single)
                ret_N = jax.lax.map(fn, items_N1TD, batch_size=self.chunk_size)
                return _, ret_N

            logits_NM = rearrange(jax.lax.scan(scan_ts, None, bel.ts)[1], "M N -> N M")

        def map_step(idx):
            inds_2 = env.get_pref_indices(idx)
            logits_M2 = rearrange(logits_NM[inds_2], "K M -> M K", K=2)
            probs_M2 = jnp.exp(jax.nn.log_softmax(logits_M2, axis=1))
            pred_M = jnp.argmax(probs_M2, axis=1)
            value = jnp.var(pred_M, axis=0)
            return value

        values_Q = jax.lax.map(map_step, pool_idxes_Q, batch_size=self.chunk_size)

        query_idx = jnp.argmax(values_Q)
        return query_idx

    @partial(jax.jit, static_argnames=["self"])
    def compute_predictive(
        self,
        ts: TrainState,
        items_NTD: Float[Array, "N T D"],
        query_idxs_Q2: Int[Array, "Q 2"],
    ) -> Float[Array, "Q 2"]:
        """
        compute predictive distribution for all items in query pool
        """

        # * prepare ensemble predictors
        def predict_fn(ts: TrainState, x: Float[Array, "1 T D"]) -> Float[Array, " "]:
            ret = ts.apply_fn(
                {"params": ts.params}, x, method=self.model.predict_traj_return
            ).squeeze(0)
            return ret

        # * precompute logits for all items
        if self.use_vmap:
            fn = jax.vmap(predict_fn, in_axes=(0, None))
            items_N1TD = rearrange(items_NTD, "N T D -> N 1 T D")
            fn = partial(fn, ts)
            logits_NM = jax.lax.map(fn, items_N1TD, batch_size=self.chunk_size)
        else:
            items_N1TD = rearrange(items_NTD, "N T D -> N 1 T D")

            def scan_ts(_, ts_single):
                fn = partial(predict_fn, ts_single)
                ret_N = jax.lax.map(fn, items_N1TD, batch_size=self.chunk_size)
                return _, ret_N

            logits_NM = rearrange(jax.lax.scan(scan_ts, None, ts)[1], "M N -> N M")

        logits_QM2 = rearrange(logits_NM[query_idxs_Q2], "Q K M -> Q M K", K=2)

        # * compute predictive distributions
        llik_QM2 = jax.nn.log_softmax(logits_QM2, axis=2)
        llik_Q2 = jax.nn.logsumexp(llik_QM2, axis=1) - jnp.log(self.n_models)
        prob_Q2 = jnp.exp(llik_Q2)
        # prob_Q2 = jnp.exp(llik_QM2).mean(1)
        return prob_Q2


if __name__ == "__main__":
    import ipdb

    def init_model(key, model, input_shape, tx):
        dummy_input = jnp.ones((1, *input_shape))
        params = model.init(key, dummy_input)["params"]
        ts = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
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
    tx = optax.adam(3e-4)
    ts = jax.vmap(init_model, in_axes=(0, None, None, None))(
        keys_model, model, input_shape, tx
    )
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

    # * Model prediction
    predict_step_vj = jax.jit(jax.vmap(predict_step, in_axes=(0, None)))
    logits_Q2 = predict_step_vj(ts, batch)
    print(logits_Q2.shape)
