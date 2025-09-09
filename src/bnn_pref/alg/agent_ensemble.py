from functools import partial
from typing import NamedTuple, Tuple

import flax
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from einops import rearrange
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import lax
from jaxtyping import Array, Float, Int, Scalar

from bnn_pref.alg.agent_utils import (
    Agent,
    bt_loss_fn,
    compute_disagreement,
    compute_info_gain,
    run_sgd,
)
from bnn_pref.alg.data_buffer import QueryBuffer
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.network import RewardNet, count_params
from bnn_pref.utils.type import QueryData, unpackable_dataclass


@unpackable_dataclass
class EnsembleBeliefState:
    ts: TrainState
    buffer: QueryBuffer
    t: int


def init_model(
    key,
    model: nn.Module,
    tx: optax.GradientTransformation,
    traj_shape: Tuple[int, ...],  # batch-less shape like (T, D)
) -> TrainState:
    """create trainstate for a single model"""
    dummy_input = jnp.ones((1, 2, *traj_shape))
    params = model.init(key, dummy_input)["params"]
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return ts


class DeepEnsemble(Agent):
    def __init__(
        self,
        model: nn.Module,
        opt: optax.GradientTransformation,
        traj_shape: Tuple[int, ...],
        n_models: int,
        max_buffer_size: int = 100,
        l2_reg: float = 0.0,
        niters: int = 1000,
        batch_size: int = 32,
        chunk_size: int = 64,
        use_vmap: bool = True,  # for training update_bel in {init,update}_bel
        n_epochs: int = 0,
        acq: str = "disagreement",
    ):
        self.traj_shape = traj_shape
        self.n_models = n_models
        self.model = model
        self.opt = opt
        self.l2_reg = l2_reg
        self.niters = niters
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.use_vmap = use_vmap
        self.max_buffer_size = max_buffer_size
        assert n_epochs >= 0, "n_epochs must be non-negative"
        self.n_epochs = n_epochs
        assert acq in ["disagreement", "infogain"]
        self.acq = acq

        # * prepare ensemble predictors
        def pred_return(ts: TrainState, x: Float[Array, "T D"]) -> Float[Array, " "]:
            x = rearrange(x, "T D -> 1 T D")
            params = {"params": ts.params}
            ret = self.model.apply(
                params, x, method=self.model.predict_traj_return
            ).squeeze(0)
            return ret

        self.pred_return = pred_return

    @staticmethod
    def get_hydra_config(sgd_cfg):
        # follow sgd.yaml config
        return {
            "acq": sgd_cfg["acq"],
            # init
            "niters": sgd_cfg["niters"],
            "batch_size": sgd_cfg["bs"],
            "l2_reg": sgd_cfg["l2_reg"],
            # update
            "n_epochs": sgd_cfg["n_epochs"],
            # ensembling
            "n_models": sgd_cfg["M"],
            "chunk_size": sgd_cfg["chunk_size"],
            "use_vmap": sgd_cfg["use_vmap"],
            "max_buffer_size": sgd_cfg["max_buffer_size"],
        }

    # @partial(jax.jit, static_argnames=["self"])
    def init_bel(self, key, warmup_data: QueryData) -> EnsembleBeliefState:
        key, *keys_init = jr.split(key, 1 + self.n_models)
        ts = jax.vmap(init_model, in_axes=(0, None, None, None))(
            jnp.array(keys_init),
            self.model,
            self.opt,
            self.traj_shape,
        )
        self.ensemble_param_count = count_params(ts.params)
        self.param_count = self.ensemble_param_count // self.n_models

        buffer = QueryBuffer.create(self.max_buffer_size, self.traj_shape)
        buffer = buffer.add_samples(warmup_data)
        bel = EnsembleBeliefState(ts=ts, buffer=buffer, t=0)

        warm_ts, warm_metrics = run_sgd(
            key,
            bel.ts,
            dataset=warmup_data,
            loss_fn=bt_loss_fn,
            has_aux=True,
            niters=self.niters,
            batch_size=self.batch_size,
            l2_reg=self.l2_reg,
            get_param_trace=False,
            n_models=self.n_models,
            split_datastream=True,
            use_dropout=False,
        )

        bel = bel.replace(ts=warm_ts)

        return bel

    def update_bel(
        self, key, bel: EnsembleBeliefState, batch: QueryData
    ) -> EnsembleBeliefState:
        return self.update_bel_all(key, bel, batch)
        # return self.update_bel_most_recent(key, bel, batch)
        # return self.update_bel_old(key, bel, batch)

    # @partial(jax.jit, static_argnames=["self"])
    def update_bel_all(
        self,
        key,
        bel: EnsembleBeliefState,
        batch: QueryData,
    ) -> EnsembleBeliefState:
        """Train on all queries in the buffer."""
        new_buffer = bel.buffer.add_samples(batch)
        bel = bel.replace(buffer=new_buffer)

        ds = bel.buffer.get_all()
        key, key_sgd = jr.split(key, 2)
        ts, _ = run_sgd(
            key_sgd,
            bel.ts,
            dataset=ds,
            loss_fn=bt_loss_fn,
            has_aux=True,
            niters=self.niters,
            batch_size=self.batch_size,
            l2_reg=self.l2_reg,
            get_param_trace=False,
            n_models=self.n_models,
            split_datastream=True,
            use_dropout=False,
        )
        bel = bel.replace(ts=ts, t=bel.t + 1)
        return bel

    # @partial(jax.jit, static_argnames=["self"])
    def update_bel_most_recent(
        self,
        key,
        bel: EnsembleBeliefState,
        batch: QueryData,
    ) -> EnsembleBeliefState:
        """Train on the most recent query."""
        new_buffer = bel.buffer.add_samples(batch)
        bel = bel.replace(buffer=new_buffer)
        ds = bel.buffer.get_newest_n(n=1)

        # def train_fn(ts, batch: QueryData):
        #     grad_fn = jax.value_and_grad(bt_loss_fn, has_aux=True)
        #     (loss, _), grads = grad_fn(ts.params, ts, batch, self.l2_reg)
        #     ts = ts.apply_gradients(grads=grads)
        #     return ts

        def train_fn(ts, batch: QueryData):
            contexts_B2TD, labels_B2 = batch.contexts, batch.labels
            logits_B2 = ts.apply_fn({"params": ts.params}, contexts_B2TD)
            grad_fn = jax.value_and_grad(bt_loss_fn, has_aux=True)
            (loss, _), grads = grad_fn(ts.params, logits_B2, labels_B2, self.l2_reg)
            ts = ts.apply_gradients(grads=grads)
            return ts

        if self.use_vmap:
            grad_fn = jax.vmap(train_fn, in_axes=(0, None))  # vmap over ts
            ts = grad_fn(bel.ts, ds)
        else:
            grad_fn = partial(train_fn, batch=ds)
            ts = jax.lax.map(grad_fn, bel.ts)

        bel = bel.replace(ts=ts, t=bel.t + 1)
        return bel

    # @partial(jax.jit, static_argnames=["self"])
    def update_bel_old(
        self,
        key,  # currently unused, static shape not compatible with scan epoch shuffle
        bel: EnsembleBeliefState,
        batch: QueryData,
    ) -> EnsembleBeliefState:
        """
        Old implementation, do not use.

        Training cases: bs = 1 for all cases
        n_epochs=0: 1 query   1 sgd step:    niters=1
        n_epochs=1: Q queries 1 sgd epoch:   niters=Q
        n_epochs>1: Q queries M sgd epochs:  niters=Q * M

        n_valids masking for batch manager is required for jit w/ dynamically sized buffer
        """
        new_buffer = bel.buffer.add_samples(batch)
        bel = bel.replace(buffer=new_buffer)
        key, key_sgd = jr.split(key)
        bs = 1  # always train on 1 query at a time

        if self.n_epochs == 0:
            # train only on the newest query, only 1 sgd step
            ds = bel.buffer.get_newest_n(n=1)

            def train_fn(ts, batch: QueryData):
                grad_fn = jax.value_and_grad(bt_loss_fn, has_aux=True)
                (loss, _), grads = grad_fn(ts.params, ts, batch, self.l2_reg)
                ts = ts.apply_gradients(grads=grads)
                return ts

            if self.use_vmap:
                grad_fn = jax.vmap(train_fn, in_axes=(0, None))  # vmap over ts
                ts = grad_fn(bel.ts, ds)
            else:
                grad_fn = partial(train_fn, batch=ds)
                ts = jax.lax.map(grad_fn, bel.ts)

            bel = bel.replace(ts=ts, t=bel.t + 1)
            return bel

        else:
            ds = bel.buffer.get_all()
            n_valids = len(ds)
            # jax.debug.print("n_valids: {n_valids}", n_valids=n_valids)

            def train_fn(
                key,
                ts: TrainState,
                dataset: QueryData,
                n_valids: int,
            ) -> TrainState:
                """
                lax.scan for epochs, lax.while for iterations within each epoch
                one iteration per valid query.
                """
                key, *keys_epochs = jr.split(key, 1 + self.n_epochs)
                keys_epochs = jnp.array(keys_epochs)

                def scan_fn(carry, key_epoch):
                    ts, epoch = carry
                    ds_e = dataset  # todo shuffle only the valid queries

                    def body_fn(carry: Tuple[TrainState, int]):
                        ts, itr = carry

                        batch = QueryData(
                            contexts=lax.dynamic_slice_in_dim(ds_e.contexts, itr, bs),
                            labels=lax.dynamic_slice_in_dim(ds_e.labels, itr, bs),
                        )
                        grad_fn = jax.value_and_grad(bt_loss_fn, has_aux=True)
                        (loss, _), grads = grad_fn(ts.params, ts, batch, self.l2_reg)
                        ts = ts.apply_gradients(grads=grads)
                        return (ts, itr + 1)

                    def cond_fn(carry: Tuple[TrainState, int]):
                        ts, itr = carry
                        return itr < n_valids

                    init_itr_val = (ts, 0)  # (ts, itr)
                    ts, itr = jax.lax.while_loop(cond_fn, body_fn, init_itr_val)
                    return (ts, epoch + 1), itr

                init_epoch_val = (ts, 0)  # (ts, epoch)
                (ts, epochs), itrs = jax.lax.scan(
                    scan_fn, init=init_epoch_val, xs=keys_epochs
                )
                # jax.debug.print("epochs: {epochs}", epochs=epochs)
                # jax.debug.print("itrs: {itrs}", itrs=itrs)
                return ts

            sgd_fn = partial(train_fn, dataset=ds, n_valids=n_valids)

            if self.use_vmap:  # vmap over ts
                ts = jax.vmap(sgd_fn, in_axes=(None, 0))(key_sgd, bel.ts)
            else:
                ts = jax.lax.map(partial(sgd_fn, key_sgd), bel.ts)

            bel = bel.replace(ts=ts, t=bel.t + 1)
            return bel

    @partial(jax.jit, static_argnames=["self", "env"])
    def compute_next_query(
        self,
        key,  # for compatibility with trainer
        bel: EnsembleBeliefState,
        env: PreferenceEnv,
        pool_idxes_Q: Int[Array, "Q"],
    ) -> int:
        """
        active learning: greedily compute query that maximizes acquisition function
        """
        M = self.n_models

        # * precompute logits for all items
        def scan_ts(_, ts_single):
            fn = partial(self.pred_return, ts_single)
            ret_N = jax.lax.map(fn, env.items_NTD, batch_size=self.chunk_size)
            return _, ret_N

        logits_NM = rearrange(
            jax.lax.scan(scan_ts, init=None, xs=bel.ts)[1],
            "M N -> N M",
        )

        # * compute info gain for each query
        def map_step(idx: int) -> Float[Array, " "]:
            inds_2 = env.get_pref_indices(idx)
            logits_M2 = rearrange(logits_NM[inds_2], "K M -> M K", K=2)
            logprobs_M2 = jax.nn.log_softmax(logits_M2, axis=1)
            if self.acq == "infogain":
                value = compute_info_gain(logprobs_M2, M)
            elif self.acq == "disagreement":
                value = compute_disagreement(logprobs_M2)
            return value

        values_Q = jax.lax.map(map_step, pool_idxes_Q, batch_size=self.chunk_size)
        query_idx = jnp.argmax(values_Q)
        return query_idx

    @partial(jax.jit, static_argnames=["self"])
    def compute_postpred(
        self,
        key,  # for compatibility with trainer
        bel: EnsembleBeliefState,
        items_NTD: Float[Array, "N T D"],
        query_idxs_Q2: Int[Array, "Q 2"],
    ) -> Float[Array, "Q 2"]:
        """
        compute predictive distribution for all items in query pool
        """
        M = self.n_models

        # * precompute logits for all items, assume ts lead dimension is M
        def scan_ts(_, ts_single):
            fn = partial(self.pred_return, ts_single)
            ret_N = jax.lax.map(fn, items_NTD, batch_size=self.chunk_size)
            return _, ret_N

        logits_NM = rearrange(
            jax.lax.scan(scan_ts, init=None, xs=bel.ts)[1],
            "M N -> N M",
        )

        # * compute posterior predictive
        logits_QM2 = rearrange(logits_NM[query_idxs_Q2], "Q K M -> Q M K", K=2)
        llik_QM2 = jax.nn.log_softmax(logits_QM2, axis=2)
        llik_Q2 = jax.nn.logsumexp(llik_QM2, axis=1) - jnp.log(M)
        prob_Q2 = jnp.exp(llik_Q2)
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
