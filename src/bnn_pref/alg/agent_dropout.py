from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from einops import rearrange
from flax import linen as nn
from jaxtyping import Array, Float, Int

from bnn_pref.alg.agent_utils import (
    Agent,
    DropoutTrainState,
    bt_loss_fn,
    compute_disagreement,
    compute_info_gain,
    run_sgd,
)
from bnn_pref.alg.data_buffer import QueryBuffer
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.network import count_params
from bnn_pref.utils.type import QueryData, unpackable_dataclass


@unpackable_dataclass
class DropoutBeliefState:
    ts: DropoutTrainState
    buffer: QueryBuffer
    t: int


def init_model(
    key,
    model: nn.Module,
    tx: optax.GradientTransformation,
    traj_shape: Tuple[int, ...],  # batch-less shape like (T, D)
) -> DropoutTrainState:
    """create trainstate for a single model"""
    dummy_input = jnp.ones((1, 2, *traj_shape))
    key, param_key, dropout_key = jr.split(key, 3)
    params = model.init(param_key, dummy_input, deterministic=True)["params"]
    ts = DropoutTrainState.create(
        apply_fn=model.apply, params=params, tx=tx, dropout_key=dropout_key
    )
    return ts


class DropoutAgent(Agent):
    def __init__(
        self,
        model: nn.Module,
        opt: optax.GradientTransformation,
        traj_shape: Tuple[int, ...],
        n_models: int,
        max_buffer_size: int = 100,
        l2_reg: float = 0.0,
        niters_init: int = 1,
        niters_update: int = 1,
        batch_size: int = 32,
        chunk_size: int = 64,
        use_vmap: bool = True,  # for training update_bel in {init,update}_bel
        acq: str = "disagreement",
        update_all: bool = True,
    ):
        self.traj_shape = traj_shape
        self.n_models = n_models
        self.model = model
        self.opt = opt
        self.l2_reg = l2_reg
        self.niters_init = niters_init
        self.niters_update = niters_update
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.use_vmap = use_vmap
        self.max_buffer_size = max_buffer_size
        assert acq in ["disagreement", "infogain"]
        self.acq = acq
        self.update_all = update_all

        # * prepare ensemble predictors
        def pred_return(
            key,
            ts: DropoutTrainState,
            x: Float[Array, "T D"],
            deterministic: bool,
        ) -> Float[Array, " "]:
            x = rearrange(x, "T D -> 1 T D")
            params = {"params": ts.params}
            ret = self.model.apply(
                params,
                x,
                method=self.model.predict_traj_return,
                deterministic=deterministic,
                rngs={"dropout": key},
            ).squeeze(0)
            return ret

        self.pred_return = pred_return

    @staticmethod
    def get_hydra_config(do_cfg):
        # follow sgd.yaml config
        return {
            "acq": do_cfg["acq"],
            # init
            "niters_init": do_cfg["niters_init"],
            "batch_size": do_cfg["bs"],
            "l2_reg": do_cfg["l2_reg"],
            # update
            "update_all": do_cfg["update_all"],
            "niters_update": do_cfg["niters_update"],
            # ensembling
            "n_models": do_cfg["M"],
            "chunk_size": do_cfg["chunk_size"],
            "use_vmap": do_cfg["use_vmap"],
            "max_buffer_size": do_cfg["max_buffer_size"],
        }

    # @partial(jax.jit, static_argnames=["self"])
    def init_bel(self, key, warmup_data: QueryData) -> DropoutBeliefState:
        key, key_model = jr.split(key)
        ts = init_model(key_model, self.model, self.opt, self.traj_shape)
        self.ensemble_param_count = count_params(ts.params)
        self.param_count = self.ensemble_param_count

        buffer = QueryBuffer.create(self.max_buffer_size, self.traj_shape)
        buffer = buffer.add_samples(warmup_data)
        bel = DropoutBeliefState(ts=ts, buffer=buffer, t=0)

        niters = (
            self.niters_init
            if self.niters_init > 0
            else int(len(warmup_data) * jnp.abs(self.niters_init))
        )
        key, key_sgd = jr.split(key, 2)
        warm_ts, warm_metrics = run_sgd(
            key_sgd,
            bel.ts,
            dataset=warmup_data,
            loss_fn=bt_loss_fn,
            has_aux=True,
            niters=niters,
            batch_size=self.batch_size,
            l2_reg=self.l2_reg,
            get_param_trace=False,
            n_models=1,
            use_dropout=True,
        )

        bel = bel.replace(ts=warm_ts)
        return bel

    def update_bel(
        self, key, bel: DropoutBeliefState, batch: QueryData
    ) -> DropoutBeliefState:
        key, key_update = jr.split(key)
        if self.update_all:
            return self.update_bel_all(key_update, bel, batch)
        else:
            return self.update_bel_most_recent(key_update, bel, batch)
        # return self.update_bel_most_recent(key, bel, batch)

    # @partial(jax.jit, static_argnames=["self"])
    def update_bel_all(
        self,
        key,
        bel: DropoutBeliefState,
        batch: QueryData,
    ) -> DropoutBeliefState:
        """Train on all queries in the buffer."""
        new_buffer = bel.buffer.add_samples(batch)
        bel = bel.replace(buffer=new_buffer)
        ds = bel.buffer.get_all()

        niters = (
            self.niters_update
            if self.niters_update > 0
            else int(len(ds) * jnp.abs(self.niters_update))
        )
        key, key_sgd = jr.split(key, 2)
        ts, _ = run_sgd(
            key_sgd,
            bel.ts,
            dataset=ds,
            loss_fn=bt_loss_fn,
            has_aux=True,
            niters=niters,
            batch_size=self.batch_size,
            l2_reg=self.l2_reg,
            get_param_trace=False,
            n_models=1,
            use_dropout=True,
        )
        bel = bel.replace(ts=ts, t=bel.t + 1)
        return bel

    # @partial(jax.jit, static_argnames=["self"])
    def update_bel_most_recent(
        self,
        key,
        bel: DropoutBeliefState,
        batch: QueryData,
    ) -> DropoutBeliefState:
        """Train on the most recent query."""
        new_buffer = bel.buffer.add_samples(batch)
        bel = bel.replace(buffer=new_buffer)
        ds = bel.buffer.get_newest_n(n=1)

        def train_step(ts, batch: QueryData):
            key, key_dropout = jr.split(ts.dropout_key)
            contexts_B2TD, labels_B2 = batch.contexts, batch.labels

            def parameterized_loss(params):
                logits_B2 = ts.apply_fn(
                    {"params": params},
                    contexts_B2TD,
                    deterministic=False,
                    rngs={"dropout": key_dropout},
                )
                return bt_loss_fn(params, logits_B2, labels_B2, self.l2_reg)

            grad_fn = jax.value_and_grad(parameterized_loss, has_aux=True)
            (loss, _), grads = grad_fn(ts.params)
            ts = ts.apply_gradients(grads=grads)
            ts = ts.replace(dropout_key=key)
            return ts

        ts = train_step(bel.ts, ds)

        bel = bel.replace(ts=ts, t=bel.t + 1)
        return bel

    @partial(jax.jit, static_argnames=["self", "env"])
    def compute_next_query(
        self,
        key,
        bel: DropoutBeliefState,
        env: PreferenceEnv,
        pool_idxes_Q: Int[Array, "Q"],
    ) -> int:
        """
        active learning: greedily compute query that maximizes acquisition function
        """
        M = self.n_models
        key, key_dropout = jr.split(key)

        # * precompute logits for all items
        def scan_ts(_, key):
            fn = partial(self.pred_return, key, bel.ts, deterministic=False)
            ret_N = jax.lax.map(fn, env.items_NTD, batch_size=self.chunk_size)
            return _, ret_N

        logits_NM = rearrange(
            jax.lax.scan(scan_ts, init=None, xs=jr.split(key_dropout, M))[1],
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
        key,
        bel: DropoutBeliefState,
        items_NTD: Float[Array, "N T D"],
        query_idxs_Q2: Int[Array, "Q 2"],
    ) -> Float[Array, "Q 2"]:
        """
        compute predictive distribution for all items in query pool
        """
        M = self.n_models
        key, key_dropout = jr.split(key)

        # * precompute logits for all items, assume ts lead dimension is M
        def scan_ts(_, key):
            fn = partial(self.pred_return, key, bel.ts, deterministic=False)
            ret_N = jax.lax.map(fn, items_NTD, batch_size=self.chunk_size)
            return _, ret_N

        logits_NM = rearrange(
            jax.lax.scan(scan_ts, init=None, xs=jr.split(key_dropout, M))[1],
            "M N -> N M",
        )

        # * compute posterior predictive
        logits_QM2 = rearrange(logits_NM[query_idxs_Q2], "Q K M -> Q M K", K=2)
        llik_QM2 = jax.nn.log_softmax(logits_QM2, axis=2)
        llik_Q2 = jax.nn.logsumexp(llik_QM2, axis=1) - jnp.log(M)
        prob_Q2 = jnp.exp(llik_Q2)
        return prob_Q2
