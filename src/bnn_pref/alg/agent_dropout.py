from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from jaxtyping import Array, Float, Int, Scalar

from bnn_pref.alg.agent_utils import (
    Agent,
    DropoutTrainState,
    bt_loss_fn,
    compute_acq_value,
    get_sgd_nsteps,
    run_sgd,
)
from bnn_pref.alg.data_buffer import QueryBuffer
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.network import RewardNet, count_params
from bnn_pref.utils.type import QueryData, unpackable_dataclass


@unpackable_dataclass
class DropoutBeliefState:
    ts: DropoutTrainState
    t: int


def init_model(
    key,
    model: RewardNet,
    tx: optax.GradientTransformation,
    traj_shape: Tuple[int, ...],  # batch-less shape like (T, D)
) -> DropoutTrainState:
    """create trainstate for a single model"""
    dummy_input = jnp.ones((1, 2, *traj_shape))
    key, param_key, dropout_key = jr.split(key, 3)
    params = model.init(param_key, dummy_input, train=False)["params"]
    ts = DropoutTrainState.create(
        apply_fn=model.apply, params=params, tx=tx, dropout_key=dropout_key
    )
    return ts


class DropoutAgent(Agent):
    def __init__(
        self,
        model: RewardNet,
        traj_shape: Tuple[int, ...],
        learning_rate: float,
        n_models: int,
        max_buffer_size: int = 100,
        l2_reg: float = 0.0,
        niters_init: int = 1,
        niters_update: int = 1,
        batch_size: int = 32,
        chunk_size: int = 64,
        use_vmap: bool = True,  # for training update_bel in {init,update}_bel
        acq: str = "disagreement",
        verbose: bool = False,
    ):
        self.traj_shape = traj_shape
        self.n_models = n_models
        self.model = model
        self.opt = optax.adam(learning_rate)
        self.l2_reg = l2_reg
        self.niters_init = niters_init
        self.niters_update = niters_update
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.use_vmap = use_vmap
        self.max_buffer_size = max_buffer_size
        assert acq in ["disagreement", "infogain", "entropy"]
        self.acq = acq
        self.buffer: QueryBuffer = QueryBuffer.create(
            self.max_buffer_size, self.traj_shape
        )
        self.verbose = verbose

        # * prepare ensemble predictors
        def pred_return(
            key,
            ts: DropoutTrainState,
            x: Float[Array, "T D"],
            train: bool = False,
        ) -> Scalar:
            x = rearrange(x, "T D -> 1 T D")
            params = {"params": ts.params}
            ret = self.model.apply(
                params,
                x,
                method=self.model.predict_traj_return,
                train=train,
                rngs={"dropout": key},
            ).squeeze(0)
            return ret

        self.pred_return = pred_return

    @staticmethod
    def get_hydra_config(alg_cfg):
        # follow sgd.yaml config
        return {
            "acq": alg_cfg["acq"],
            "learning_rate": alg_cfg["learning_rate"],
            # init
            "niters_init": alg_cfg["niters_init"],
            "batch_size": alg_cfg["bs"],
            "l2_reg": alg_cfg["l2_reg"],
            # update
            "niters_update": alg_cfg["niters_update"],
            # ensembling
            "n_models": alg_cfg["M"],
            "chunk_size": alg_cfg["chunk_size"],
            "use_vmap": alg_cfg["use_vmap"],
            "max_buffer_size": alg_cfg["max_buffer_size"],
        }

    def get_alg_info(self):
        return {
            "param_count": self.param_count,
        }

    # @partial(jax.jit, static_argnames=["self"])
    def init_bel(self, key, warmup_data: QueryData) -> DropoutBeliefState:
        key, key_model = jr.split(key)
        ts = init_model(key_model, self.model, self.opt, self.traj_shape)
        self.ensemble_param_count = count_params(ts.params)
        self.param_count = self.ensemble_param_count

        niters = get_sgd_nsteps(self.niters_init, len(warmup_data))
        if niters > 0:
            self.buffer = self.buffer.add_samples(warmup_data)
            key, key_sgd = jr.split(key, 2)
            warm_ts, _ = run_sgd(
                key_sgd,
                ts,
                dataset=warmup_data,
                loss_fn=bt_loss_fn,
                niters=niters,
                batch_size=self.batch_size,
                l2_reg=self.l2_reg,
                get_param_trace=False,
                n_models=1,
                use_dropout=True,
                use_vmap=self.use_vmap,
                verbose=self.verbose,
            )
        else:
            warm_ts = ts

        bel = DropoutBeliefState(ts=warm_ts, t=0)
        return bel

    def update_bel(
        self, key, bel: DropoutBeliefState, batch: QueryData
    ) -> DropoutBeliefState:
        """Train on all queries in the buffer."""
        self.buffer = self.buffer.add_samples(batch)
        ds = self.buffer.get_all()

        niters = get_sgd_nsteps(self.niters_update, len(ds))
        bs = min(self.batch_size, len(ds))
        key, key_sgd = jr.split(key, 2)
        new_ts, _ = run_sgd(
            key_sgd,
            bel.ts,
            dataset=ds,
            loss_fn=bt_loss_fn,
            niters=niters,
            batch_size=bs,
            l2_reg=self.l2_reg,
            get_param_trace=False,
            n_models=1,
            use_dropout=True,
            use_vmap=self.use_vmap,
        )
        bel = bel.replace(ts=new_ts, t=bel.t + 1)
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
            fn = partial(self.pred_return, key, bel.ts, train=True)
            ret_N = jax.lax.map(fn, env.items_NTD, batch_size=self.chunk_size)
            return _, ret_N

        logits_NM = rearrange(
            jax.lax.scan(scan_ts, init=None, xs=jr.split(key_dropout, M))[1],
            "M N -> N M",
        )

        # * compute info gain for each query
        def map_step(idx: int) -> Scalar:
            inds_2 = env.get_pref_indices(idx)
            logits_M2 = rearrange(logits_NM[inds_2], "K M -> M K", K=2)
            logprobs_M2 = jax.nn.log_softmax(logits_M2, axis=1)
            value = compute_acq_value(logprobs_M2, self.acq)  # scalar
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
            fn = partial(self.pred_return, key, bel.ts, train=True)
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

    @staticmethod
    def load_reward_model(
        key,
        cfg: Dict,
        traj_shape: Tuple[int, ...],
        ckpt_fp: str,
    ) -> Callable:
        """
        Load reward model from checkpoint.
        Args:
            cfg: hydra config
            traj_shape: (N, D)
            fp: checkpoint file path, e.g. f'{ckpts_dir}/{task_name}_{alg}_al={is_al}'
        Returns:
            reward_fn: reward function
        """

        ckptr = ocp.PyTreeCheckpointer()
        sharding = jax.sharding.PositionalSharding(jax.local_devices())

        alg_cfg = cfg["do"]
        model = RewardNet(alg_cfg["hidden_sizes"], dropout_prob=alg_cfg["dropout_prob"])
        key, key_init = jr.split(key)
        opt = optax.adam(alg_cfg["learning_rate"])
        dummy_item = init_model(key_init, model, opt, traj_shape)
        dummy_item = dummy_item.replace(dropout_key=jr.key_data(dummy_item.dropout_key))
        dummy_items = jax.tree.map(lambda x: jax.device_put(x, sharding), dummy_item)
        restore_kw = {
            "restore_args": ocp.checkpoint_utils.construct_restore_args(
                dummy_items, jax.tree.map(lambda _: sharding, dummy_items)
            )
        }
        ts = ckptr.restore(ckpt_fp, item=dummy_items, **restore_kw)

        key, *key_dropout = jr.split(key, cfg["do"]["M"] + 1)

        def reward_fn(obs: Float[Array, "T D"]) -> Float[Array, "M T"]:
            # (T,D -> M,T)
            obs = rearrange(obs, "T D -> 1 T D")
            apply_fn = lambda key, obs: ts.apply_fn(
                {"params": ts.params},
                obs,
                method=model.predict_traj_rewards,
                train=True,
                rngs={"dropout": key},
            )
            out_M1T = jax.vmap(apply_fn, in_axes=(0, None))(jnp.array(key_dropout), obs)
            return out_M1T.squeeze(1)

        return reward_fn
