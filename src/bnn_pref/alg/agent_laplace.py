import sys
from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, Int, Scalar
from laplax import laplace
from laplax.eval.pushforward import get_dist_state
from loguru import logger

from bnn_pref.alg.agent_utils import (
    Agent,
    bt_loss_fn,
    compute_disagreement,
    compute_info_gain,
    get_sgd_nsteps,
    run_sgd,
)
from bnn_pref.alg.data_buffer import QueryBuffer
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.network import ParamsDict, RewardNet, count_params, perturb_params
from bnn_pref.utils.type import QueryData, unpackable_dataclass

logger.remove()
logger.add(sys.stderr, level="INFO")


@unpackable_dataclass
class LaplaceBeliefState:
    ts: TrainState  # SGD initialized model params
    particles: Array  # (ensemble_size, last_layer_dim)
    t: int


def init_model(
    key,
    model: RewardNet,
    tx: optax.GradientTransformation,
    traj_shape: Tuple[int, ...],  # batch-less shape like (T, D)
) -> TrainState:
    """create trainstate for a single model"""
    dummy_input = jnp.ones((1, 2, *traj_shape))
    key, param_key = jr.split(key, 2)
    params = model.init(param_key, dummy_input, train=False)["params"]
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return ts


class BatchedLoader:
    def __init__(
        self,
        x: Float[Array, "N 2 T D"],
        y: Float[Array, "N 2"],
        batch_size: int,
    ):
        N = len(x)
        self.x = x
        self.y = y
        self.batch_idxes = []
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            self.batch_idxes.append((i, end_idx))

    def __iter__(self):
        for bgn, end in self.batch_idxes:
            yield (self.x[bgn:end], self.y[bgn:end])


def laplace_belief_update(
    key,
    model_def: RewardNet,
    params: ParamsDict,  # {"params": actual_params}
    data: QueryData,  # all queries seen so far
    n_particles: int,
    # * laplace hyperparameter
    curv_type: str = "full",
    prior_prec: float = 1000.0,
    laplace_bs: int = 32,
) -> ParamsDict:  # leading axis is M
    def model_fn(input, params):
        """
        vmap_over_data=True,
            input: (2, T, ...)
        otherwise,
            input: (B, 2, T, ...)

        output logits: (2,)
        """
        input = jnp.expand_dims(input, axis=0)  # (2,T,D) -> (1, 2, T, D)
        logits = model_def.apply(
            params,
            input,
            train=False,
        )  # (1, 2)
        logits = logits.squeeze(0)
        return logits  # (2,)

    def perform_laplace():
        x, y = data.contexts, data.labels  # (N, 2, T, D), (N, 2)
        N = len(x)
        laplace_fn = partial(
            laplace,
            model_fn=model_fn,
            params=params,
            loss_fn="cross_entropy",
            curv_type=curv_type,
            vmap_over_data=True,
        )

        if N <= laplace_bs:
            posterior_fn, _ = laplace_fn(
                data=(x, y),
                num_curv_samples=N,
                num_total_samples=N,
            )
        else:
            posterior_fn, _ = laplace_fn(
                data=BatchedLoader(x, y, laplace_bs),
                num_curv_samples=laplace_bs,
                num_total_samples=N,
            )
        return posterior_fn

    posterior_fn = perform_laplace()

    prior_arguments = {"prior_prec": prior_prec}

    key, key_particles = jr.split(key, 2)
    dist_state = get_dist_state(
        mean_params=params,
        model_fn=model_fn,
        posterior_state=posterior_fn(prior_arguments, loss_scaling_factor=1.0),
        linearized=False,
        num_samples=n_particles,
        key=key_particles,
    )
    particles = []
    for i in range(n_particles):
        # list of dicts with arrays of shape (param)
        particle = dist_state["get_weight_samples"](i)
        particles.append(particle)
    particles = jax.tree_util.tree_map(
        lambda *x: jnp.stack(x, axis=0), *particles
    )  # dict of arrays with leading axis (M, param)
    return particles  # particles["params"]["pw_mlp"]["Dense_0"]


class LaplaceAgent(Agent):
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
        prior_prec: float = 0.2,
        curv_type: str = "full",
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
        self.prior_prec = prior_prec
        self.curv_type = curv_type
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.use_vmap = use_vmap
        self.max_buffer_size = max_buffer_size
        assert acq in ["disagreement", "infogain"]
        self.acq = acq
        self.buffer: QueryBuffer = QueryBuffer.create(
            self.max_buffer_size, self.traj_shape
        )
        self.verbose = verbose

        # * prepare ensemble predictors
        def pred_return(
            params: ParamsDict,
            x: Float[Array, "T D"],
            train: bool = False,
        ) -> Scalar:
            x = jnp.expand_dims(x, axis=0)
            ret = self.model.apply(
                params,
                x,
                method=self.model.predict_traj_return,
                train=train,
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
            "prior_prec": alg_cfg["prior_prec"],
            "curv_type": alg_cfg["curv_type"],
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
    def init_bel(self, key, warmup_data: QueryData) -> LaplaceBeliefState:
        key, key_model = jr.split(key)
        ts = init_model(key_model, self.model, self.opt, self.traj_shape)
        self.param_count = count_params(ts.params)

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
                use_dropout=False,
                use_vmap=self.use_vmap,
                verbose=self.verbose,
            )

            key, key_laplace = jr.split(key, 2)
            new_particles = laplace_belief_update(
                key=key_laplace,
                model_def=self.model,
                params={"params": warm_ts.params},
                data=warmup_data,
                n_particles=self.n_models,
                curv_type=self.curv_type,
                prior_prec=self.prior_prec,
            )  # particles["params"]["pw_mlp"]["Dense_0"]
        else:
            warm_ts = ts
            key, key_perturb = jr.split(key, 2)
            new_particles, _ = perturb_params(
                key=key_perturb,
                params={"params": warm_ts.params},
                perturb_std=0.1,
                n_particles=self.n_models,
            )
        bel = LaplaceBeliefState(ts=warm_ts, particles=new_particles, t=0)
        return bel

    def update_bel(
        self, key, bel: LaplaceBeliefState, batch: QueryData
    ) -> LaplaceBeliefState:
        """Train on all queries in the buffer."""
        key, key_sgd, key_laplace = jr.split(key, 3)
        self.buffer = self.buffer.add_samples(batch)
        ds = self.buffer.get_all()

        niters = get_sgd_nsteps(self.niters_update, len(ds))
        bs = min(self.batch_size, len(ds))
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
            use_dropout=False,
            use_vmap=self.use_vmap,
        )

        new_particles = laplace_belief_update(
            key=key_laplace,
            model_def=self.model,
            params={"params": new_ts.params},
            data=ds,
            n_particles=self.n_models,
            curv_type=self.curv_type,
            prior_prec=self.prior_prec,
        )  # particles["params"]["pw_mlp"]["Dense_0"]
        bel = bel.replace(ts=new_ts, particles=new_particles, t=bel.t + 1)
        return bel

    @partial(jax.jit, static_argnames=["self", "env"])
    def compute_next_query(
        self,
        key,
        bel: LaplaceBeliefState,
        env: PreferenceEnv,
        pool_idxes_Q: Int[Array, "Q"],
    ) -> int:
        """
        active learning: greedily compute query that maximizes acquisition function
        """
        particles = bel.particles  # particles["params"]["pw_mlp"]["Dense_0"]

        # * precompute logits for all items
        def scan_ts(_, particle: ParamsDict):
            fn = partial(self.pred_return, particle, train=False)
            ret_N = jax.lax.map(fn, env.items_NTD, batch_size=self.chunk_size)
            return _, ret_N

        logits_NM = rearrange(
            jax.lax.scan(scan_ts, init=None, xs=particles)[1],
            "M N -> N M",
        )

        # * compute info gain for each query
        def map_step(idx: int) -> Scalar:
            inds_2 = env.get_pref_indices(idx)
            logits_M2 = rearrange(logits_NM[inds_2], "K M -> M K", K=2)
            logprobs_M2 = jax.nn.log_softmax(logits_M2, axis=1)
            if self.acq == "infogain":
                value = compute_info_gain(logprobs_M2)
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
        bel: LaplaceBeliefState,
        items_NTD: Float[Array, "N T D"],
        query_idxs_Q2: Int[Array, "Q 2"],
    ) -> Float[Array, "Q 2"]:
        """
        compute predictive distribution for all items in query pool
        """
        M = self.n_models
        particles = bel.particles  # particles["params"]["pw_mlp"]["Dense_0"]

        # * precompute logits for all items
        def scan_ts(_, particle: ParamsDict):
            fn = partial(self.pred_return, particle, train=False)
            ret_N = jax.lax.map(fn, items_NTD, batch_size=self.chunk_size)
            return _, ret_N

        logits_NM = rearrange(
            jax.lax.scan(scan_ts, init=None, xs=particles)[1],
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

        alg_cfg = cfg["laplace"]
        model = RewardNet(alg_cfg["hidden_sizes"])
        key, key_init = jr.split(key)
        opt = optax.adam(alg_cfg["learning_rate"])
        dummy_ts = init_model(key_init, model, opt, traj_shape)
        params = {"params": dummy_ts.params}
        params = jax.tree.map(
            lambda x: jnp.broadcast_to(x[None, ...], (alg_cfg["M"],) + x.shape),
            params,
        )  # ParamDict with leading axis M
        dummy_item = LaplaceBeliefState(ts=dummy_ts, particles=params, t=0)

        dummy_items = jax.tree.map(lambda x: jax.device_put(x, sharding), dummy_item)
        restore_kw = {
            "restore_args": ocp.checkpoint_utils.construct_restore_args(
                dummy_items, jax.tree.map(lambda _: sharding, dummy_items)
            )
        }

        bel = ckptr.restore(ckpt_fp, item=dummy_items, **restore_kw)
        ts = bel.ts
        params = bel.particles  # ParamsDict with leading axis M

        def reward_fn(obs: Float[Array, "T D"]) -> Float[Array, "M T"]:
            # (T,D -> M,T)
            apply_fn = partial(ts.apply_fn, method=model.predict_traj_rewards)
            out_MT = jax.vmap(apply_fn, in_axes=(0, None))(params, obs)
            # return out_MT.mean(axis=0)
            return out_MT

        return reward_fn
