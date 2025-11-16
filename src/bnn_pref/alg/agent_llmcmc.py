from functools import partial
from typing import Dict, Tuple

import blackjax
import distrax as dtx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from einops import rearrange
from flax import linen as nn
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Int, Scalar

from bnn_pref.alg.agent_utils import (
    Agent,
    bt_loss_fn,
    compute_disagreement,
    compute_info_gain,
    get_sgd_niters,
    run_sgd,
)
from bnn_pref.alg.data_buffer import QueryBuffer
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.network import RewardNet, count_params
from bnn_pref.utils.type import QueryData, unpackable_dataclass


@unpackable_dataclass
class LMCMCBeliefState:
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
    params = model.init(param_key, dummy_input, deterministic=True)["params"]
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return ts


def mcmc_belief_update(
    key,
    model: RewardNet,
    params: Dict,  # {"params": actual_params}
    data: QueryData,
    n_particles: int,
    # mcmc hyperparameters
    n_warmups: int,
    n_steps: int,
) -> Float[Array, "M last_layer_dim"]:
    assert (n_steps - n_warmups) % n_particles == 0
    every = (n_steps - n_warmups) // n_particles
    fixed_params = model.get_fixed_params(params)
    last_params = model.get_last_layer_params(params)
    last_params_flat, last_params_unraveler = ravel_pytree(last_params)

    def bnn_logjoint_lastlayer(
        last_params_flat,
        fixed_params,
        last_unraveler,
        data: QueryData,
        model: RewardNet,
    ):
        x, y = data  # (B, 2, T, D), (B, 2)
        params = model.recombine_params(last_params_flat, fixed_params, last_unraveler)
        logits = model.apply(params, x, deterministic=True)  # (B, 2)
        logits_corr = logits[:, 1]
        y_corr = y[:, 1]

        logprior = dtx.Normal(0.0, 1.0).log_prob(last_params_flat).sum()
        loglikeli = dtx.Bernoulli(logits=logits_corr).log_prob(y_corr).sum()

        log_joint = logprior + loglikeli
        return log_joint

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    key, key_warmup, key_samples = jr.split(key, 3)

    potential = partial(
        bnn_logjoint_lastlayer,
        fixed_params=fixed_params,
        last_unraveler=last_params_unraveler,
        data=data,
        model=model,
    )
    adapt = blackjax.window_adaptation(blackjax.nuts, potential)
    (state, parameters), _ = adapt.run(key_warmup, last_params_flat, n_warmups)

    kernel = blackjax.nuts(potential, **parameters).step
    states = inference_loop(key_samples, kernel, state, n_steps)
    sampled_params = states.position  # (n_steps, last_layer_dim)

    ll_params = sampled_params[n_warmups::every]

    return ll_params  # (ensemble_size, last_layer_dim)


class LMCMCAgent(Agent):
    def __init__(
        self,
        model: nn.Module,
        traj_shape: Tuple[int, ...],
        learning_rate: float,
        n_models: int,
        max_buffer_size: int = 100,
        l2_reg: float = 0.0,
        niters_init: int = 1,
        niters_update: int = 1,
        num_warmup: int = 100,
        num_steps: int = 1000,
        batch_size: int = 32,
        chunk_size: int = 64,
        use_vmap: bool = True,  # for training update_bel in {init,update}_bel
        acq: str = "disagreement",
        update_all: bool = True,
    ):
        self.traj_shape = traj_shape
        self.n_models = n_models
        self.model: RewardNet = model
        self.opt = optax.adam(learning_rate)
        self.l2_reg = l2_reg
        self.niters_init = niters_init
        self.niters_update = niters_update
        self.num_warmup = num_warmup
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.use_vmap = use_vmap
        self.max_buffer_size = max_buffer_size
        assert acq in ["disagreement", "infogain"]
        self.acq = acq
        self.update_all = update_all
        self.buffer: QueryBuffer = QueryBuffer.create(
            self.max_buffer_size, self.traj_shape
        )

        # * prepare ensemble predictors
        def pred_return(
            params: Dict,  # {"params": actual_params}
            x: Float[Array, "T D"],
            deterministic: bool = True,
        ) -> Scalar:
            x = rearrange(x, "T D -> 1 T D")
            ret = self.model.apply(
                params,
                x,
                method=self.model.predict_traj_return,
                deterministic=deterministic,
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
            "update_all": alg_cfg["update_all"],
            "niters_update": alg_cfg["niters_update"],
            "num_warmup": alg_cfg["num_warmup"],
            "num_steps": alg_cfg["num_steps"],
            # ensembling
            "n_models": alg_cfg["M"],
            "chunk_size": alg_cfg["chunk_size"],
            "use_vmap": alg_cfg["use_vmap"],
            "max_buffer_size": alg_cfg["max_buffer_size"],
        }

    def get_alg_info(self):
        return {
            "param_count": self.param_count,
            "last_layer_param_count": self.last_layer_param_count,
        }

    # @partial(jax.jit, static_argnames=["self"])
    def init_bel(self, key, warmup_data: QueryData) -> LMCMCBeliefState:
        key, key_model = jr.split(key)
        ts = init_model(key_model, self.model, self.opt, self.traj_shape)
        self.fixed_params = self.model.get_fixed_params({"params": ts.params})
        last_params = self.model.get_last_layer_params({"params": ts.params})
        _, self.last_params_unraveler = ravel_pytree(last_params)
        self.last_layer_param_count = count_params(last_params)
        self.param_count = count_params(ts.params)

        self.buffer = self.buffer.add_samples(warmup_data)

        niters = get_sgd_niters(self.niters_init, len(warmup_data))

        key, key_sgd, key_mcmc = jr.split(key, 3)
        warm_ts, _ = run_sgd(
            key_sgd,
            ts,
            dataset=warmup_data,
            loss_fn=bt_loss_fn,
            has_aux=True,
            niters=niters,
            batch_size=self.batch_size,
            l2_reg=self.l2_reg,
            get_param_trace=False,
            n_models=1,
            use_dropout=False,
            use_vmap=self.use_vmap,
        )

        particles = mcmc_belief_update(
            key=key_mcmc,
            model=self.model,
            params={"params": warm_ts.params},
            data=warmup_data,
            n_particles=self.n_models,
            n_warmups=self.num_warmup,
            n_steps=self.num_steps,
        )  # (M, last_layer_dim)

        bel = LMCMCBeliefState(ts=warm_ts, particles=particles, t=0)
        return bel

    def update_bel(
        self, key, bel: LMCMCBeliefState, batch: QueryData
    ) -> LMCMCBeliefState:
        """Train on all queries in the buffer."""
        key, key_mcmc = jr.split(key, 2)
        self.buffer = self.buffer.add_samples(batch)

        particles = mcmc_belief_update(
            key=key_mcmc,
            model=self.model,
            params={"params": bel.ts.params},
            data=self.buffer.get_all(),
            n_particles=self.n_models,
            n_warmups=self.num_warmup,
            n_steps=self.num_steps,
        )

        bel = bel.replace(particles=particles, t=bel.t + 1)
        return bel

    @partial(jax.jit, static_argnames=["self", "env"])
    def compute_next_query(
        self,
        key,
        bel: LMCMCBeliefState,
        env: PreferenceEnv,
        pool_idxes_Q: Int[Array, "Q"],
    ) -> int:
        """
        active learning: greedily compute query that maximizes acquisition function
        """
        M = self.n_models

        particles = bel.particles  # (M, last_layer_dim)

        # * precompute logits for all items
        def scan_ts(_, llparam: Float[Array, "last_layer_dim"]):
            param = self.model.recombine_params(
                llparam, self.fixed_params, self.last_params_unraveler
            )  # {"params": actual_params}
            fn = partial(self.pred_return, param)
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
        bel: LMCMCBeliefState,
        items_NTD: Float[Array, "N T D"],
        query_idxs_Q2: Int[Array, "Q 2"],
    ) -> Float[Array, "Q 2"]:
        """
        compute predictive distribution for all items in query pool
        """
        M = self.n_models
        particles = bel.particles  # (M, last_layer_dim)

        # * precompute logits for all items
        def scan_ts(_, llparam: Float[Array, "last_layer_dim"]):
            param = self.model.recombine_params(
                llparam, self.fixed_params, self.last_params_unraveler
            )  # {"params": actual_params}
            fn = partial(self.pred_return, param)
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
