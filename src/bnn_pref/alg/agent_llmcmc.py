from functools import partial
from typing import Callable, Dict, Optional, Tuple

import blackjax
import distrax as dtx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Int, Scalar

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
from bnn_pref.utils.network import (
    LastLayerHelpers,
    ParamsDict,
    ParamsFlat,
    RewardNet,
    count_params,
    perturb_params,
)
from bnn_pref.utils.type import QueryData, unpackable_dataclass


@unpackable_dataclass
class LMCMCBeliefState:
    ts: TrainState  # SGD initialized model params
    particles: Array  # (ensemble_size, llayer_dim)
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


def mcmc_belief_update(
    key,
    model: RewardNet,
    params: ParamsDict,
    data: QueryData,
    n_particles: int,  # ensemble size
    # mcmc hyperparameters
    n_warmups: int,
    n_steps: int,
    # optionally starting mcmc from a given state
    initial_particle: Optional[Float[Array, "llayer_dim"]] = None,
) -> ParamsFlat:  # leading axis is M
    fixed_params = LastLayerHelpers.get_frozen_params(params)
    llayer_params = LastLayerHelpers.get_trainable_params(params)
    llayer_params_flat, llayer_unraveler = ravel_pytree(llayer_params)

    if initial_particle is not None:
        llayer_params_flat = initial_particle

    def bnn_logjoint_lastlayer(
        llayer_flat: ParamsFlat,
        fixed_params: ParamsDict,
        llayer_unraveler: Callable,
        data: QueryData,
        model: RewardNet,
    ) -> Scalar:
        x, y = data.contexts, data.labels  # (B, 2, T, D), (B, 2)
        params = LastLayerHelpers.recombine_params(
            llayer_flat, fixed_params, llayer_unraveler
        )
        logits = model.apply(params, x, train=False)  # (B, 2)
        y = jnp.argmax(y, axis=1)  # (B,), y can't be 1-hot for distrax

        logprior = dtx.Normal(0.0, 1.0).log_prob(llayer_flat).sum()
        loglikeli = dtx.Categorical(logits=logits).log_prob(y).sum()

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
        llayer_unraveler=llayer_unraveler,
        data=data,
        model=model,
    )
    adapt = blackjax.window_adaptation(blackjax.nuts, potential)
    (state, parameters), _ = adapt.run(key_warmup, llayer_params_flat, n_warmups)

    kernel = blackjax.nuts(potential, **parameters).step
    states = inference_loop(key_samples, kernel, state, n_steps)
    sampled_params_flat = states.position  # (n_steps, llayer_dim)
    idxes = jnp.linspace(0, n_steps, n_particles, dtype=jnp.int32)
    subsampled_params_flat = sampled_params_flat[idxes]

    return subsampled_params_flat  # (n_particles, llayer_dim)


class LMCMCAgent(Agent):
    def __init__(
        self,
        model: RewardNet,
        traj_shape: Tuple[int, ...],
        learning_rate: float,
        n_models: int,
        max_buffer_size: int = 100,
        batch_size: int = 32,
        l2_reg: float = 0.0,
        niters_init: int = 1,
        niters_update: int = 1,
        # mcmc hyperparameters
        mcmc_warmups_init: int = 500,
        mcmc_warmups_update: int = 20,
        mcmc_steps: int = 1000,
        chunk_size: int = 64,
        use_vmap: bool = True,  # for training update_bel in {init,update}_bel
        acq: str = "disagreement",
        verbose: bool = False,
    ):
        self.traj_shape = traj_shape
        self.n_models = n_models
        self.model = model
        self.opt = optax.adam(learning_rate)
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.niters_init = niters_init
        self.niters_update = niters_update
        # mcmc hyperparameters
        self.mcmc_warmups_init = mcmc_warmups_init
        self.mcmc_warmups_update = mcmc_warmups_update
        self.mcmc_steps = mcmc_steps
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
            params: ParamsDict,  # {"params": actual_params}
            x: Float[Array, "T D"],
            train: bool = False,
        ) -> Scalar:
            x = rearrange(x, "T D -> 1 T D")
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
            "mcmc_warmups_init": alg_cfg["mcmc_warmups_init"],
            "mcmc_warmups_update": alg_cfg["mcmc_warmups_update"],
            "mcmc_steps": alg_cfg["mcmc_steps"],
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
        last_params = LastLayerHelpers.get_trainable_params({"params": ts.params})
        _, self.last_params_unraveler = ravel_pytree(last_params)
        self.last_layer_param_count = count_params(last_params)
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

            key, key_mcmc = jr.split(key, 2)
            new_particles = mcmc_belief_update(
                key=key_mcmc,
                model=self.model,
                params={"params": warm_ts.params},
                data=warmup_data,
                n_particles=self.n_models,
                n_warmups=self.mcmc_warmups_init,
                n_steps=self.mcmc_steps,
                initial_particle=None,
            )  # (M, llayer_dim)
        else:
            warm_ts = ts

            key, key_perturb = jr.split(key, 2)
            llparams = LastLayerHelpers.get_trainable_params({"params": warm_ts.params})
            _, new_particles = perturb_params(
                key=key_perturb,
                params={"params": llparams},
                perturb_std=0.1,
                n_particles=self.n_models,
            )  # (M, llayer_dim)

        self.fixed_params = LastLayerHelpers.get_frozen_params(
            {"params": warm_ts.params}
        )

        bel = LMCMCBeliefState(ts=warm_ts, particles=new_particles, t=0)
        return bel

    def update_bel(
        self, key, bel: LMCMCBeliefState, batch: QueryData
    ) -> LMCMCBeliefState:
        """Train on all queries in the buffer."""
        self.buffer = self.buffer.add_samples(batch)
        ds = self.buffer.get_all()

        key, key_mcmc = jr.split(key, 2)
        bs = min(self.batch_size, len(ds))
        new_particles = mcmc_belief_update(
            key=key_mcmc,
            model=self.model,
            params={"params": bel.ts.params},
            data=ds,
            n_particles=self.n_models,
            n_warmups=self.mcmc_warmups_update,
            n_steps=self.mcmc_steps,
            initial_particle=bel.particles[-1],
        )

        bel = bel.replace(particles=new_particles, t=bel.t + 1)
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
        particles = bel.particles  # (M, llayer_dim)

        # * precompute logits for all items
        def scan_ts(_, llparam: ParamsFlat):
            param = LastLayerHelpers.recombine_params(
                llparam, self.fixed_params, self.last_params_unraveler
            )  # {"params": actual_params}
            fn = partial(self.pred_return, param, train=False)
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
        bel: LMCMCBeliefState,
        items_NTD: Float[Array, "N T D"],
        query_idxs_Q2: Int[Array, "Q 2"],
    ) -> Float[Array, "Q 2"]:
        """
        compute predictive distribution for all items in query pool
        """
        M = self.n_models
        particles = bel.particles  # (M, llayer_dim)

        # * precompute logits for all items
        def scan_ts(_, llparam: Float[Array, "llayer_dim"]):
            param = LastLayerHelpers.recombine_params(
                llparam, self.fixed_params, self.last_params_unraveler
            )  # {"params": actual_params}
            fn = partial(self.pred_return, param, train=False)
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

        alg_cfg = cfg["llmcmc"]
        model = RewardNet(alg_cfg["hidden_sizes"])
        key, key_init = jr.split(key)
        opt = optax.adam(alg_cfg["learning_rate"])
        dummy_ts = init_model(key_init, model, opt, traj_shape)
        params = {"params": dummy_ts.params}
        unraveler = ravel_pytree(params)[1]

        train_params = LastLayerHelpers.get_trainable_params(params)
        _, unraveler = ravel_pytree(train_params)
        param_count = count_params(train_params)
        dummy_item = LMCMCBeliefState(
            ts=dummy_ts,
            particles=jnp.zeros((alg_cfg["M"], param_count)),
            t=0,
        )

        dummy_items = jax.tree.map(lambda x: jax.device_put(x, sharding), dummy_item)
        restore_kw = {
            "restore_args": ocp.checkpoint_utils.construct_restore_args(
                dummy_items, jax.tree.map(lambda _: sharding, dummy_items)
            )
        }

        bel = ckptr.restore(ckpt_fp, item=dummy_items, **restore_kw)
        ts = bel.ts
        frozen_params = LastLayerHelpers.get_frozen_params({"params": ts.params})
        particles_flat = bel.particles
        params = jax.vmap(LastLayerHelpers.recombine_params, in_axes=(0, None, None))(
            particles_flat, frozen_params, unraveler
        )

        def reward_fn(obs: Float[Array, "T D"]) -> Float[Array, "M T"]:
            # (T,D -> M,T)
            apply_fn = partial(ts.apply_fn, method=model.predict_traj_rewards)
            out_MT = jax.vmap(apply_fn, in_axes=(0, None))(params, obs)
            # return out_MT.mean(axis=0)
            return out_MT

        return reward_fn
