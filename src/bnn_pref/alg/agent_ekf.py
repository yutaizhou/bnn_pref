from functools import partial
from typing import Dict, Tuple, Union

import distrax
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from dynamax.nonlinear_gaussian_ssm import ParamsNLGSSM, extended_kalman_filter
from einops import rearrange
from flax import linen as nn
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Int

from bnn_pref.alg.agent_utils import (
    Agent,
    BatchNormTrainState,
    bt_loss_fn,
    compute_disagreement,
    compute_info_gain,
    run_sgd,
)
from bnn_pref.alg.pca_jax import JaxPCA
from bnn_pref.alg.projection_matrix import generate_random_basis, sub2full_params_flat
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.network import count_params
from bnn_pref.utils.type import QueryData, unpackable_dataclass


@unpackable_dataclass
class EKFBeliefState:
    """
    `offset_ts` is the full param model after SGD init. `mean` and `cov` govern the
    Gaussian distribution over subspace params.

    Full param is computed as: offset + ss_param @ proj_matrix.
    """

    mean: Float[Array, "system_dim"]
    cov: Float[Array, "system_dim system_dim"]
    t: int
    proj_matrix: Float[Array, "sub_dim full_dim"]
    offset_ts: Union[TrainState, BatchNormTrainState]


class EKFAgent(Agent):
    def __init__(
        self,
        model: nn.Module,
        traj_shape: Tuple[int, ...],  # kept for compat with Ensemble buffer
        learning_rate: float,
        momentum: float,
        l2_reg: float = 0.0,
        niters_init: int = 420,
        batch_size: int = 32,
        warm_burns: int = 1000,
        thinning: int = 2,
        sub_dim: Union[float, int] = 200,
        rnd_proj: bool = False,
        proj_type: str = "dense",
        sparsity: float = 0.001,
        prior_noise: float = 0.0001,
        dynamics_noise: float = 0.0,
        obs_noise: float = 1.0,
        iekf: int = 1,
        acq: str = "infogain",
        n_models: int = 20,
        chunk_size: int = 64,
        use_vmap: bool = True,
    ):
        self.model = model
        self.traj_shape = traj_shape  # (T, D) or (T, H, W, C)
        self.is_pixel = True if len(traj_shape) == 4 else False
        self.opt = optax.sgd(learning_rate, momentum)
        self.l2_reg = l2_reg
        self.niters_init = niters_init
        self.batch_size = batch_size
        self.warm_burns = warm_burns
        self.thinning = thinning
        self.sub_dim = sub_dim
        self.rnd_proj = rnd_proj
        self.proj_type = proj_type
        self.sparsity = sparsity
        self.prior_noise = prior_noise
        self.dynamics_noise = dynamics_noise
        self.obs_noise = obs_noise
        self.iekf = iekf
        self.n_models = n_models
        self.chunk_size = chunk_size
        self.use_vmap = use_vmap

        #! only use this for subdim sweep
        n_eff_iterates = (niters_init - warm_burns) // thinning
        # assert n_eff_iterates >= sub_dim, f"{n_eff_iterates=} < {sub_dim=}"
        if n_eff_iterates < sub_dim:
            self.niters_init = sub_dim * thinning + warm_burns

        assert acq in ["infogain", "disagreement"]
        self.acq = acq

    @staticmethod
    def get_hydra_config(alg_cfg):
        # follow ekf.yaml config
        return {
            "acq": alg_cfg["acq"],
            "learning_rate": alg_cfg["learning_rate"],
            "momentum": alg_cfg["momentum"],
            # subspace init
            "niters_init": alg_cfg["niters_init"],
            "batch_size": alg_cfg["bs"],
            "l2_reg": alg_cfg["l2_reg"],
            "warm_burns": alg_cfg["warm_burns"],
            "thinning": alg_cfg["thinning"],
            "sub_dim": alg_cfg["sub_dim"],
            "rnd_proj": alg_cfg["rnd_proj"],
            "proj_type": alg_cfg["proj_type"],
            "sparsity": alg_cfg["sparsity"],
            # subspace inference
            "prior_noise": alg_cfg["prior_noise"],
            "dynamics_noise": alg_cfg["dynamics_noise"],
            "obs_noise": alg_cfg["obs_noise"],
            "iekf": alg_cfg["iekf"],
            # ensembling
            "n_models": alg_cfg["M"],
            "chunk_size": alg_cfg["chunk_size"],
            "use_vmap": alg_cfg["use_vmap"],
        }

    def get_alg_info(self):
        return {
            "param_count": self.param_count,
            "subspace_param_count": self.subspace_param_count,
        }

    # @partial(jax.jit, static_argnames=["self"])
    def init_bel(self, key, warmup_data: QueryData) -> EKFBeliefState:
        """
        Run SGD on warmup data, get subspace projection matrix, initialize EKF
        contexts: Q2TD
        labels: Q2 (onehot)
        """
        self.subspace_param_count = self.sub_dim
        key, model_key = jr.split(key, 2)
        dummy_input = jnp.ones((1, 2, *self.traj_shape))
        if not self.is_pixel:
            params = self.model.init(model_key, dummy_input)["params"]
            ts = TrainState.create(
                apply_fn=self.model.apply,
                params=params,
                tx=self.opt,
            )
        else:
            variables = self.model.init(
                model_key,
                dummy_input,
            )
            params = variables["params"]
            batch_stats = variables["batch_stats"]

            ts = BatchNormTrainState.create(
                apply_fn=self.model.apply,
                params=params,
                tx=self.opt,
                batch_stats=batch_stats,
            )

        self.param_count = count_params(params)
        key, key_sgd = jr.split(key, 2)
        warm_ts, warm_metrics = run_sgd(
            key_sgd,
            ts,
            dataset=warmup_data,
            loss_fn=bt_loss_fn,
            niters=self.niters_init,
            batch_size=self.batch_size,
            l2_reg=self.l2_reg,
            get_param_trace=not self.rnd_proj,
            n_models=1,
            split_datastream=False,
            use_dropout=False,
            use_batch_norm=self.is_pixel,  # only image experiments using BN
            use_vmap=self.use_vmap,
        )
        self.batch_stats = warm_ts.batch_stats if self.is_pixel else None

        if self.rnd_proj:
            assert isinstance(self.sub_dim, int)
            full_dim = self.param_count
            sub_dim = self.sub_dim
            key, proj_key = jr.split(key, 2)
            proj_matrix = generate_random_basis(
                proj_key, sub_dim, full_dim, self.proj_type, self.sparsity
            )
        else:
            # (niters_init, full_dim)
            params_trace = warm_metrics["params"][self.warm_burns :: self.thinning]
            pca = JaxPCA(n_components=self.sub_dim)
            pca.fit(params_trace)
            sub_dim = pca.n_components_
            if isinstance(self.sub_dim, float):
                print(f"PCA found {sub_dim} components ({self.sub_dim=:.2%} var)")
            self.sub_dim = pca.n_components_
            proj_matrix = pca.components_  # (sub_dim, full_dim)

        params_offset, params_unraveler = ravel_pytree(warm_ts.params)
        self.warmed_params = warm_ts.params

        # these two are used for projection matrix "efficient" version
        def sub2full_params(ss_param_flat):
            """
            ss_param_flat: flattened vector of subspace params (sub_dim,)
            returns: unflatten pytree of fullspace params
            """
            param_flat = sub2full_params_flat(
                params_subspace=ss_param_flat,
                proj_matrix=proj_matrix,
                params_full=params_offset,
                type=self.proj_type,
            )
            return params_unraveler(param_flat)

        def pred_return(
            param: Dict,
            input: Union[Float[Array, "T D"], Float[Array, "T H W C"]],
        ) -> Float[Array, " "]:
            inputs = jnp.expand_dims(input, axis=0)
            variables = {"params": param}
            if self.is_pixel:
                variables["batch_stats"] = self.batch_stats
            outputs = self.model.apply(
                variables,
                inputs,
                method=self.model.predict_traj_return,
                train=False,
                mutable=False,
            ).squeeze(0)
            return outputs

        self.sub2full_params = sub2full_params
        self.pred_return = pred_return

        def emission_fn_state(ss_param_flat, input) -> Float[Array, "2"]:
            """
            ss_param_flat: (sub_dim,)
            input: (2 * T * D,)
            """
            params = sub2full_params(ss_param_flat)
            input = jnp.reshape(input, (1, 2, *self.traj_shape))
            variables = {"params": params}
            if self.is_pixel:
                variables["batch_stats"] = self.batch_stats
            logits = self.model.apply(
                variables,
                input,
                train=False,
                mutable=False,
            )  # (1, 2)
            probs_2 = jnp.exp(jax.nn.log_softmax(logits.squeeze(0)))
            return probs_2

        def emission_fn_pixel(ss_param_flat, embd) -> Float[Array, "2"]:
            """
            ss_param_flat: (sub_dim,)
            input: (2 * E,)
            """
            embd = jnp.reshape(embd, (2, -1))  # (2, E,)
            params = self.sub2full_params(ss_param_flat)
            variables = {"params": params, "batch_stats": self.batch_stats}
            logits = self.model.apply(
                variables,
                embd,
                train=False,
                method=self.model.compute_return_from_agg_embeddings,
            )  # (2, E) -> (2, 1)
            probs_2 = jnp.exp(jax.nn.log_softmax(logits.squeeze(1)))
            return probs_2

        init_mean = jnp.zeros(sub_dim)
        init_cov = jnp.eye(sub_dim) * self.prior_noise
        Q = jnp.eye(sub_dim) * self.dynamics_noise
        R = jnp.eye(2) * self.obs_noise
        self.ekf_params = ParamsNLGSSM(
            initial_mean=init_mean,
            initial_covariance=init_cov,
            dynamics_function=lambda z, u: z,  # constant dynamics
            dynamics_covariance=Q,
            emission_function=emission_fn_pixel if self.is_pixel else emission_fn_state,
            emission_covariance=R,
        )

        bel = EKFBeliefState(
            mean=init_mean,
            cov=init_cov,
            t=0,
            proj_matrix=proj_matrix,
            offset_ts=warm_ts,
        )
        return bel

    @partial(jax.jit, static_argnames=["self"])
    def update_bel(
        self,
        key,  # for compatibility with ensemble
        bel: EKFBeliefState,
        batch: QueryData,
    ) -> EKFBeliefState:
        prior_mean, prior_cov, t = bel.mean, bel.cov, bel.t
        context, label = batch.contexts, batch.labels

        emissions = label  # (1,2)
        if not self.is_pixel:
            inputs = jnp.reshape(context, (1, -1))  # (1, 2*T*D) or (1, 2*T*H*W*C)
        else:
            inputs = self.model.apply(
                {"params": bel.offset_ts.params, "batch_stats": self.batch_stats},
                jnp.reshape(context, (2, *self.traj_shape)),  # (2, T, H, W, C),
                train=False,
                method=self.model.compute_embeddings,
                agg=True,
            )  # (2, T, H, W, C) -> (2, E)
            inputs = jnp.reshape(inputs, (1, -1))  # (2 * E,)

        self.ekf_params = self.ekf_params._replace(
            initial_mean=prior_mean,
            initial_covariance=prior_cov,
        )
        posterior = extended_kalman_filter(
            self.ekf_params,
            inputs=inputs,
            emissions=emissions,
            num_iter=self.iekf,
        )

        posterior_mean = posterior.filtered_means[-1]
        posterior_cov = posterior.filtered_covariances[-1]
        bel = bel.replace(mean=posterior_mean, cov=posterior_cov, t=t + 1)
        return bel

    @partial(jax.jit, static_argnames=["self", "env"])
    def compute_next_query(
        self,
        key,
        bel: EKFBeliefState,
        env: PreferenceEnv,
        pool_idxes_Q: Int[Array, "Q"],
    ) -> int:
        """
        active learning: greedily compute query that maximizes acquisition function
        """
        # * sample M (subspace) models from posterior
        M = self.n_models
        distr = distrax.MultivariateNormalFullCovariance(bel.mean, bel.cov)
        key, key_sample = jr.split(key, 2)
        ss_params = distr.sample(seed=key_sample, sample_shape=(M,))  # (M, sub_dim)
        if self.use_vmap:
            params = jax.vmap(self.sub2full_params)(ss_params)  # pytree (lead axis M)
        else:
            params = jax.lax.map(self.sub2full_params, ss_params, batch_size=8)

        # * precompute logits for all items, assume ts lead dimension is M
        # efficient version (sub2full only called once)
        def scan_param(_, param):
            fn = partial(self.pred_return, param)
            ret_N = jax.lax.map(fn, env.items_NTD, batch_size=self.chunk_size)
            return _, ret_N

        logits_NM = rearrange(
            jax.lax.scan(scan_param, init=None, xs=params)[1],
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
        bel: EKFBeliefState,
        items_NTD: Float[Array, "N T D"],
        query_idxs_Q2: Int[Array, "Q 2"],
    ) -> Float[Array, "Q 2"]:
        """
        sample params from posterior, then compute posterior predictive
        """
        # * sample M (subspace) models from posterior
        M = self.n_models
        distr = distrax.MultivariateNormalFullCovariance(bel.mean, bel.cov)
        key, key_sample = jr.split(key, 2)
        ss_params = distr.sample(seed=key_sample, sample_shape=(M,))  # (M, sub_dim)
        if self.use_vmap:
            params = jax.vmap(self.sub2full_params)(ss_params)  # pytree (lead axis M)
        else:
            params = jax.lax.map(self.sub2full_params, ss_params, batch_size=8)

        # * precompute logits for all items, assume ts lead dimension is M
        # efficient version (sub2full only called once)
        def scan_param(_, param):
            fn = partial(self.pred_return, param)
            ret_N = jax.lax.map(fn, items_NTD, batch_size=self.chunk_size)
            return _, ret_N

        logits_NM = rearrange(
            jax.lax.scan(scan_param, init=None, xs=params)[1],
            "M N -> N M",
        )

        # * compute posterior predictive
        logits_QM2 = rearrange(logits_NM[query_idxs_Q2], "Q K M -> Q M K", K=2)
        llik_QM2 = jax.nn.log_softmax(logits_QM2, axis=2)
        llik_Q2 = jax.nn.logsumexp(llik_QM2, axis=1) - jnp.log(M)
        prob_Q2 = jnp.exp(llik_Q2)
        return prob_Q2
