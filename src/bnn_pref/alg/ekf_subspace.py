from functools import partial
from typing import Dict, Tuple, Union

import distrax
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from dynamax.generalized_gaussian_ssm import EKFIntegrals, ParamsGGSSM
from dynamax.generalized_gaussian_ssm import conditional_moments_gaussian_filter as cmgf
from dynamax.nonlinear_gaussian_ssm import ParamsNLGSSM, extended_kalman_filter
from einops import rearrange
from flax import linen as nn
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Int
from sklearn.decomposition import PCA

from bnn_pref.alg.agent_utils import (
    Agent,
    JaxPCA,
    bt_loss_fn,
    generate_random_basis,
    run_gradient_descent,
    sub2full_params_flat,
)
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.network import count_params
from bnn_pref.utils.type import QueryData, unpackable_dataclass


@unpackable_dataclass
class EKFBeliefState:
    mean: Float[Array, "system_dim"]
    cov: Float[Array, "system_dim system_dim"]
    t: int
    proj_matrix: Float[Array, "sub_dim full_dim"]
    offset_ts: TrainState


class SubspaceEKF(Agent):
    def __init__(
        self,
        model: nn.Module,
        opt: optax.GradientTransformation,
        traj_shape: Tuple[int, ...],  # kept for compat with Ensemble buffer
        l2_reg: float = 0.0,
        niters: int = 1000,
        batch_size: int = 32,
        warm_burns: int = 1000,
        thinning: int = 2,
        sub_dim: Union[float, int] = 0.9999,
        rnd_proj: bool = False,
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
        self.opt = opt
        self.l2_reg = l2_reg
        self.niters = niters
        self.batch_size = batch_size
        self.warm_burns = warm_burns
        self.thinning = thinning
        self.sub_dim = sub_dim
        self.rnd_proj = rnd_proj
        self.prior_noise = prior_noise
        self.dynamics_noise = dynamics_noise
        self.obs_noise = obs_noise
        self.iekf = iekf
        self.n_models = n_models
        self.n_feats = None
        self.chunk_size = chunk_size
        self.use_vmap = use_vmap
        if not rnd_proj:
            n_eff_iterates = (niters - warm_burns) // thinning
            assert n_eff_iterates >= sub_dim, f"{n_eff_iterates=} < {sub_dim=}"
        assert acq in ["infogain", "disagreement"]
        self.acq = acq

    @staticmethod
    def get_hydra_config(ekf_cfg):
        # follow ekf.yaml config
        return {
            "acq": ekf_cfg["acq"],
            # subspace init
            "niters": ekf_cfg["niters"],
            "batch_size": ekf_cfg["bs"],
            "l2_reg": ekf_cfg["l2_reg"],
            "warm_burns": ekf_cfg["warm_burns"],
            "thinning": ekf_cfg["thinning"],
            "sub_dim": ekf_cfg["sub_dim"],
            "rnd_proj": ekf_cfg["rnd_proj"],
            # subspace inference
            "prior_noise": ekf_cfg["prior_noise"],
            "dynamics_noise": ekf_cfg["dynamics_noise"],
            "obs_noise": ekf_cfg["obs_noise"],
            "iekf": ekf_cfg["iekf"],
            # ensembling
            "n_models": ekf_cfg["M"],
            "chunk_size": ekf_cfg["chunk_size"],
            "use_vmap": ekf_cfg["use_vmap"],
        }

    # @partial(jax.jit, static_argnames=["self"])
    def init_bel(self, key, warmup_data: QueryData) -> EKFBeliefState:
        """
        Run SGD on warmup data, get subspace projection matrix, initialize EKF
        contexts: Q2TD
        labels: Q2 (onehot)
        """
        contexts = warmup_data.contexts  # (Q,2,T,D)
        self.n_feats = contexts.shape[-1]  # D
        key, model_key = jr.split(key, 2)
        dummy_context = rearrange(jnp.ones_like(contexts[0]), "K T D  -> 1 K T D", K=2)
        initial_params = self.model.init(model_key, dummy_context)["params"]
        # print(nn.tabulate(self.model, model_key)(dummy_context))
        # print(count_params(initial_params))

        ts = TrainState.create(
            apply_fn=self.model.apply, params=initial_params, tx=self.opt
        )

        key, key_sgd = jr.split(key, 2)
        warm_ts, warm_metrics = run_gradient_descent(
            key_sgd,
            ts,
            loss_fn=bt_loss_fn,
            has_aux=True,
            dataset=warmup_data,
            niters=self.niters,
            batch_size=self.batch_size,
            l2_reg=self.l2_reg,
        )

        params_trace = warm_metrics["params"][self.warm_burns :: self.thinning]

        if self.rnd_proj:
            assert isinstance(self.sub_dim, int)
            full_dim = params_trace.shape[-1]
            sub_dim = self.sub_dim
            key, proj_key = jr.split(key, 2)
            proj_matrix = generate_random_basis(proj_key, sub_dim, full_dim)
        else:
            # pca = PCA(n_components=self.sub_dim)
            pca = JaxPCA(n_components=self.sub_dim)
            pca.fit(params_trace)
            sub_dim = pca.n_components_
            if isinstance(self.sub_dim, float):
                print(f"PCA found {sub_dim} components ({self.sub_dim=:.2%} var)")
            self.sub_dim = pca.n_components_
            proj_matrix = pca.components_  # (sub_dim, full_dim)

        self.param_count = count_params(initial_params)
        self.subspace_param_count = sub_dim

        params_offset, params_unravel_fn = ravel_pytree(warm_ts.params)
        self.warmed_params = warm_ts.params

        # these two are used for projection matrix "efficient" version
        def sub2full_params(ss_param_flat):
            """
            ss_param_flat: flattened vector of subspace params (sub_dim,)
            returns: unflatten pytree of fullspace params
            """
            param_flat = sub2full_params_flat(ss_param_flat, proj_matrix, params_offset)
            return params_unravel_fn(param_flat)

        def pred_return(
            param: Dict,
            items_TD: Float[Array, "T D"],
        ) -> Float[Array, " "]:
            inputs = rearrange(items_TD, "T D -> 1 T D")
            params = {"params": param}
            outputs = self.model.apply(
                params, inputs, method=self.model.predict_traj_return
            ).squeeze(0)
            return outputs

        self.sub2full_params = sub2full_params
        self.pred_return = pred_return

        # these two are used for projection matrix "inefficient" version
        def sub2full_predict_return(
            ss_param_flat,
            traj: Float[Array, "T D"],
        ) -> Float[Array, " "]:
            params = sub2full_params(ss_param_flat)
            inputs = rearrange(traj, "T D -> 1 T D")
            outputs = self.model.apply(
                {"params": params}, inputs, method=self.model.predict_traj_return
            ).squeeze(0)
            return outputs

        def sub2full_predict_logits(
            ss_param_flat,
            inputs: Float[Array, "2 T D"],
        ) -> Float[Array, "2"]:
            """
            Project params from subspace to full space, then apply model
            to get logits for both trajectories
            """
            params = sub2full_params(ss_param_flat)
            inputs = rearrange(inputs, "K T D -> 1 K T D", K=2)
            outputs = self.model.apply({"params": params}, inputs)
            outputs = rearrange(outputs, "1 K -> K", K=2)
            return outputs

        self.sub2full_predict_return = sub2full_predict_return
        self.sub2full_predict_logits = sub2full_predict_logits

        def emission_fn(
            ss_param_flat,
            inputs: Float[Array, "2 T D"],
        ) -> Float[Array, "2"]:
            """
            emission model where
                inputs: (2 * T * D,) query features -> (2,) traj rewards as logits
                predicted measurement: (2,) # probabilities of traj 2 > traj 1
                gt measurement: (2,) # one hot labels

            params: (sub_dim,)
            inputs: (2 * T * D,)
            """
            inputs = rearrange(inputs, "(K T D) -> K T D", K=2, D=self.n_feats)
            logits = sub2full_predict_logits(ss_param_flat, inputs)  # (2,)

            probs_2 = jnp.exp(jax.nn.log_softmax(logits))
            return probs_2

        # def emission_mean_cmgf(params, inputs):
        #     """
        #     emission model where
        #         inputs: (2 * T * D,) query features -> (2,) traj rewards as logits
        #         predicted measurement: (2,) # probabilities of traj 2 > traj 1
        #         gt measurement: (2,) # one hot labels
        #     """
        #     context = inputs.reshape(2, -1, self.n_feats)
        #     logits = sub2full_predict_logits(params, context)  # (2,)
        #     p = jax.nn.softmax(logits, axis=0)[1][None]  # (1,1)

        #     return p

        # def emission_cov_cmgf(params, inputs):
        #     """
        #     emission model where
        #         inputs: (2 * T * D,) query features -> (2,) traj rewards as logits
        #         predicted measurement: (2,) # probabilities of traj 2 > traj 1
        #         gt measurement: (2,) # one hot labels
        #     """
        #     context = inputs.reshape(2, -1, self.n_feats)
        #     logits = sub2full_predict_logits(params, context)  # (2,)
        #     p = jax.nn.softmax(logits, axis=0)[1][None]
        #     return p * (1 - p)

        init_mean = jnp.zeros(sub_dim)
        init_cov = jnp.eye(sub_dim) * self.prior_noise
        Q = jnp.eye(sub_dim) * self.dynamics_noise
        R = jnp.eye(2) * self.obs_noise
        self.ekf_params = ParamsNLGSSM(
            initial_mean=init_mean,
            initial_covariance=init_cov,
            dynamics_function=lambda z, u: z,  # constant dynamics
            dynamics_covariance=Q,
            emission_function=emission_fn,
            emission_covariance=R,
        )

        # self.cmgf_params = ParamsGGSSM(
        #     initial_mean=init_mean,
        #     initial_covariance=S,
        #     dynamics_function=lambda z, u: z,  # constant dynamics
        #     dynamics_covariance=Q,
        #     emission_mean_function=emission_mean_cmgf,
        #     emission_cov_function=emission_cov_cmgf,
        # )

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

        inputs = rearrange(context, "1 K T D -> 1 (K T D)", K=2)
        emissions = rearrange(label, "1 K -> 1 K", K=2)

        self.ekf_params = self.ekf_params._replace(
            initial_mean=prior_mean,
            initial_covariance=prior_cov,
        )
        posterior = extended_kalman_filter(
            self.ekf_params,
            emissions=emissions,
            inputs=inputs,
            num_iter=self.iekf,
        )

        # emissions_cmgf = rearrange(label[1][None], "K -> 1 K", K=1)  # OH: always 1

        # self.cmgf_params = self.cmgf_params._replace(
        #     initial_mean=prior_mean,
        #     initial_covariance=prior_cov,
        # )
        # posterior = cmgf(
        #     self.cmgf_params,
        #     EKFIntegrals(),
        #     emissions=emissions_cmgf,
        #     inputs=inputs,
        #     num_iter=self.iekf,
        # )

        posterior_mean = posterior.filtered_means[-1]
        posterior_cov = posterior.filtered_covariances[-1]
        # bel = EKFBeliefState(mean=posterior_mean, cov=posterior_cov, t=t + 1)
        bel = bel.replace(mean=posterior_mean, cov=posterior_cov, t=t + 1)
        return bel

    @partial(jax.jit, static_argnames=["self", "env"])
    def acquire_next_query(
        self,
        key,
        bel: EKFBeliefState,
        env: PreferenceEnv,
        pool_idxes_Q: Int[Array, "Q"],
    ) -> int:
        """
        active learning: greedily compute query that maximizes InfoGain acquisition fn
        """
        # * sample M (subspace) models from posterior
        M = self.n_models  # number of models to sample
        distr = distrax.MultivariateNormalFullCovariance(bel.mean, bel.cov)
        key, key_sample = jr.split(key, 2)
        ss_params = distr.sample(seed=key_sample, sample_shape=(M,))
        params = jax.vmap(self.sub2full_params)(ss_params)  # pytree (lead axis M)

        if self.use_vmap:
            fn = jax.vmap(self.pred_return, in_axes=(0, None))
            fn = partial(fn, params)
            logits_NM = jax.lax.map(fn, env.items_NTD, batch_size=self.chunk_size)

        else:

            def scan_param(_, param):
                fn = partial(self.pred_return, param)
                ret_N = jax.lax.map(fn, env.items_NTD, batch_size=self.chunk_size)
                return _, ret_N

            logits_NM = rearrange(
                jax.lax.scan(scan_param, None, params)[1],
                "M N -> N M",
            )

        # * precompute logits for all items
        # if self.use_vmap:
        #     fn = jax.vmap(self.sub2full_predict_return, in_axes=(0, None))
        #     fn = partial(fn, ss_params)
        #     logits_NM = jax.lax.map(fn, env.items_NTD, batch_size=self.chunk_size)

        # else:

        #     def scan_param(_, ss_param):
        #         fn = partial(self.sub2full_predict_return, ss_param)
        #         ret_N = jax.lax.map(fn, env.items_NTD, batch_size=self.chunk_size)
        #         return _, ret_N

        #     logits_NM = rearrange(
        #         jax.lax.scan(scan_param, None, ss_params)[1],
        #         "M N -> N M",
        #     )

        # def compute_info_gain(logprobs_M2):
        #     probs_M2 = jnp.exp(logprobs_M2)
        #     probs_M2 = jnp.nan_to_num(probs_M2, posinf=1.0, neginf=1e-8)

        #     mi_M2 = probs_M2 * jnp.log2(M * probs_M2 / jnp.sum(probs_M2, axis=0))
        #     value = jnp.sum(mi_M2) / M
        #     return value

        logM = jnp.log(M)
        log2 = jnp.log(2)

        def compute_info_gain(logprobs_M2):
            """work in logspace for numerical stability"""
            log_sum_p = jax.nn.logsumexp(logprobs_M2, axis=0, keepdims=True)
            mi_M2 = jnp.exp(logprobs_M2) * (logM + logprobs_M2 - log_sum_p) / log2
            value = jnp.sum(mi_M2) / M
            return value

        def map_step(idx):
            inds_2 = env.get_pref_indices(idx)
            logits_M2 = rearrange(logits_NM[inds_2], "K M -> M K", K=2)
            if self.acq == "infogain":
                logprobs_M2 = jax.nn.log_softmax(logits_M2, axis=1)
                value = compute_info_gain(logprobs_M2)
            elif self.acq == "disagreement":
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
        key,
        bel: EKFBeliefState,
        items_NTD: Float[Array, "N T D"],
        query_idxs_Q2: Int[Array, "Q 2"],
    ) -> Float[Array, "Q 2"]:
        """sample params from posterior, then compute predictive"""
        # * sample model parameters
        M = self.n_models
        dist = distrax.MultivariateNormalFullCovariance(bel.mean, bel.cov)
        key, key_sample = jr.split(key, 2)
        ss_params = dist.sample(seed=key_sample, sample_shape=(M,))
        params = jax.vmap(self.sub2full_params)(ss_params)  # pytree (lead axis M)

        if self.use_vmap:
            fn = jax.vmap(self.pred_return, in_axes=(0, None))
            fn = partial(fn, params)
            logits_NM = jax.lax.map(fn, items_NTD, batch_size=self.chunk_size)

        else:

            def scan_param(_, param):
                fn = partial(self.pred_return, param)
                ret_N = jax.lax.map(fn, items_NTD, batch_size=self.chunk_size)
                return _, ret_N

            logits_NM = rearrange(
                jax.lax.scan(scan_param, None, params)[1],
                "M N -> N M",
            )

        # * precompute logits for all items
        # if self.use_vmap:
        #     fn = jax.vmap(self.sub2full_predict_return, in_axes=(0, None))
        #     fn = partial(fn, ss_params)
        #     logits_NM = jax.lax.map(fn, items_NTD, batch_size=self.chunk_size)
        # else:

        #     def scan_param(_, ss_param):
        #         fn = partial(self.sub2full_predict_return, ss_param)
        #         ret_N = jax.lax.map(fn, items_NTD, batch_size=self.chunk_size)
        #         return _, ret_N

        #     logits_NM = rearrange(
        #         jax.lax.scan(scan_param, None, ss_params)[1],
        #         "M N -> N M",
        #     )

        logits_QM2 = rearrange(logits_NM[query_idxs_Q2], "Q K M -> Q M K", K=2)

        # * compute predictive distributions
        llik_QM2 = jax.nn.log_softmax(logits_QM2, axis=2)
        llik_Q2 = jax.nn.logsumexp(llik_QM2, axis=1) - jnp.log(M)
        prob_Q2 = jnp.exp(llik_Q2)
        return prob_Q2
