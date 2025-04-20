from functools import partial
from typing import Union

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
from jaxtyping import Array, Float, Scalar
from sklearn.decomposition import PCA
from tensorflow_probability.substrates import jax as tfp

from bnn_pref.alg.agent_utils import (
    Agent,
    JaxPCA,
    bt_loss_fn,
    generate_random_basis,
    run_gradient_descent,
    subspace2full_params,
)
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.network import count_params
from bnn_pref.utils.type import CARL, unpackable_dataclass

tfd = tfp.distributions


@unpackable_dataclass
class EKFBeliefState:
    mean: Float[Array, "system_dim"]
    cov: Float[Array, "system_dim system_dim"]
    t: int


class SubspaceNeuralEKF(Agent):
    def __init__(
        self,
        model: nn.Module,
        opt: optax.GradientTransformation,
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
        mi_samples: int = 20,
        chunk_size: int = 64,
    ):
        """
        Subspace Neural Bandit implementation.
        Parameters
        ----------
        n_feats : int
            The number of input features of the model.
        model : flax.nn.Module
            The flax model to be used for the bandits.
        opt: flax.optim.Optimizer
            The optimizer to be used for training the model.
        warm_burns : int
            The number of SGD iterates to be thrown away for the warmup phase.
        sub_dim: Union[float, int]
            The number of components to be used for the PCA.
        rnd_proj: bool
            Whether to use random projection.
        prior_noise : float
            The prior noise for the EKF.
        dynamics_noise: float
            The dynamics noise for the EKF.
        obs_noise: float
            The observation noise for the EKF.
        batch_size: Optional[int]
            The batch size for mini-batch SGD.
        """
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
        self.mi_samples = mi_samples
        self.n_feats = None
        self.chunk_size = chunk_size
        if not rnd_proj:
            n_eff_iterates = (niters - warm_burns) // thinning
            assert n_eff_iterates >= sub_dim, f"{n_eff_iterates=} < {sub_dim=}"

    # @partial(jax.jit, static_argnames=["self"])
    def init_bel(self, key, warmup_data: CARL) -> EKFBeliefState:
        """
        Run SGD on warmup data, get subspace projection matrix, initialize EKF
        contexts: Q2TD
        actions: Q
        rewards: Q
        labels: Q2 (onehot)
        """
        contexts, _, _, labels = warmup_data
        self.n_feats = contexts.shape[-1]  # (Q,2,T, D) -> D
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
            assert type(self.sub_dim) is int
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

        self.full_params_count = count_params(initial_params)
        self.subspace_params_count = sub_dim

        params_full_init, reconstruct_tree_params = ravel_pytree(warm_ts.params)
        self.warmed_params = warm_ts.params

        def sub2full_predict_return(
            params_subspace,
            traj: Float[Array, "T D"],
        ) -> Float[Array, ""]:
            params_full = subspace2full_params(
                params_subspace, proj_matrix, params_full_init
            )
            params = reconstruct_tree_params(params_full)
            outputs = self.model.apply(
                {"params": params},
                rearrange(traj, "T D -> 1 T D"),
                method=self.model.predict_traj_return,
            )
            return outputs

        def sub2full_predict_logits(
            params_subspace,
            inputs: Float[Array, "2 T D"],
        ) -> Float[Array, "2"]:
            """
            Project params from subspace to full space, then apply model
            to get logits for both trajectories
            """
            params_full = subspace2full_params(
                params_subspace, proj_matrix, params_full_init
            )
            params = reconstruct_tree_params(params_full)

            inputs = rearrange(inputs, "K T D -> 1 K T D", K=2)
            outputs = self.model.apply({"params": params}, inputs)
            outputs = rearrange(outputs, "1 K -> K", K=2)
            return outputs

        self.sub2full_predict_return = sub2full_predict_return
        self.sub2full_predict_logits = sub2full_predict_logits

        def emission_fn(
            params,
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
            logits = sub2full_predict_logits(params, inputs)  # (2,)

            probs_2 = jnp.exp(jax.nn.log_softmax(logits))
            return probs_2

            # return rearrange(probs_2[1], " -> 1")

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
        S = jnp.eye(sub_dim) * self.prior_noise
        Q = jnp.eye(sub_dim) * self.dynamics_noise
        R = jnp.eye(2) * self.obs_noise
        self.ekf_params = ParamsNLGSSM(
            initial_mean=init_mean,
            initial_covariance=S,
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

        bel = EKFBeliefState(mean=init_mean, cov=S, t=0)
        return bel

    @partial(jax.jit, static_argnames=["self"])
    def update_bel(
        self,
        bel: EKFBeliefState,
        batch: CARL,
    ) -> EKFBeliefState:
        prior_mean, prior_cov, t = bel
        context, *_, label = batch

        inputs = rearrange(context, "K T D -> 1 (K T D)", K=2)
        emissions = rearrange(label, "K -> 1 K", K=2)  # (1,2)
        # emissions = rearrange(label[1], " -> 1")

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
        bel = EKFBeliefState(mean=posterior_mean, cov=posterior_cov, t=t + 1)
        return bel

    # @partial(jax.jit, static_argnames=["self", "env"])
    # def acquire_next_query(
    #     self, key, bel: EKFBeliefState, env: PreferenceEnv, pool_idxes_Q: Array
    # ) -> int:
    #     """
    #     active learning: greedily compute query that maximizes InfoGain acquisition fn
    #     """
    #     # * sample M (subspace) models from posterior
    #     M = self.mi_samples  # number of models to sample
    #     mean, cov = bel.mean, bel.cov
    #     distr = tfd.MultivariateNormalFullCovariance(mean, cov)
    #     key, key_sample = jr.split(key, 2)
    #     ss_params = distr.sample(seed=key_sample, sample_shape=(M,))
    #     fn = jax.vmap(self.sub2full_predict_logits, in_axes=(0, None))  # (param, input)

    #     # * compute logits for all contexts
    #     def compute_info_gain(probs_M2):
    #         mi = probs_M2 * jnp.log2(M * probs_M2 / jnp.sum(probs_M2, axis=0))
    #         mi = jnp.sum(mi) / M
    #         return mi

    #     def map_step(idx):
    #         context_2TD = env.get_context(idx)
    #         logits_M2 = fn(ss_params, context_2TD)
    #         probs_M2 = jnp.exp(jax.nn.log_softmax(logits_M2, axis=1))
    #         value = compute_info_gain(probs_M2)
    #         return value

    #     values_Q = jax.lax.map(map_step, pool_idxes_Q, batch_size=self.chunk_size)
    #     query_idx = jnp.argmax(values_Q)
    #     return query_idx

    @partial(jax.jit, static_argnames=["self", "env"])
    def acquire_next_query(
        self,
        key,
        bel: EKFBeliefState,
        env: PreferenceEnv,
        pool_idxes_Q: Array,
    ) -> int:
        """
        active learning: greedily compute query that maximizes InfoGain acquisition fn
        """
        # * sample M (subspace) models from posterior
        M = self.mi_samples  # number of models to sample
        mean, cov = bel.mean, bel.cov
        distr = tfd.MultivariateNormalFullCovariance(mean, cov)
        key, key_sample = jr.split(key, 2)
        ss_params = distr.sample(seed=key_sample, sample_shape=(M,))
        fn = jax.vmap(self.sub2full_predict_return, in_axes=(0, None))
        fn = partial(fn, ss_params)

        # precompute logits for all items
        logits_NM = jax.lax.map(fn, env.items_NTD, batch_size=self.chunk_size).squeeze()

        def compute_info_gain(probs_M2):
            mi = probs_M2 * jnp.log2(M * probs_M2 / jnp.sum(probs_M2, axis=0))
            mi = jnp.sum(mi) / M
            return mi

        def map_step(idx):
            inds_2 = env.get_pref_indices(idx)
            logits_M2 = rearrange(logits_NM[inds_2], "K M -> M K", K=2)
            probs_M2 = jnp.exp(jax.nn.log_softmax(logits_M2, axis=1))
            value = compute_info_gain(probs_M2)
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
        query_idxs_Q2: Float[Array, "Q 2"],
    ) -> Float[Array, "Q 2"]:
        # * sample model parameters
        mean, cov, t = bel
        dist = tfd.MultivariateNormalFullCovariance(mean, cov)
        ss_params = dist.sample(seed=key, sample_shape=(self.mi_samples,))
        fn = jax.vmap(self.sub2full_predict_return, in_axes=(0, None))  # over params
        fn = partial(fn, ss_params)

        # precompute logits for all items
        logits_NM = jax.lax.map(fn, items_NTD, batch_size=self.chunk_size).squeeze()
        logits_QM2 = rearrange(logits_NM[query_idxs_Q2], "Q K M -> Q M K", K=2)

        # * compute predictive distributions
        # logits_QM2 = jax.lax.map(fn, feats_Q2TD, batch_size=self.chunk_size)
        # llik_QM2 = logits_QM2 - jax.nn.logsumexp(logits_QM2, axis=2, keepdims=True)
        llik_QM2 = jax.nn.log_softmax(logits_QM2, axis=2)
        prob_Q2 = jnp.exp(llik_QM2).mean(1)
        return prob_Q2

    # # mode only, for debugging
    # @partial(jax.jit, static_argnames=["self"])
    # def compute_predictive(
    #     self,
    #     key,
    #     bel: EKFBeliefState,
    #     feats_Q2TD: Float[Array, "Q 2 T D"],
    # ) -> Float[Array, "Q 2"]:
    #     # * sample model parameters
    #     mean, cov, t = bel
    #     fn = partial(self.sub2full_predict_logits, mean)  # param, inputs

    #     # * compute predictive distributions
    #     logits_Q2 = jax.lax.map(fn, feats_Q2TD, batch_size=self.chunk_size)
    #     # llik_QM2 = logits_QM2 - jax.nn.logsumexp(logits_QM2, axis=2, keepdims=True)
    #     llik_Q2 = jax.nn.log_softmax(logits_Q2, axis=2)
    #     prob_Q2 = jnp.exp(llik_Q2)
    #     return prob_Q2
