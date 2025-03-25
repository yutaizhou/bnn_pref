from typing import Union

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from dynamax.nonlinear_gaussian_ssm import ParamsNLGSSM, extended_kalman_filter
from einops import rearrange
from flax import linen as nn
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Scalar
from tensorflow_probability.substrates import jax as tfp

from bnn_pref.alg.agent_utils import (
    JaxPCA,
    generate_random_basis,
    run_gradient_descent,
    subspace2full_params,
)
from bnn_pref.data.ekf_env import retrieve
from bnn_pref.utils.network import count_params
from bnn_pref.utils.type import CAR, CARL, BeliefState

tfd = tfp.distributions


class SubspaceNeuralEKF:
    def __init__(
        self,
        n_feats: int,
        model: nn.Module,
        opt,
        l2_reg: float = 0.0,
        n_iterates: int = 1000,
        batch_size: int = 32,
        warm_burns: int = 1000,
        thinning: int = 2,
        sub_dim: Union[float, int] = 0.9999,
        rnd_proj: bool = False,
        prior_noise: float = 0.0001,
        dynamics_noise: float = 0.0,
        obs_noise: float = 1.0,
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
        self.n_feats = n_feats
        self.model = model
        self.opt = opt
        self.prior_noise = prior_noise
        self.warm_burns = warm_burns
        self.n_iterates = n_iterates
        self.thinning = thinning
        self.dynamics_noise = dynamics_noise
        self.obs_noise = obs_noise
        self.sub_dim = sub_dim
        self.rnd_proj = rnd_proj
        self.l2_reg = l2_reg
        self.batch_size = batch_size

        if not rnd_proj:
            n_eff_iterates = (n_iterates - warm_burns) // thinning
            assert n_eff_iterates >= sub_dim, f"{n_eff_iterates=} < {sub_dim=}"

    def init_bel(self, key, warmup_data: CARL) -> BeliefState:
        """
        Run SGD on warmup data, get subspace projection matrix, initialize EKF
        contexts: Q2TD
        actions: Q
        rewards: Q
        labels: Q2 (onehot)
        """
        contexts, _, _, labels = warmup_data
        key, model_key = jr.split(key, 2)
        dummy_context = rearrange(jnp.ones_like(contexts[0]), "K T D  -> 1 K T D", K=2)
        initial_params = self.model.init(model_key, dummy_context)["params"]
        # print(nn.tabulate(self.model, model_key)(dummy_context))

        ts = TrainState.create(
            apply_fn=self.model.apply, params=initial_params, tx=self.opt
        )

        def loss_fn(params, batch_idx: Float[Array, "batch_size"]):
            # For full batch, use all data
            bs = self.batch_size
            contexts_batch = contexts if bs == -1 else retrieve(contexts, batch_idx)
            labels_batch = labels if bs == -1 else retrieve(labels, batch_idx)
            logits_N2 = self.model.apply({"params": params}, contexts_batch)
            loss = optax.softmax_cross_entropy(logits_N2, labels_batch).mean()
            params_flat, _ = ravel_pytree(params)
            l2_loss = self.l2_reg * (params_flat**2).sum()
            return loss + l2_loss, logits_N2

        key, key_sgd = jr.split(key, 2)
        warm_ts, warm_metrics = run_gradient_descent(
            key_sgd,
            ts,
            loss_fn,
            n_iterates=self.n_iterates,
            data_size=contexts.shape[0],
            batch_size=self.batch_size,
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
        ) -> Scalar:
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
            context: Float[Array, "2 T D"],
        ) -> Float[Array, "2"]:
            """
            Project params from subspace to full space, then apply model
            to get logits for both trajectories
            """
            params_full = subspace2full_params(
                params_subspace, proj_matrix, params_full_init
            )
            params = reconstruct_tree_params(params_full)

            context = rearrange(context, "K T D -> 1 K T D", K=2)
            outputs = self.model.apply({"params": params}, context)
            outputs = rearrange(outputs, "1 K -> K", K=2)
            return outputs

        self.sub2full_predict_return = sub2full_predict_return
        self.sub2full_predict_logits = sub2full_predict_logits

        def emission_fn(params, inputs):
            """
            emission model where inputs is (N, D + 1)
            """
            context = inputs[..., :-1].reshape(2, -1, self.n_feats)
            action = inputs[..., -1].astype(int)
            return sub2full_predict_logits(params, context)[action, None]

        init_mean = jnp.zeros(sub_dim)
        # key, key_ekf_init = jr.split(key, 2)
        # params_subspace_init = jr.normal(key_ekf_init, (sub_dim,))
        S = jnp.eye(sub_dim) * self.prior_noise
        Q = jnp.eye(sub_dim) * self.dynamics_noise
        R = jnp.eye(1) * self.obs_noise  # emission is (1,) for scalar one-hot reward
        self.ekf_params = ParamsNLGSSM(
            initial_mean=init_mean,
            initial_covariance=S,
            dynamics_function=lambda z, u: z,  # constant dynamics
            dynamics_covariance=Q,
            emission_function=emission_fn,
            emission_covariance=R,
        )

        bel = BeliefState(mean=init_mean, cov=S, t=0)
        return bel

    def update_bel(
        self,
        bel: BeliefState,
        batch: CAR,
    ) -> BeliefState:
        prior_mean, prior_cov, t = bel
        context, action, reward = batch

        # context = rearrange(context_2D, "Two D -> (Two D)")
        context = rearrange(context, "K T D -> (K T D)", K=2)
        action = rearrange(action, " -> 1")
        inputs = rearrange(jnp.concat((context, action)), "d -> 1 d")
        emission = rearrange(reward, " -> 1 1")

        self.ekf_params = self.ekf_params._replace(
            initial_mean=prior_mean,
            initial_covariance=prior_cov,
        )
        ekf_posterior = extended_kalman_filter(
            self.ekf_params,
            emissions=emission,  # reward
            inputs=inputs,  # context + action
        )

        posterior_mean = ekf_posterior.filtered_means[-1]
        posterior_cov = ekf_posterior.filtered_covariances[-1]
        bel = BeliefState(mean=posterior_mean, cov=posterior_cov, t=t + 1)
        return bel

    def choose_action(
        self,
        key,
        bel: BeliefState,
        context: Float[Array, "2 T D"],
    ) -> Scalar:
        # Thompson sampling strategy
        # Could also use epsilon greedy or UCB
        w = self.sample_params(key, bel)
        logits_2 = self.sub2full_predict_logits(w, context)
        action = logits_2.argmax()
        return action

    def sample_params(self, key, bel: BeliefState) -> Array:
        """only used in choose_action()"""
        mean, cov, t = bel
        mvg = tfd.MultivariateNormalFullCovariance(mean, cov)
        params = mvg.sample(seed=key)
        return params
