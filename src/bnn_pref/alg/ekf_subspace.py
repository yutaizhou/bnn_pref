from typing import Optional, Tuple, Union

import einops
import jax.numpy as jnp
import jax.random as jr
import optax
from dynamax.nonlinear_gaussian_ssm import ParamsNLGSSM, extended_kalman_filter
from einops import rearrange
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import device_put, jit
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float
from sklearn.decomposition import PCA
from tensorflow_probability.substrates import jax as tfp

from bnn_pref.alg.agent_utils import (
    generate_random_basis,
    run_sgd,
    subspace2full_params,
)
from bnn_pref.utils.network import MLP, count_params
from bnn_pref.utils.type import CAR, CARL, BeliefState, D, Scalar, TwoD

tfd = tfp.distributions


class SubspaceNeuralBandit:
    def __init__(
        self,
        num_features: int,
        model: nn.Module,
        opt,
        warm_epochs: int = 1000,
        warm_burns: int = 1000,
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
        num_features : int
            The number of input features of the model.
        model : flax.nn.Module
            The flax model to be used for the bandits.
        opt: flax.optim.Optimizer
            The optimizer to be used for training the model.
        warm_epochs : int
            The number of SGD epochs to be used for the warmup phase.
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
        """
        self.num_features = num_features
        self.model = model
        self.opt = opt
        self.prior_noise = prior_noise
        self.warm_burns = warm_burns
        self.warm_epochs = warm_epochs
        self.system_noise = dynamics_noise
        self.observation_noise = obs_noise
        self.sub_dim = sub_dim
        self.rnd_proj = rnd_proj
        self.context_dim = None

        assert self.warm_burns < self.warm_epochs

    def init_bel(self, key, warmup_data: CARL) -> BeliefState:
        """
        Run SGD on warmup data, get subspace projection matrix, initialize EKF
        contexts: Q2D
        actions: Q
        rewards: Q
        labels: Q2 (onehot)
        """
        contexts, _, _, labels = warmup_data
        self.n_feats = contexts.shape[-1]
        key, model_key = jr.split(key, 2)
        dummy_context = jnp.ones((1, 2, self.n_feats))
        initial_params = self.model.init(model_key, dummy_context)["params"]
        # print(nn.tabulate(self.model, model_key)(dummy_context))

        ts = TrainState.create(
            apply_fn=self.model.apply, params=initial_params, tx=self.opt
        )

        def loss_fn(params):
            logits_N2 = self.model.apply({"params": params}, contexts)
            loss = optax.softmax_cross_entropy(logits_N2, labels).mean()
            return loss, logits_N2

        warm_ts, warm_metrics = run_sgd(ts, loss_fn=loss_fn, n_epochs=self.warm_epochs)
        params_trace = warm_metrics["params"][self.warm_burns :: 2]

        if self.rnd_proj:
            assert type(self.sub_dim) is int
            full_dim = params_trace.shape[-1]
            sub_dim = self.sub_dim
            key, proj_key = jr.split(key, 2)
            projection_matrix = generate_random_basis(proj_key, sub_dim, full_dim)
        else:
            pca = PCA(n_components=self.sub_dim)
            pca.fit(params_trace)
            sub_dim = pca.n_components_
            if type(self.sub_dim) is float:
                print(f"PCA found {sub_dim} components ({self.sub_dim=:.2%} var)")
            self.sub_dim = pca.n_components_
            projection_matrix = device_put(pca.components_)

        print(f"Full Space Param Count: {count_params(initial_params)}")
        print(f"Subspace   Param Count: {sub_dim}")

        params_full_init, reconstruct_tree_params = ravel_pytree(warm_ts.params)

        def sub2full_predict_reward(params_subspace, x: D) -> Scalar:
            params_full = subspace2full_params(
                params_subspace, projection_matrix, params_full_init
            )
            params = reconstruct_tree_params(params_full)
            outputs = self.model.apply(
                {"params": params},
                x,
                method=self.model.predict_single,
            )
            return outputs  # (1,)

        def sub2full_apply(
            params_subspace,
            context: Float[Array, "2 D"],
        ) -> Float[Array, "2"]:
            """
            Project params from subspace to full space, then apply model
            """
            params_full = subspace2full_params(
                params_subspace, projection_matrix, params_full_init
            )
            params = reconstruct_tree_params(params_full)
            context = jnp.expand_dims(context, axis=0)  # (1, 2, D)
            outputs = self.model.apply({"params": params}, context)  # (1,2)
            return outputs.squeeze(0)  # (2,)

        self.predict_reward = sub2full_predict_reward
        self.apply_model = sub2full_apply

        def dynamics_fn(params, inputs):
            """
            dynamics model constant dynamics
            """
            return params

        def emission_fn(params, inputs):
            """
            emission model where inputs is (N, D + 1)
            """
            context = inputs[..., :-1].reshape(2, -1)
            action = inputs[..., -1].astype(int)
            return sub2full_apply(params, context)[action, None]

        params_subspace_init = jnp.zeros(sub_dim)
        Sigma = jnp.eye(sub_dim) * self.prior_noise
        Q = jnp.eye(sub_dim) * self.system_noise  # transition model noise
        R = jnp.eye(1) * self.observation_noise  # obs model noise
        ekf = ParamsNLGSSM(
            initial_mean=params_subspace_init,
            initial_covariance=Sigma,
            dynamics_function=dynamics_fn,
            dynamics_covariance=Q,
            emission_function=emission_fn,
            emission_covariance=R,
        )
        self.ekf_params = ekf

        bel = BeliefState(params_subspace_init, Sigma, 0)
        return bel

    def update_bel(
        self,
        bel: BeliefState,
        batch: CAR,
    ) -> BeliefState:
        prior_mean, prior_cov, t = bel
        context_2D, action, reward = batch

        context = rearrange(context_2D, "Two D -> (Two D)")
        action = rearrange(action, " -> 1")
        inputs = rearrange(jnp.concat((context, action)), "CA -> 1 CA")
        emission = rearrange(reward, " -> 1 1")

        self.ekf_params = self.ekf_params._replace(
            initial_mean=prior_mean,
            initial_covariance=prior_cov,
        )
        ekf_posterior = extended_kalman_filter(
            self.ekf_params, emissions=emission, inputs=inputs
        )

        posterior_mean = ekf_posterior.filtered_means[-1]
        posterior_cov = ekf_posterior.filtered_covariances[-1]
        bel = BeliefState(posterior_mean, posterior_cov, t + 1)
        return bel

    def choose_action(
        self,
        key,
        bel: BeliefState,
        context_2D: Float[Array, "2 D"],
    ) -> int:
        # Thompson sampling strategy
        # Could also use epsilon greedy or UCB
        w = self.sample_params(key, bel)
        logits_2 = self.apply_model(w, context_2D)
        action = logits_2.argmax()
        return action

    def sample_params(self, key, bel: BeliefState) -> Array:
        """only used in choose_action()"""
        mean, cov, t = bel
        mvg = tfd.MultivariateNormalFullCovariance(mean, cov)
        params = mvg.sample(seed=key)
        return params
