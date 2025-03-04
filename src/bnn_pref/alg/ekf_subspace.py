from typing import Optional, Tuple, Union

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
    subspace2full_params,
    train_sgd,
)
from bnn_pref.utils.network import MLP, count_params
from bnn_pref.utils.type import CAR, CARL, BeliefState, D, Scalar, TwoD

tfd = tfp.distributions


class SubspaceNeuralBandit:
    def __init__(
        self,
        num_features: int,
        num_arms: int,
        model: Optional[nn.Module],
        opt,
        prior_noise: float,
        n_warmup_iterates: int = 1000,
        n_epochs: int = 1000,
        dynamics_noise: float = 0.0,
        obs_noise: float = 1.0,
        n_components: Union[float, int] = 0.9999,
        rnd_proj: bool = False,
    ):
        """
        Subspace Neural Bandit implementation.
        Parameters
        ----------
        num_arms: int
            Number of bandit arms / number of actions
        environment : Environment
            The environment to be used.
        model : flax.nn.Module
            The flax model to be used for the bandits. Note that this model is independent of the
            model architecture. The only constraint is that the last layer should have the same
            number of outputs as the number of arms.
        opt: flax.optim.Optimizer
            The optimizer to be used for training the model.
        learning_rate : float
            The learning rate for the optimizer used for the warmup phase.
        momentum : float
            The momentum for the optimizer used for the warmup phase.
        nepochs : int
            The number of epochs to be used for the warmup SGD phase.
        n_warmup_iterates: int
            How many of the SGD iterates to be treated / thrown away as warmup
        system_noise: float
            The system noise for the EKF.
        observation_noise: float
            The observation noise for the EKF.
        n_components: Union[float, int]
            The number of components to be used for the PCA.
        rnd_proj: bool
            Whether to use random projection.
        """
        self.num_features = num_features
        self.num_arms = num_arms

        if model is None:
            self.model = MLP(500, num_arms)
        else:
            try:
                self.model = model()
            except:
                self.model = model

        self.opt = opt
        self.prior_noise = prior_noise
        self.n_warmup_iterates = n_warmup_iterates
        self.n_epochs = n_epochs
        self.system_noise = dynamics_noise
        self.observation_noise = obs_noise
        self.n_components = n_components
        self.rnd_proj = rnd_proj
        self.context_dim = None

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

        initial_ts = TrainState.create(
            apply_fn=self.model.apply, params=initial_params, tx=self.opt
        )

        def loss_fn(params):
            logits_N2 = self.model.apply({"params": params}, contexts)
            loss = optax.softmax_cross_entropy(logits_N2, labels).mean()
            return loss, logits_N2

        warmup_ts, warmup_metrics = train_sgd(
            initial_ts, loss_fn=loss_fn, n_epochs=self.n_epochs
        )
        assert self.n_warmup_iterates < self.n_epochs

        # (n_iterates, n_full_params)
        thinned_samples = warmup_metrics["params"][::2]
        params_trace = thinned_samples[-self.n_warmup_iterates :]

        if self.rnd_proj:
            assert type(self.n_components) is int
            full_dim = params_trace.shape[-1]
            sub_dim = self.n_components
            key, proj_key = jr.split(key, 2)
            projection_matrix = generate_random_basis(proj_key, sub_dim, full_dim)
        else:
            pca = PCA(n_components=self.n_components)
            pca.fit(params_trace)
            sub_dim = pca.n_components_
            if type(self.n_components) is float:
                print(f"PCA found {sub_dim} components ({self.n_components=:.2%} var)")
            self.n_components = pca.n_components_
            projection_matrix = device_put(pca.components_)

        print(f"Full Space Param Count: {count_params(initial_params)}")
        print(f"Subspace   Param Count: {sub_dim}")
        Q = jnp.eye(sub_dim) * self.system_noise  # transition model noise
        R = jnp.eye(1) * self.observation_noise  # obs model noise

        params_full_init, reconstruct_tree_params = ravel_pytree(warmup_ts.params)
        params_subspace_init = jnp.zeros(sub_dim)
        covariance_subspace_init = jnp.eye(sub_dim) * self.prior_noise

        def sub2full_predict_reward(params_subspace, x: D) -> Scalar:
            params_full = subspace2full_params(
                params_subspace, projection_matrix, params_full_init
            )
            params = reconstruct_tree_params(params_full)
            # x = jnp.expand_dims(x, axis=0)  # (1, D)
            outputs = self.model.apply(
                {"params": params},
                x,
                method=self.model.predict_single,
            )  # (1,)
            return outputs

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

        ekf = ParamsNLGSSM(
            initial_mean=params_subspace_init,
            initial_covariance=covariance_subspace_init,
            dynamics_function=dynamics_fn,
            dynamics_covariance=Q,
            emission_function=emission_fn,
            emission_covariance=R,
        )
        self.ekf_params = ekf

        bel = BeliefState(params_subspace_init, covariance_subspace_init, 0)
        return bel

    def update_bel(
        self,
        bel: BeliefState,
        batch: CAR,
    ) -> BeliefState:
        prior_mean, prior_cov, t = bel
        context, action, reward = batch

        emission = rearrange(reward, " -> 1 1")
        inputs = jnp.concat(
            (
                jnp.ravel(context),  # (2 * D,)
                action[None],  # (1,)
            )
        )
        inputs = jnp.expand_dims(inputs, axis=0)  # (1, 2 * D + 1)

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

    def choose_action(self, key, bel: BeliefState, context: Float[Array, "2 D"]) -> int:
        # Thompson sampling strategy
        # Could also use epsilon greedy or UCB
        w = self.sample_params(key, bel)
        logits_2 = self.apply_model(w, context)
        action = logits_2.argmax()
        return action

    def sample_params(self, key, bel: BeliefState) -> Array:
        """only used in choose_action()"""
        mean, cov, t = bel
        mvg = tfd.MultivariateNormalFullCovariance(mean, cov)
        params = mvg.sample(seed=key)
        return params
