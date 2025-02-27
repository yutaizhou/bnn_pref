from typing import Optional, Tuple, Union

import jax.numpy as jnp
import optax
from dynamax.nonlinear_gaussian_ssm import ParamsNLGSSM, extended_kalman_filter
from einops import rearrange
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import device_put, jit
from jax.flatten_util import ravel_pytree
from jax.random import split
from jaxtyping import Array
from sklearn.decomposition import PCA
from tensorflow_probability.substrates import jax as tfp

from bnn_pref.alg.agent_utils import (
    generate_random_basis,
    subspace2full_params,
    train_sgd,
)
from bnn_pref.alg.train_utils import MLP
from bnn_pref.utils.type import CAR, CARL, BeliefState

tfd = tfp.distributions


class SubspaceNeuralBanditDynamax:
    def __init__(
        self,
        num_features: int,
        num_arms: int,
        model: Optional[nn.Module],
        opt,
        prior_noise_variance: float,
        nwarmup: int = 1000,
        nepochs: int = 1000,
        system_noise: float = 0.0,
        observation_noise: float = 1.0,
        n_components: Union[float, int] = 0.9999,
        random_projection: bool = False,
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
        nwarmup: int
            How many of the SGD iterates to be treated / thrown away as warmup
        system_noise: float
            The system noise for the EKF.
        observation_noise: float
            The observation noise for the EKF.
        n_components: Union[float, int]
            The number of components to be used for the PCA.
        random_projection: bool
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
        self.prior_noise_variance = prior_noise_variance
        self.nwarmup = nwarmup
        self.nepochs = nepochs
        self.system_noise = system_noise
        self.observation_noise = observation_noise
        self.n_components = n_components
        self.random_projection = random_projection
        self.context_dim = None

    def init_bel(self, key, warmup_data: CARL) -> BeliefState:
        """
        Run SGD on warmup data, get subspace projection matrix, initialize EKF
        """
        contexts, actions, _, labels = warmup_data
        self.context_dim = contexts.shape[-1]
        warmup_key, projection_key = split(key, 2)
        actions = actions.astype(int)
        dummy_context = jnp.ones((1, self.num_features))
        initial_params = self.model.init(warmup_key, dummy_context)["params"]

        initial_ts = TrainState.create(
            apply_fn=self.model.apply, params=initial_params, tx=self.opt
        )

        def loss_fn(params):
            pred_reward = self.model.apply({"params": params}, contexts)[:, actions]
            loss = optax.l2_loss(pred_reward, labels[:, actions]).mean()
            return loss, pred_reward

        warmup_ts, warmup_metrics = train_sgd(
            initial_ts, loss_fn=loss_fn, nepochs=self.nepochs
        )

        thinned_samples = warmup_metrics["params"][::2]  # (n_iterates, n_full_params)
        params_trace = thinned_samples[-self.nwarmup :]  # (n_iterates, n_full_params)

        if not self.random_projection:
            pca = PCA(n_components=self.n_components)
            pca.fit(params_trace)
            subspace_dim = pca.n_components_
            self.n_components = pca.n_components_
            projection_matrix = device_put(pca.components_)
        else:
            if type(self.n_components) is not int:
                raise ValueError(f"{self.n_components=} must be an integer")
            total_dim = params_trace.shape[-1]
            subspace_dim = self.n_components
            projection_matrix = generate_random_basis(
                key=projection_key, d=subspace_dim, D=total_dim
            )

        Q = jnp.eye(subspace_dim) * self.system_noise  # transition model noise
        R = jnp.eye(1) * self.observation_noise  # obs model noise

        params_full_init, reconstruct_tree_params = ravel_pytree(warmup_ts.params)
        params_subspace_init = jnp.zeros(subspace_dim)
        covariance_subspace_init = jnp.eye(subspace_dim) * self.prior_noise_variance

        def predict_rewards(params_subspace, context):
            """
            Project params from subspace to full space, then apply model
            """
            params_full = subspace2full_params(
                params_subspace, projection_matrix, params_full_init
            )
            params = reconstruct_tree_params(params_full)
            outputs = self.model.apply({"params": params}, context)
            return outputs

        self.predict_rewards = predict_rewards

        def dynamics_fn(params, inputs):
            """
            dynamics model constant dynamics
            """
            return params

        def emission_fn(params, inputs):
            """
            emission model where inputs is (N, D + 1)
            """
            context = inputs[..., : self.context_dim]
            action = inputs[..., self.context_dim].astype(int)
            return predict_rewards(params, context)[action, None]

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

        obs = rearrange(reward, " -> 1 1")
        inputs = jnp.concat((context, action[None]))
        inputs = rearrange(inputs, " d -> 1 d")

        self.ekf_params = self.ekf_params._replace(
            initial_mean=prior_mean,
            initial_covariance=prior_cov,
        )
        ekf_posterior = extended_kalman_filter(
            self.ekf_params, emissions=obs, inputs=inputs
        )

        posterior_mean = ekf_posterior.filtered_means[-1]
        posterior_cov = ekf_posterior.filtered_covariances[-1]
        bel = BeliefState(posterior_mean, posterior_cov, t + 1)
        return bel

    def choose_action(self, key, bel: BeliefState, context: Array) -> int:
        # Thompson sampling strategy
        # Could also use epsilon greedy or UCB
        w = self.sample_params(key, bel)
        predicted_reward = self.predict_rewards(w, context)
        action = predicted_reward.argmax()
        return action

    def sample_params(self, key, bel: BeliefState) -> Array:
        """only used in choose_action()"""
        mean, cov, t = bel
        mvg = tfd.MultivariateNormalFullCovariance(mean, cov)
        params = mvg.sample(seed=key)
        return params
