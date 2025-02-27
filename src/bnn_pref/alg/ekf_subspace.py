import jax.numpy as jnp
import optax
from dynamax.nonlinear_gaussian_ssm import ParamsNLGSSM, extended_kalman_filter
from einops import rearrange
from flax.training import train_state
from jax import device_put, jit
from jax.flatten_util import ravel_pytree
from jax.random import split
from sklearn.decomposition import PCA
from tensorflow_probability.substrates import jax as tfp

from bnn_pref.alg.agent_utils import (
    convert_params_from_subspace_to_full,
    generate_random_basis,
    train,
)
from bnn_pref.alg.train_utils import MLP

tfd = tfp.distributions


class SubspaceNeuralBanditDynamax:
    def __init__(
        self,
        num_features,
        num_arms,
        model,
        opt,
        prior_noise_variance,
        nwarmup=1000,
        nepochs=1000,
        system_noise=0.0,
        observation_noise=1.0,
        n_components=0.9999,
        random_projection=False,
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

    def init_bel(self, key, contexts, states, actions, rewards):
        """
        contexts: (num_steps, num_features)
        states: (num_steps, num_actions) # predict class label given features
        actions: (num_steps,) # taken actions, if warmup, should be round-robin
        rewards: (num_steps,)
        """
        self.context_dim = contexts.shape[-1]
        warmup_key, projection_key = split(key, 2)
        actions = actions.astype(int)
        initial_params = self.model.init(
            warmup_key,
            jnp.ones((1, self.num_features)),
        )["params"]

        initial_train_state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=initial_params, tx=self.opt
        )

        def loss_fn(params):
            pred_reward = self.model.apply({"params": params}, contexts)[:, actions]
            loss = optax.l2_loss(pred_reward, states[:, actions]).mean()
            return loss, pred_reward

        warmup_state, warmup_metrics = train(
            initial_train_state, loss_fn=loss_fn, nepochs=self.nepochs
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
                raise ValueError(
                    f"n_components must be an integer, got {self.n_components}"
                )
            total_dim = params_trace.shape[-1]
            subspace_dim = self.n_components
            projection_matrix = generate_random_basis(
                projection_key, subspace_dim, total_dim
            )

        Q = jnp.eye(subspace_dim) * self.system_noise  # transition model noise
        R = jnp.eye(1) * self.observation_noise  # obs model noise

        params_full_init, reconstruct_tree_params = ravel_pytree(warmup_state.params)
        params_subspace_init = jnp.zeros(subspace_dim)
        covariance_subspace_init = jnp.eye(subspace_dim) * self.prior_noise_variance

        def predict_rewards(params_subspace_sample, context):
            """
            params_full_init and projection_matrix do not change. only the subspace
            samples change.
            """
            params_full = convert_params_from_subspace_to_full(
                params_subspace_sample,
                projection_matrix,
                params_full_init,
            )
            params = reconstruct_tree_params(params_full)
            outputs = self.model.apply({"params": params}, context)
            return outputs

        self.predict_rewards = predict_rewards

        def fz(params, inputs):
            """state transition model"""
            return params

        def fx(params, inputs):
            """observation model"""
            context = inputs[..., : self.context_dim]
            action = inputs[..., self.context_dim].astype(int)
            return predict_rewards(params, context)[action, None]

        ekf = ParamsNLGSSM(
            initial_mean=params_subspace_init,
            initial_covariance=covariance_subspace_init,
            dynamics_function=fz,
            dynamics_covariance=Q,
            emission_function=fx,
            emission_covariance=R,
        )
        self.ekf_params = ekf

        bel = (params_subspace_init, covariance_subspace_init, 0)
        return bel

    def update_bel(self, bel, context, action, reward):
        mean, cov, t = bel

        obs = rearrange(reward, " -> 1 1")
        inputs = jnp.concat((context, action[None]))
        inputs = rearrange(inputs, " d -> 1 d")

        self.ekf_params = self.ekf_params._replace(
            initial_mean=mean,
            initial_covariance=cov,
        )
        ekf_posterior = extended_kalman_filter(
            self.ekf_params, emissions=obs, inputs=inputs
        )

        new_mean = ekf_posterior.filtered_means[-1]
        new_cov = ekf_posterior.filtered_covariances[-1]
        bel = (new_mean, new_cov, t + 1)
        return bel

    def choose_action(self, key, bel, context):
        # Thompson sampling strategy
        # Could also use epsilon greedy or UCB
        w = self.sample_params(key, bel)
        predicted_reward = self.predict_rewards(w, context)
        action = predicted_reward.argmax()
        return action

    def sample_params(self, key, bel):
        """only used in choose_action()"""
        params_subspace, covariance_subspace, t = bel
        mv_normal = tfd.MultivariateNormalFullCovariance(
            loc=params_subspace,
            covariance_matrix=covariance_subspace,
        )
        params_subspace = mv_normal.sample(seed=key)
        return params_subspace
