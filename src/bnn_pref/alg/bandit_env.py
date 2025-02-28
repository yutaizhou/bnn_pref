from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from bnn_pref.utils.type import CAR, CARL, Q2, Q2D


class BanditEnvironment:
    def __init__(self, key, X: Q2D, Y: Q2, opt_rewards=None):
        # Randomise dataset rows
        self.n_obs, _, self.n_feats = X.shape
        self.n_actions = Y.shape[1]
        new_ixs = jr.choice(key, self.n_obs, (self.n_obs,), replace=False)
        X = jnp.asarray(X)[new_ixs]
        Y = jnp.asarray(Y)[new_ixs]
        if opt_rewards is not None:
            opt_rewards = jnp.asarray(opt_rewards)[new_ixs]

        self.contexts = X
        self.labels_onehot = Y
        self.opt_rewards = opt_rewards

    def get_context(self, t) -> Float[Array, "2 D"]:
        return self.contexts[t]

    def get_label(self, t) -> Float[Array, "n_actions"]:
        return self.labels_onehot[t]

    def get_reward(self, t, action) -> float:
        label = self.labels_onehot[t, action]
        return jnp.float32(label)

    def warmup(self, key, n_warmups: int) -> CARL:
        """
        collect random samples from the dataset

        Outputs:
            contexts: jnp.ndarray
                (n_pulls * n_actions, n_features)
            actions: jnp.ndarray
                (n_pulls * n_actions,)
            rewards: jnp.ndarray
                (n_pulls * n_actions,)
            labels: jnp.ndarray
                (n_pulls * n_actions, n_actions), one-hot
        """
        assert n_warmups <= self.n_obs, "more warmups than dataset size"
        # Create array of round-robin actions: 0, 1, 2, 0, 1, 2,  ...
        # actions = jnp.tile(jnp.arange(self.n_actions), n_warmups)
        actions = jr.randint(key, shape=(n_warmups,), minval=0, maxval=self.n_actions)
        time_steps = jnp.arange(len(actions))

        @partial(jax.vmap, in_axes=(0, 0))
        def get_contexts_and_rewards(t: int, a: int):
            context = self.get_context(t)
            label = self.get_label(t)
            reward = self.get_reward(t, a)
            return context, label, reward

        contexts, labels, rewards = get_contexts_and_rewards(time_steps, actions)

        return CARL(contexts, actions, rewards, labels)
