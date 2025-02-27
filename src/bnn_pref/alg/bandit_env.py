from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from bnn_pref.utils.type import CAR, CARL


class BanditEnvironment:
    def __init__(self, key, X, Y, opt_rewards):
        # Randomise dataset rows
        n_obs, n_features = X.shape

        new_ixs = jr.choice(key, n_obs, (n_obs,), replace=False)

        X = jnp.asarray(X)[new_ixs]
        Y = jnp.asarray(Y)[new_ixs]
        opt_rewards = jnp.asarray(opt_rewards)[new_ixs]

        self.contexts_ND = X
        self.labels_onehot_NA = Y
        self.opt_rewards_NA = opt_rewards

    def get_context(self, t) -> Float[Array, "n_features"]:
        return self.contexts_ND[t]

    def get_label(self, t) -> Float[Array, "n_actions"]:
        return self.labels_onehot_NA[t]

    def get_reward(self, t, action) -> float:
        label = self.labels_onehot_NA[t, action]
        return jnp.float32(label)

    def warmup(self, num_pulls: int) -> CARL:
        """
        Outputs:
            contexts: jnp.ndarray
                (n_pulls * n_actions, n_features)
            states: jnp.ndarray
                (n_pulls * n_actions, n_actions), one-hot
            actions: jnp.ndarray
                (n_pulls * n_actions,)
            rewards: jnp.ndarray
                (n_pulls * n_actions,)
        """
        num_steps, num_actions = self.labels_onehot_NA.shape
        # Create array of round-robin actions: 0, 1, 2, 0, 1, 2, 0, 1, 2, ...
        warmup_actions = jnp.arange(num_actions)
        warmup_actions = jnp.repeat(warmup_actions, num_pulls).reshape(num_actions, -1)
        actions = warmup_actions.reshape(-1, order="F").astype(jnp.int32)
        # num_warmup_actions, *_ = warmup_actions.shape

        time_steps = jnp.arange(len(actions))

        @partial(jax.vmap, in_axes=(0, 0))
        def get_contexts_and_rewards(t: int, a: int):
            context = self.get_context(t)
            label = self.get_label(t)
            reward = self.get_reward(t, a)
            return context, label, reward

        contexts, labels, rewards = get_contexts_and_rewards(time_steps, actions)

        return CARL(contexts, actions, rewards, labels)
