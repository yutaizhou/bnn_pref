from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from bnn_pref.utils.type import CARL, Q2, Q2TD


def retrieve(data, batch_idx: Float[Array, "B"]):
    """
    Take a (dynamic) batch of indices and return the corresponding elements from the original dataset.
    Designed to play nicely with jax.jit and jax.vmap
    """
    retrieve_fn = jax.vmap(
        partial(jax.lax.dynamic_index_in_dim, keepdims=False),
        in_axes=(None, 0),
    )
    return retrieve_fn(data, batch_idx)


class EKFEnvironment:
    def __init__(self, key, X: Q2TD, Y: Q2, opt_rewards=None):
        # Randomise dataset rows
        self.n_obs, *_, self.n_feats = X.shape
        self.n_actions = Y.shape[1]
        perm_idx = jr.permutation(key, jnp.arange(self.n_obs))
        X = jnp.asarray(X)[perm_idx]
        Y = jnp.asarray(Y)[perm_idx]
        if opt_rewards is not None:
            opt_rewards = jnp.asarray(opt_rewards)[perm_idx]

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
        Collect samples from the (already randomly permuted) dataset.
        Randomly sample actions so rewards are not all the same.

        Outputs:
            contexts: jnp.ndarray
                (n_warmups, n_features)
            actions: jnp.ndarray
                (n_warmups,)
            rewards: jnp.ndarray
                (n_warmups,)
            labels: jnp.ndarray
                (n_warmups, n_actions), one-hot
        """
        assert n_warmups <= self.n_obs, "more warmups than dataset size"
        # Create array of round-robin actions: 0, 1, 2, 0, 1, 2,  ...
        # actions = jnp.tile(jnp.arange(self.n_actions), n_warmups)
        idxes = jnp.arange(n_warmups)
        actions = jr.randint(key, shape=(n_warmups,), minval=0, maxval=self.n_actions)

        @partial(jax.vmap, in_axes=(0, 0))
        def get_contexts_and_rewards(idx: int, action: int):
            context = self.get_context(idx)
            label = self.get_label(idx)
            reward = self.get_reward(idx, action)
            return context, label, reward

        contexts, labels, rewards = get_contexts_and_rewards(idxes, actions)

        return CARL(contexts, actions, rewards, labels)


def get_batch_idxs(key, data_size: int, batch_size: int, n_iterates: int):
    """
    Simplified function version of BatchIndexManager.
    Generate batch indices for the exact number of iterations, reshuffling when needed.
    """
    all_idxs = []
    curr_idxs = None

    for i in range(n_iterates):
        if curr_idxs is None or len(curr_idxs) < batch_size:
            # Need to reshuffle
            key, key_shuffle = jr.split(key)
            curr_idxs = jr.permutation(key_shuffle, data_size)

        # Take the next batch
        batch = curr_idxs[:batch_size]
        curr_idxs = curr_idxs[batch_size:]
        all_idxs.append(batch)

    return jnp.stack(all_idxs)


class BatchIndexManager:
    """Manages batch indices for mini-batch training, handling shuffling and epoch transitions."""

    def __init__(self, key, data_size: int, batch_size: int):
        """
        Initialize the batch index manager.

        Parameters
        ----------
        key : PRNGKey
            Random key for shuffling
        data_size : int
            Total number of data points
        batch_size : int
            Size of each batch
        """
        self.key = key
        self.data_size = data_size
        self.batch_size = batch_size if batch_size != -1 else data_size
        self.curr_idxs = None
        self._n_batches_remaining = None  # For iterator interface

    def _shuffle(self):
        """Shuffle indices for a new epoch."""
        self.key, key_shuffle = jr.split(self.key)
        self.curr_idxs = jr.permutation(key_shuffle, self.data_size)

    def next_batch(self):
        """Get indices for the next batch, reshuffling if needed."""
        if self.curr_idxs is None or len(self.curr_idxs) < self.batch_size:
            self._shuffle()

        batch = self.curr_idxs[: self.batch_size]
        self.curr_idxs = self.curr_idxs[self.batch_size :]  # remove used indices
        return batch

    def get_n_batches(self, n: int):
        """Get indices for n batches."""
        batches = jnp.zeros((n, self.batch_size), dtype=jnp.int32)
        for i in range(n):
            batches = batches.at[i].set(self.next_batch())
        return batches

    def __iter__(self):
        """Make the manager iterable for a specified number of batches."""
        return self

    def __next__(self):
        """Get the next batch of indices."""
        if self._n_batches_remaining is None:
            raise RuntimeError("Must call take(n) before iterating")
        if self._n_batches_remaining <= 0:
            self._n_batches_remaining = None
            raise StopIteration

        self._n_batches_remaining -= 1
        return self.next_batch()

    def take(self, n: int):
        """Set up the manager to iterate over n batches."""
        self._n_batches_remaining = n
        return self
