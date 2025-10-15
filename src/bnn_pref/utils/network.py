import itertools as it
from typing import List, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float

B2D = Float[Array, "batch 2 dim"]
BD = Float[Array, "batch dim"]
B1 = Float[Array, "batch 1 "]
B2 = Float[Array, "batch 2 "]

B2TD = Float[Array, "batch 2 steps dim"]
BTD = Float[Array, "batch steps dim"]
BT = Float[Array, "batch steps"]
B = Float[Array, "batch"]


def count_params(params_dict: dict) -> int:
    """
    params_dict = model.init(key, dummy)["params"]
    """
    return sum(x.size for x in jax.tree.leaves(params_dict))


def isfinite_param_pytree(param):
    """
    pytree param finiteness check. works on both pytree and arrays, count number of non-finite params
    """
    isfinite = jax.tree.map(lambda x: jnp.isfinite(x).all(), param)
    # return sum(si for si in isfinite if ~si)
    return jax.tree.all(isfinite)


def isfinite_param(param):
    return jnp.isfinite(param).all()


default_init = nn.initializers.xavier_uniform


class PositionWiseMLP(nn.Module):
    hidden_sizes: List[int]
    n_splits: int = 1
    dropout_prob: Union[float, List[float]] = 0.0

    @nn.compact
    def __call__(self, x: BTD, deterministic: bool) -> BT:
        """
        BTD -> BT
        Applies position-wise MLP to get per-timestep rewards, with optional dropout.
        If n_splits > 1, splits T into `n_splits` (divisible) chunks, avoid OOM for large T.

        T: trajectory length
        S: segment length (if n_splits > 1)
        """

        def forward_block(x: BTD, deterministic: bool) -> Float[Array, "B T 1"]:
            n_hidden = len(self.hidden_sizes)
            dropout_probs = (
                [self.dropout_prob] * n_hidden
                if isinstance(self.dropout_prob, float)
                else self.dropout_prob
            )
            for i in range(n_hidden):
                x = nn.Dense(self.hidden_sizes[i], kernel_init=default_init())(x)
                x = nn.leaky_relu(x)
                prob = dropout_probs[i]
                if prob > 0:
                    x = nn.Dropout(prob)(x, deterministic=deterministic)
            x = nn.Dense(1)(x)
            return x

        if self.n_splits == 1:
            x = forward_block(x, deterministic=deterministic)
        else:
            T = x.shape[1]
            split_size = T // self.n_splits
            x_chunks = jnp.split(x, self.n_splits, axis=1)  # List[(B,S,D) * n_splits]
            out = [
                forward_block(x_chunk, deterministic=deterministic)
                for x_chunk in x_chunks
            ]
            x = rearrange(out, "k B S 1 -> B (k S) 1", k=self.n_splits, S=split_size)
        return jnp.squeeze(x, axis=-1)  # works also for batch-less TD -> T


class RewardNet(nn.Module):
    hidden_sizes: List[int]
    n_splits: int = 1
    dropout_prob: Union[float, List[float]] = 0.0

    def setup(self):
        assert self.n_splits > 0, f"{self.n_splits=} must be positive"
        assert 0 <= self.dropout_prob <= 1
        self.pw_mlp = PositionWiseMLP(
            hidden_sizes=self.hidden_sizes,
            n_splits=self.n_splits,
            dropout_prob=self.dropout_prob,
        )

    def __call__(self, x: B2TD, deterministic: bool = True) -> B2:
        """
        Take batches of trajectory pairs, outputs returns for both trajectories
        """
        r1 = self.predict_traj_return(x[:, 0], deterministic=deterministic)  # BTD -> B
        r2 = self.predict_traj_return(x[:, 1], deterministic=deterministic)  # BTD -> B
        logits = rearrange([r1, r2], "K B -> B K", K=2)  # B 2
        return logits

    def predict_traj_return(self, x: BTD, deterministic: bool = True) -> B:
        B, T, D = x.shape
        rewards = self.predict_traj_rewards(
            x, deterministic=deterministic
        )  # (B,T,D) -> (B,T)
        returns = rewards.sum(axis=1)  # (B,)
        returns /= T
        return returns

    def predict_traj_rewards(self, x: BTD, deterministic: bool = True) -> BT:
        return self.pw_mlp(x, deterministic=deterministic)
