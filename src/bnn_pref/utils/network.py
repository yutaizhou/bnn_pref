import itertools as it
from typing import List

import einops
import flax.linen as nn
import ipdb
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


class RewardNet(nn.Module):
    hidden_sizes: List[int]
    n_splits: int = 1

    def setup(self):
        assert self.n_splits > 0, f"{self.n_splits=} must be positive"
        layers = [[nn.Dense(size), nn.leaky_relu] for size in self.hidden_sizes]
        layers += [[nn.Dense(1)]]
        self.layers = nn.Sequential(list(it.chain.from_iterable(layers)))

    def __call__(self, x: B2TD) -> B2:
        """
        Take batches of trajectory pairs, outputs returns for both trajectories
        """
        r1 = self.predict_traj_return(x[:, 0])  # B
        r2 = self.predict_traj_return(x[:, 1])  # B
        logits = rearrange([r1, r2], "K B -> B K", K=2)  # B 2
        return logits

    def predict_traj_rewards(self, x: BTD) -> BT:
        """
        batch MLP over T dimension
        if n_splits > 1, split T into `n_splits` (divisible) chunks, avoid OOM
        """
        if self.n_splits == 1:
            x = self.layers(x)  # (B,T,D) -> (B,T,1)
        else:
            T = x.shape[1]
            split_size = T // self.n_splits
            x_chunks = jnp.split(x, self.n_splits, axis=1)  # List[(B,S,D) * n_splits]
            out = [self.layers(x_chunk) for x_chunk in x_chunks]
            x = rearrange(out, "k B S 1 -> B (k S) 1", k=self.n_splits, S=split_size)
        return jnp.squeeze(x, axis=-1)  # works also for batch-less TD -> T

    def predict_traj_return(self, x: BTD) -> B:
        B, T, D = x.shape
        rewards = self.predict_traj_rewards(x)  # (B,T,D) -> (B,T)
        returns = rewards.sum(axis=1)  # (B,)
        returns /= T
        return returns
