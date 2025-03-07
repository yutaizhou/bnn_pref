from typing import List

import einops
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
    return sum(x.size for x in jax.tree_util.tree_leaves(params_dict))


class RewardNet(nn.Module):
    hidden_sizes: List[int]

    def setup(self):
        self.layers = [nn.Dense(size) for size in self.hidden_sizes] + [nn.Dense(1)]

    def __call__(self, x: B2TD) -> B2:
        r1 = self.predict_traj_return(x[:, 0])  # B
        r2 = self.predict_traj_return(x[:, 1])  # B
        return rearrange([r1, r2], "K B -> B K", K=2)  # B 2

    def predict_traj_rewards(self, x: BTD) -> BT:
        B, T, D = x.shape
        for layer in self.layers:
            x = layer(x)
            if layer != self.layers[-1]:
                x = nn.relu(x)
        # for stability in computing logits for Bradley-Terry
        if T > 1:
            x = nn.tanh(x) * 0.5
        return rearrange(x, "B T 1 -> B T")

    def predict_traj_return(self, x: BTD) -> B:
        B, T, D = x.shape
        rewards = self.predict_traj_rewards(x)  # (B,T,D) -> (B,T)
        traj_return = rewards.sum(axis=1)
        traj_return /= jnp.sqrt(T)
        return traj_return


class MLP(nn.Module):
    num_arms: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(50, name="last_layer")(x))
        x = nn.Dense(self.num_arms)(x)
        return x


class MLPWide(nn.Module):
    num_arms: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(200)(x))
        x = nn.relu(nn.Dense(200, name="last_layer")(x))
        x = nn.Dense(self.num_arms)(x)
        return x


class LeNet5(nn.Module):
    num_arms: int

    @nn.compact
    def __call__(self, x):
        x = x if len(x.shape) > 1 else x[None, :]
        x = x.reshape((x.shape[0], 28, 28, 1))
        x = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(features=16, kernel_size=(5, 5), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=120)(x)
        x = nn.relu(x)
        x = nn.Dense(features=84, name="last_layer")(x)  # There are 10 classes in MNIST
        x = nn.relu(x)
        x = nn.Dense(features=self.num_arms)(x)
        return x.squeeze()
