from typing import List

import einops
import flax.linen as nn
import jax.numpy as jnp


class RewardNet(nn.Module):
    hidden_sizes: List[int]

    def setup(self):
        self.layers = [nn.Dense(size) for size in self.hidden_sizes] + [nn.Dense(1)]

    def __call__(self, x):  # Q2D
        r1 = self.predict_single(x[:, 0])  # N1
        r2 = self.predict_single(x[:, 1])
        return jnp.concatenate([r1, r2], axis=1)  # N2

    def predict_single(self, x):
        # x is (N, D)
        for layer in self.layers:
            x = layer(x)
            if layer != self.layers[-1]:
                x = nn.relu(x)
        return x


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
