import itertools as it
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
    return sum(x.size for x in jax.tree.leaves(params_dict))


class MLPBlock(nn.Module):
    hidden_sizes: List[int]

    @nn.compact
    def __call__(self, c, x):
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = nn.leaky_relu(x)
        x = nn.Dense(1)(x)
        x = rearrange(x, "B 1 -> B")
        return c, x


class RewardNet(nn.Module):
    hidden_sizes: List[int]

    def setup(self):
        layers = [[nn.Dense(size), nn.leaky_relu] for size in self.hidden_sizes]
        layers += [[nn.Dense(1)]]
        self.layers = nn.Sequential(list(it.chain.from_iterable(layers)))

        # self.scanned_net = nn.scan(
        #     MLPBlock,
        #     variable_broadcast="params",
        #     split_rngs={"params": False},
        #     in_axes=1,  # scan over axis 1 (T)
        #     out_axes=1,  # output has axis 1 (T)
        #     length=None,
        # )(self.hidden_sizes)

    def __call__(self, x: B2TD) -> B2:
        r1 = self.predict_traj_return(x[:, 0])  # B
        r2 = self.predict_traj_return(x[:, 1])  # B
        logits = rearrange([r1, r2], "K B -> B K", K=2)  # B 2
        return logits

    def predict_traj_rewards(self, x: BTD) -> BT:
        # * split T n_splits chunks, avoid OOM
        B, T, D = x.shape
        n_splits = 5
        split_size = T // n_splits
        x_chunks = jnp.split(x, n_splits, axis=1)  # List[(B,Tp,D)]

        # preallocated version
        # out = jnp.empty((n_splits, B, split_size, 1))
        # for i, x_chunk in enumerate(x_chunks):
        #     x_chunk = self.layers(x_chunk)  # (B,Tp,D) -> (B,Tp,1)
        #     out = out.at[i, :, :, :].set(x_chunk)

        # list version
        out = [self.layers(x_chunk) for x_chunk in x_chunks]
        x = rearrange(out, "k B Tp D -> B (k Tp) D", k=n_splits, Tp=split_size)

        # * original, batch MLP over T dimension
        # x = self.layers(x)  # (B,T,D) -> (B,T,1)

        # todo: stability trick
        # if T > 1:
        #     x = nn.tanh(x) * 1.0
        return rearrange(x, "B T 1 -> B T")

    # def predict_traj_rewards_scan(self, x: BTD) -> B:
    #     _, rewards = self.scanned_net(None, x)
    #     return rewards

    def predict_traj_return(self, x: BTD) -> B:
        B, T, D = x.shape
        rewards = self.predict_traj_rewards(x)  # (B,T,D) -> (B,T)
        traj_return = rewards.sum(axis=1)
        # todo: stability trick
        traj_return /= T
        return traj_return


class MLP(nn.Module):
    num_arms: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(50, name="last_layer")(x))
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


if __name__ == "__main__":
    import jax.random as jr
    from flax import linen as nn
    from flax.core import Array, Scope, init, lift, unfreeze
    from jax import random

    # * https://github.com/google/flax/discussions/2067
    class MLPBlock2(nn.Module):
        @nn.compact
        def __call__(self, c, x):
            h = nn.Dense(features=10)(x)
            y = nn.Dense(features=1)(h)
            return c, y

    class ScanMLP(nn.Module):
        @nn.compact
        def __call__(self, c, xs):
            axis = 1
            scan = nn.scan(
                MLPBlock2,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=axis,
                out_axes=axis,
            )
            return scan(name="MLP")(c, xs)

    class TwoMLP(nn.Module):
        @nn.compact
        def __call__(self, c, xs):
            mlp1 = ScanMLP()
            mlp2 = ScanMLP()
            return mlp1(c, xs), mlp2(c, xs)

    # xs = jnp.ones((4, 10, 2))
    # scan_mlp = ScanMLP()
    # params_vars = scan_mlp.init(random.PRNGKey(1), (), xs)

    # cs, ys = scan_mlp.apply(params_vars, (), xs)
    # print(xs.shape)
    # print(ys.shape)

    # print(jax.tree.map(lambda x: x.shape, params_vars))
    # two_mlp = TwoMLP()
    # params_vars = two_mlp.init(random.PRNGKey(1), (), xs)
    # cs, ys = two_mlp.apply(params_vars, (), xs)

    # # * lifted tutorial
    # key = jr.key(0)
    # BatchDense = nn.vmap(
    #     nn.Dense,
    #     in_axes=0,
    #     out_axes=0,
    #     variable_axes={"params": 0},
    #     split_rngs={"params": True},
    # )

    # batch_dense = BatchDense(features=10)
    # # batch_dense = nn.Dense(features=10)
    # dummy = jnp.ones((1, 10))
    # key, *keys_init = jr.split(key, 5)
    # params_vars = batch_dense.init(jnp.array(keys_init), dummy)
    # ys = batch_dense.apply(params_vars, dummy)
    # print(ys.shape)
    # # import ipdb

    # ipdb.set_trace()
    import itertools as it

    key = jr.key(0)
    sizes = [32, 32]
    dummy = jnp.ones((1, 10))

    class MLPContainer(nn.Module):
        def setup(self):
            l = [[nn.Dense(size), nn.leaky_relu] for size in sizes] + [[nn.Dense(1)]]
            self.seq = nn.Sequential(list(it.chain.from_iterable(l)))
            print(self.seq)

        def __call__(self, x):
            return self.seq(x)

    mlp_container = MLPContainer()
    params_vars = mlp_container.init(key, dummy)
    ys = mlp_container.apply(params_vars, dummy)

    print(ys.shape)
    print(jax.tree.map(lambda x: x.shape, params_vars))
    print(nn.tabulate(mlp_container, key)(dummy))
    print(mlp_container)
    print(ys)
