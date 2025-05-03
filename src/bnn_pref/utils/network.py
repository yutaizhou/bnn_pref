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

        # todo: stability trick
        # if T > 1:
        #     x = nn.tanh(x)
        # return rearrange(x, "B T 1 -> B T")
        return jnp.squeeze(x, axis=-1)  # works also for TD -> T

    def predict_traj_return(self, x: BTD) -> B:
        B, T, D = x.shape
        rewards = self.predict_traj_rewards(x)  # (B,T,D) -> (B,T)
        returns = rewards.sum(axis=1)  # (B,)
        # todo: stability trick
        returns /= T
        return returns


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
