from typing import List, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

# from flaxmodels import ResNet18 as FMResNet18
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


def setup_resnet_encoder():
    from flaxmodels import ResNet18  # actual training run

    # from flaxmodels.flaxmodels import ResNet18  # run network.py

    return ResNet18(
        output="embeddings",
        pretrained="imagenet",
        use_classifier_head=False,
    )


def setup_identity_encoder():
    class IdentityEncoder(nn.Module):
        def __call__(self, x):
            return x

    return IdentityEncoder()


ENCODERS = {
    "resnet": setup_resnet_encoder(),
    "identity": setup_identity_encoder(),
}


class PositionWiseMLP(nn.Module):
    hidden_sizes: List[int]
    n_splits: int = 1
    dropout_prob: Union[float, List[float]] = 0.0
    encoder: nn.Module = ENCODERS["identity"]

    @nn.compact
    def __call__(self, x: BTD, deterministic: bool = True) -> BT:
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

            # * encode input: (B,T,state_dim) or (B,T,H,W,C) -> (B,T,encoder_dim)
            if x.ndim == 3:  # (B, T, D)
                x = self.encoder(x)
            elif x.ndim == 5:  # (B, T, H, W, C)
                B, T, H, W, C = x.shape
                x = rearrange(x, "B T H W C -> (B T) H W C", B=B, T=T)
                x = self.encoder(x, train=not deterministic)
                x = rearrange(x, "(B T) D -> B T D", B=B, T=T)

            # * forward pass: (B,T,encoder_dim) -> (B,T,1)
            for i in range(n_hidden):
                x = nn.Dense(self.hidden_sizes[i], kernel_init=default_init())(x)
                x = nn.leaky_relu(x)
                prob = dropout_probs[i]
                if prob > 0:
                    x = nn.Dropout(prob)(x, deterministic=deterministic)
            x = nn.Dense(1, name="last_layer")(x)
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
    encoder_type: str = "identity"

    def setup(self):
        assert self.n_splits > 0, f"{self.n_splits=} must be positive"
        assert 0 <= self.dropout_prob <= 1
        encoder = ENCODERS[self.encoder_type]
        self.pw_encoder = PositionWiseMLP(
            hidden_sizes=self.hidden_sizes,
            n_splits=self.n_splits,
            dropout_prob=self.dropout_prob,
            encoder=encoder,
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
        B, T, *D = x.shape
        rewards = self.predict_traj_rewards(
            x, deterministic=deterministic
        )  # (B,T,D) -> (B,T)
        returns = rewards.sum(axis=1)  # (B,)
        returns /= T
        return returns

    def predict_traj_rewards(self, x: BTD, deterministic: bool = True) -> BT:
        return self.pw_encoder(x, deterministic=deterministic)

    # * For last layer Bayesian methods: last layer must be named "last_layer"
    @staticmethod
    def recombine_params(last_param_flat, fixed_param_dict, last_unravel_fn):
        last_layer_dict = last_unravel_fn(last_param_flat)
        # Combine fixed params with reconstructed last layer
        params = {
            **fixed_param_dict["params"],
            "last_layer": last_layer_dict["params"]["last_layer"],
        }
        return {"params": {"pw_encoder": params}}

    @staticmethod
    def get_fixed_params(params):
        mlp_params = params["params"]["pw_encoder"]
        params = {k: v for k, v in mlp_params.items() if k != "last_layer"}
        return {"params": params}

    @staticmethod
    def get_last_layer_params(params):
        mlp_params = params["params"]["pw_encoder"]
        params = {"last_layer": mlp_params["last_layer"]}
        return {"params": params}


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    model_def = RewardNet(
        hidden_sizes=[128, 128],
        n_splits=1,
        dropout_prob=0.0,
        encoder_type="identity",
    )
    T, D = 50, 3
    state_input = jnp.ones((1, 2, T, D))
    params = model_def.init(key, state_input)["params"]
    print(model_def.tabulate(key, state_input))

    model_def = RewardNet(
        hidden_sizes=[128, 128],
        n_splits=1,
        dropout_prob=0.0,
        encoder_type="resnet",
    )
    H, W, C = 84, 84, 3
    state_input = jnp.ones((1, 2, T, H, W, C))
    params = model_def.init(key, state_input)["params"]
    print(model_def.tabulate(key, state_input))
