from typing import Callable, Dict, List, Optional, Union

import flax.linen as nn
import ipdb
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

# to be used with ravel_pytree and unraveler
ParamsDict = Dict
ParamsFlat = Float[Array, "n_params"]


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
        def __call__(self, x, train: bool = False):
            return x

    return IdentityEncoder()


ENCODERS = {
    "resnet": setup_resnet_encoder(),
    "identity": setup_identity_encoder(),
}


class MLP(nn.Module):
    hidden_sizes: List[int]
    dropout_prob: Union[float, List[float]] = 0.0

    @nn.compact
    def __call__(self, x, train: bool = False):
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
                x = nn.Dropout(prob)(x, deterministic=not train)
        x = nn.Dense(1, name="last_layer")(x)
        return x


class PositionWiseMLP(nn.Module):
    hidden_sizes: List[int]
    n_splits: int = 1
    dropout_prob: Union[float, List[float]] = 0.0
    encoder_type: str = "identity"

    def setup(self):
        self.encoder = ENCODERS[self.encoder_type]
        self.mlp = MLP(hidden_sizes=self.hidden_sizes, dropout_prob=self.dropout_prob)

    def compute_pw_mlp(self, x, train: bool = False):
        """(B, T, dim) -> (B, T, 1)"""
        return self.mlp(x, train=train)

    def compute_embeddings(self, x, train: bool = False):
        """(B, T, ...) -> (B, T, embedding_dim)"""
        if x.ndim == 3:  # (B, T, D)
            x = self.encoder(x, train=train)
        elif x.ndim == 5:  # (B, T, H, W, C)
            B, T, H, W, C = x.shape
            x = rearrange(x, "B T H W C -> (B T) H W C", B=B, T=T)
            x = self.encoder(x, train=train)
            x = rearrange(x, "(B T) D -> B T D", B=B, T=T)
        return x

    def __call__(self, x, train: bool = False):
        """(B, T, ...) -> (B, T)"""

        def forward_block(x, train: bool):
            x = self.compute_embeddings(x, train=train)  # (B, T, embedding_dim)
            x = self.compute_pw_mlp(x, train=train)  # (B, T)
            return x

        if self.n_splits == 1:
            x = forward_block(x, train=train)
        else:
            T = x.shape[1]
            split_size = T // self.n_splits
            x_chunks = jnp.split(x, self.n_splits, axis=1)  # List[(B,S,D) * n_splits]
            out = [forward_block(x_chunk, train=train) for x_chunk in x_chunks]
            x = rearrange(out, "k B S -> B (k S)", k=self.n_splits, S=split_size)
        return jnp.squeeze(x, axis=-1)  # works also for batch-less TD -> T


class RewardNet(nn.Module):
    hidden_sizes: List[int]
    n_splits: int = 1
    dropout_prob: Union[float, List[float]] = 0.0
    encoder_type: str = "identity"

    def setup(self):
        assert self.n_splits > 0, f"{self.n_splits=} must be positive"
        assert 0 <= self.dropout_prob <= 1
        self.pw_encoder = PositionWiseMLP(
            hidden_sizes=self.hidden_sizes,
            n_splits=self.n_splits,
            dropout_prob=self.dropout_prob,
            encoder_type=self.encoder_type,
        )

    def __call__(self, x, train: bool = False):
        """
        Take batches of trajectory pairs, outputs returns for both trajectories, which
        can be then be softmaxed to get Bradley-Terry probabilities.
        (B, T, ...) -> (B, 2)
        """
        r1 = self.predict_traj_return(x[:, 0], train=train)  # BTD -> B
        r2 = self.predict_traj_return(x[:, 1], train=train)  # BTD -> B
        logits = rearrange([r1, r2], "K B -> B K", K=2)  # B 2
        return logits

    def predict_traj_return(self, x, train: bool = False):
        """(B, T, ...) -> (B,)"""
        B, T, *D = x.shape
        rewards = self.predict_traj_rewards(x, train=train)  # (B,T,D) -> (B,T)
        returns = rewards.sum(axis=1)  # (B,)
        returns /= T
        return returns

    def predict_traj_rewards(self, x, train: bool = False):
        """(B, T, ...) -> (B, T)"""
        return self.pw_encoder(x, train=train)

    def compute_embeddings(self, x, train: bool = False, agg: bool = False):
        """(B, T, ...) -> (B, T, embedding_dim)"""
        embd = self.pw_encoder.compute_embeddings(x, train=train)  # (B, T, E)
        if agg:
            embd = embd.mean(axis=1)  # (B, E)
        return embd

    def compute_return_from_agg_embeddings(self, x, train: bool = False):
        """
        (B, E) -> (B, 1)
        meant to be used after compute_embeddings(..., agg=True)
        """
        x = jnp.expand_dims(x, axis=1)  # (B, E) -> (B, 1, E)
        x = self.pw_encoder.compute_pw_mlp(x, train=train)  # (B, 1, 1)
        x = jnp.squeeze(x, axis=1)  # (B, 1, 1) -> (B, 1)
        return x  # (B, 1)


class LastLayerHelpers:
    """
    For last layer Bayesian methods, where we train only the last layer MLP params.
    Rest of the MLP and preceding encoder params are frozen.

    Assumes the following structure from model_def.init(...):

    variables["params"]["pw_encoder"]
        |-> ["mlp"]
            |-> ["Dense_0"]
            |-> ["Dense_1"]
            |-> ["last_layer"]
        |-> ["encoder"]
    """

    @staticmethod
    def recombine_params(
        last_param_flat: ParamsFlat,
        fixed_param_dict: ParamsDict,
        last_unravel_fn: Callable,
    ):
        last_layer_dict = last_unravel_fn(last_param_flat)
        params = {
            "mlp": {
                **fixed_param_dict["mlp"],
                "last_layer": last_layer_dict["mlp"]["last_layer"],
            }
        }
        if "encoder" in fixed_param_dict:
            params["encoder"] = fixed_param_dict["encoder"]
        return {"params": {"pw_encoder": params}}

    @staticmethod
    def get_frozen_params(variables: ParamsDict) -> ParamsDict:
        """Take variables dict, extract all but last layer MLP params"""
        mlp_params = variables["params"]["pw_encoder"]["mlp"]
        enc_params = variables["params"]["pw_encoder"].get("encoder", None)
        params = {
            "mlp": {k: v for k, v in mlp_params.items() if k != "last_layer"},
        }
        if enc_params is not None:
            params["encoder"] = enc_params
        return params

    @staticmethod
    def get_trainable_params(variables: ParamsDict) -> ParamsDict:
        ll_params = variables["params"]["pw_encoder"]["mlp"]["last_layer"]
        params = {
            "mlp": {"last_layer": ll_params},
        }

        return params


class ResNetHelpers:
    """
    For ResNet-based RewardNet, where we train only the MLP params, while freezing the ResNet encoder.

    Assumes the following structure from model_def.init(...):

    variables["params"]["pw_encoder"]
        |-> ["mlp"]
            |-> ["Dense_0"]
            |-> ["Dense_1"]
            |-> ["last_layer"]
        |-> ["encoder"]
    """

    @staticmethod
    def recombine_params(
        trainable_params: Union[ParamsDict, ParamsFlat],
        fixed_params: ParamsDict,
        unraveler: Optional[Callable] = None,
    ) -> ParamsDict:
        # if no unraveler, assume trainable_params is a dict. o.w. it's a flat array
        trainable_dict = (
            unraveler(trainable_params) if unraveler is not None else trainable_params
        )
        params = {
            "mlp": trainable_dict["mlp"],
            "encoder": fixed_params["encoder"],
        }
        return {"params": {"pw_encoder": params}}

    @staticmethod
    def get_frozen_params(variables: ParamsDict) -> ParamsDict:
        """Take variables dict, extract only ResNet encoder params"""
        enc_params = variables["params"]["pw_encoder"]["encoder"]
        params = {"encoder": enc_params}
        return params

    @staticmethod
    def get_trainable_params(variables: ParamsDict) -> ParamsDict:
        mlp_params = variables["params"]["pw_encoder"]["mlp"]
        params = {"mlp": mlp_params}

        return params


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    T = 6
    D = 3
    H, W, C = 84, 84, 3
    model_def = RewardNet(
        hidden_sizes=[128, 128],
        n_splits=1,
        dropout_prob=0.0,
        encoder_type="identity",
    )
    input = jnp.ones((1, 2, T, D))
    params = model_def.init(key, input)["params"]
    # print(model_def.tabulate(key, input))

    # * ResNet based RewardNet
    model_def = RewardNet(
        hidden_sizes=[128, 128],
        n_splits=1,
        dropout_prob=0.0,
        encoder_type="resnet",
    )
    input = jnp.ones((1, 2, T, H, W, C))
    variables = model_def.init(key, input)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    out = model_def.apply(variables, input)
    # print(model_def.tabulate(key, input))
    print("========== RewardNet output ==========")
    print(f"{input.shape=}")
    print(f"{out.shape=}")

    # * ResNet by itself
    input = jnp.ones((1, T, H, W, C))
    input = rearrange(input, "B T H W C -> (B T) H W C", B=1, T=T)
    resnet_def = setup_resnet_encoder()
    resnet_variables = resnet_def.init(key, input)
    resnet_params = resnet_variables["params"]
    resnet_batch_stats = resnet_variables["batch_stats"]
    resnet_out = resnet_def.apply(resnet_variables, input, train=False)
    # print(resnet_def.tabulate(key, input))
    print("========== ResNet output ==========")
    print(f"{input.shape=}")
    print(f"{resnet_out.shape=}")

    # * Compute ResNet embeddings, followed by RewardNet return prediction
    B = 2
    input = jnp.ones((B, T, H, W, C))
    embedding = model_def.apply(
        variables, input, train=False, method=model_def.compute_embeddings, agg=True
    )  # (B, T, H, W, C) -> (B, E)
    return_from_embeddings = model_def.apply(
        variables,
        embedding,
        train=False,
        method=model_def.compute_return_from_agg_embeddings,
    )  # (B, E) -> (B, 1)
    print("========== Embedding output ==========")
    print(f"{input.shape=}")
    print(f"{embedding.shape=}")
    print(f"{return_from_embeddings.shape=}")
