from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, value_and_grad
from jax.flatten_util import ravel_pytree
from jax.lax import scan
from jaxtyping import Array, Float, Int

from bnn_pref.data.data_env import BatchIndexManager, PreferenceEnv, retrieve
from bnn_pref.utils.type import QueryData


class Agent(ABC):
    @abstractmethod
    def init_bel(self, key, warmup_data: QueryData):
        pass

    @abstractmethod
    def update_bel(self, bel, batch: QueryData):
        pass

    @abstractmethod
    def compute_next_query(
        self, key, bel, env: PreferenceEnv, pool_idxes_Q: Int[Array, "Q"]
    ) -> int:
        pass

    @abstractmethod
    def compute_postpred(
        self,
        key,
        bel,
        items_NTD: Float[Array, "N T D"],
        query_idxs_Q2: Float[Array, "Q 2"],
    ) -> Float[Array, "Q 2"]:
        pass

    @staticmethod
    @abstractmethod
    def get_hydra_config(cls_cfg: Dict):
        pass


def sub2full_params_flat(
    params_subspace: Float[Array, "sub_dim"],
    proj_matrix: Float[Array, "sub_dim full_dim"],
    params_full: Float[Array, "full_dim"],
) -> Float[Array, "full_dim"]:
    params = params_subspace @ proj_matrix + params_full
    return params


def generate_random_basis(key, d: int, D: int):
    """
    return projection matrix P: fixed but random Gaussian matrix
    with columns normalized to 1,
    """
    P = jr.normal(key, shape=(d, D))
    P = P / jnp.linalg.norm(P, axis=-1, keepdims=True)
    return P


def bt_loss_fn(params, ts: TrainState, batch: QueryData, l2_reg: float = 0.0):
    contexts_B2TD, labels_B2 = batch.contexts, batch.labels
    logits_B2 = ts.apply_fn({"params": params}, contexts_B2TD)
    loss = optax.softmax_cross_entropy(logits_B2, labels_B2).mean()
    params_flat, _ = ravel_pytree(params)
    l2_loss = l2_reg * (params_flat**2).sum()
    return loss + l2_loss, logits_B2


def compute_info_gain(logprobs_M2, M: int) -> Float[Array, " "]:
    """
    Compute InfoGain for a single query, given binary logprob from each model of ensemble
    work in logspace for numerical stability
    """
    log_sum_p = jax.nn.logsumexp(logprobs_M2, axis=0, keepdims=True)
    mi_M2 = jnp.exp(logprobs_M2) * (jnp.log(M) + logprobs_M2 - log_sum_p) / jnp.log(2)
    value = jnp.sum(mi_M2) / jnp.log(M)
    return value


# def compute_info_gain_old(logprobs_M2, M: int):
#     """old version without logspace"""
#     probs_M2 = jnp.exp(logprobs_M2)
#     probs_M2 = jnp.nan_to_num(probs_M2, posinf=1.0, neginf=1e-8)
#     mi_M2 = probs_M2 * jnp.log2(M * probs_M2 / jnp.sum(probs_M2, axis=0))
#     value = jnp.sum(mi_M2) / M
#     return value


def compute_disagreement(logprobs_M2) -> Float[Array, " "]:
    """
    Compute disagreement for a single query, given binary logprob from each model of ensemble
    """
    probs_M2 = jnp.exp(logprobs_M2)
    pred_M = jnp.argmax(probs_M2, axis=1)
    value = jnp.var(pred_M, axis=0)
    return value


def run_gradient_descent(
    key,
    ts: TrainState,
    dataset: QueryData,
    loss_fn: Callable,
    has_aux: bool,
    niters: int,
    batch_size: int = -1,
    l2_reg: float = 0.0,
):
    """
    Run GD training for exactly niters steps.
    If batch_size == -1, run full-batch GD. Otherwise, run mini-batch SGD.
    """

    contexts, labels = dataset.contexts, dataset.labels
    N = contexts.shape[0]

    @jit
    def train_step(ts: TrainState, idxs_B: Int[Array, "B"]) -> Tuple[TrainState, Dict]:
        # retrieve batch. If full batch (bs==-1), use all data
        contexts_B2TD = contexts if batch_size == -1 else retrieve(contexts, idxs_B)
        labels_B2 = labels if batch_size == -1 else retrieve(labels, idxs_B)  # one-hot
        batch = QueryData(contexts_B2TD, labels_B2)

        # loss, grad, update
        grad_fn = value_and_grad(loss_fn, has_aux=has_aux)
        val, grads = grad_fn(ts.params, ts, batch, l2_reg)
        loss = val[0] if has_aux else val
        ts = ts.apply_gradients(grads=grads)
        flat_params, _ = ravel_pytree(ts.params)
        return ts, {"loss": loss, "params": flat_params}

    # Create batch manager and get all batches upfront
    key, key_data = jr.split(key)
    batch_manager = BatchIndexManager(key_data, data_size=N, batch_size=batch_size)
    batch_idxs = batch_manager.get_n_batches(n=niters)  # (niters, batch_size)

    # Run `niters` steps
    ts, metrics = scan(train_step, init=ts, xs=batch_idxs)
    return ts, metrics

