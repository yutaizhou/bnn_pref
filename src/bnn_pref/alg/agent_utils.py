from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
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


class DropoutTrainState(TrainState):
    dropout_key: jax.Array


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


# def bt_loss_fn(params, ts: TrainState, batch: QueryData, l2_reg: float = 0.0):
#     contexts_B2TD, labels_B2 = batch.contexts, batch.labels
#     logits_B2 = ts.apply_fn({"params": params}, contexts_B2TD)
#     loss = optax.softmax_cross_entropy(logits_B2, labels_B2).mean()
#     params_flat, _ = ravel_pytree(params)
#     l2_loss = l2_reg * (params_flat**2).sum()
#     return loss + l2_loss, logits_B2


def bt_loss_fn(params, logits_B2, labels_B2, l2_reg: float = 0.0):
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


def run_sgd(
    key,
    ts: TrainState,
    dataset: QueryData,
    *,
    loss_fn: Callable,
    has_aux: bool,
    niters: int,
    batch_size: int = -1,
    l2_reg: float = 0.0,
    get_param_trace: bool = False,
    n_models: int = 1,
    split_datastream: bool = False,
    use_dropout: bool = False,
) -> Tuple[TrainState, Dict]:
    """
    Run SGD training for exactly niters steps, using for loop not scan

    supports:
    - single or batched trainstate for ensembles
    - same datastream for all models or different datastreams for each model

    """
    if not use_dropout:

        def train_step(ts: TrainState, batch: QueryData) -> Tuple[TrainState, Dict]:
            """unbatched ts and a single batch of data"""
            contexts_B2TD, labels_B2 = batch.contexts, batch.labels

            def parameterized_loss(params):
                logits_B2 = ts.apply_fn({"params": params}, contexts_B2TD)
                return loss_fn(params, logits_B2, labels_B2, l2_reg)

            grad_fn = jax.value_and_grad(parameterized_loss, has_aux=has_aux)
            val, grads = grad_fn(ts.params)
            loss = val[0] if has_aux else val
            ts = ts.apply_gradients(grads=grads)
            flat_params, _ = (
                ravel_pytree(ts.params) if get_param_trace else (None, None)
            )
            return ts, {"loss": loss, "params": flat_params}

    else:

        def train_step(
            ts: DropoutTrainState, batch: QueryData
        ) -> Tuple[DropoutTrainState, Dict]:
            """unbatched ts and a single batch of data"""
            key, key_dropout = jr.split(ts.dropout_key)
            contexts_B2TD, labels_B2 = batch.contexts, batch.labels

            def parameterized_loss(params):
                logits_B2 = ts.apply_fn(
                    {"params": params},
                    contexts_B2TD,
                    deterministic=False,
                    rngs={"dropout": key_dropout},
                )
                return loss_fn(params, logits_B2, labels_B2, l2_reg)

            grad_fn = jax.value_and_grad(parameterized_loss, has_aux=has_aux)

            val, grads = grad_fn(ts.params)
            loss = val[0] if has_aux else val
            ts = ts.apply_gradients(grads=grads)
            ts = ts.replace(dropout_key=key)
            flat_params, _ = (
                ravel_pytree(ts.params) if get_param_trace else (None, None)
            )
            return ts, {"loss": loss, "params": flat_params}

    def get_batch_iter(
        key,
        dataset: QueryData,
        n_models: int,
        niters: int,
        bs: int,
        split_datastream: bool,
    ):
        """
        Returns a generator over batches from `dataset`. Supports two options with differing usecases.
        - (niters, bs)
            - one model, one datastream
            - ensemble models, shared datastream, to be broadcasted over train_step(ts, batch)
        - (niters, n_models, bs)
            - ensemble models, split datastream

        Args:
            key: PRNGKey
            dataset: QueryData = dataset to sample from
            n_models: int = number of models
            split_datastream: bool = whether to split the datastream into n_models
            niters: int = number of iterations
            bs: int = batch size

        """
        N = len(dataset)

        def create_batches(key) -> Int[Array, "niters bs"]:
            """
            Shuffle and split the `N` indices into chunks of size `bs`.
            Drop last chunk if it's not of size `bs`.
            Do so for `niters`, reshuffle if: len(curr_batch) < bs)

            for `niters` times:
                if remaining inds < bs
                    reshuffle
                take next chunk of size `bs`
            """
            batch_idxs = jnp.empty((niters, bs), dtype=jnp.int32)
            idxs = jnp.arange(N, dtype=jnp.int32)
            cumsum = 0
            reshuffle_count = 0
            for i in range(niters):
                if cumsum + bs > N:
                    key, key_perm = jr.split(key)
                    idxs = jr.permutation(key_perm, idxs)  # N
                    cumsum = 0
                    reshuffle_count += 1
                batch = jax.lax.dynamic_slice_in_dim(idxs, cumsum, bs)
                batch_idxs = batch_idxs.at[i].set(batch)
                cumsum += bs
            # print(f"reshuffled {reshuffle_count} times on {N} indices")
            return batch_idxs  # (niters, bs)

        batch_idxs = None
        retriever_fn = None
        key, key_data = jr.split(key)
        if n_models > 1 and split_datastream:
            # retriever: (M, B, 2, T, D), (M, B, 2)
            batch_fn = jax.vmap(create_batches, out_axes=1)
            batch_idxs = batch_fn(
                jr.split(key_data, n_models)
            )  # (niters, n_models, bs)
            retriever_fn = jax.vmap(retrieve, in_axes=(None, 0))
        else:
            # retriever: (B, 2, T, D), (B, 2)
            batch_idxs = create_batches(key_data)  # (niters, bs)
            retriever_fn = retrieve

        def iterate_over_batches():
            for batch_idx in batch_idxs:
                contexts = retriever_fn(dataset.contexts, batch_idx)
                labels = retriever_fn(dataset.labels, batch_idx)
                batch = QueryData(contexts=contexts, labels=labels)
                yield batch

        return iterate_over_batches()

    # * different model training cases:
    key, key_data = jr.split(key)
    batch_iterator = get_batch_iter(
        key_data,
        dataset=dataset,
        n_models=n_models,
        niters=niters,
        bs=batch_size,
        split_datastream=split_datastream,
    )
    if n_models > 1 and split_datastream:
        train_step = jax.vmap(train_step, in_axes=(0, 0))
    elif n_models > 1 and not split_datastream:
        train_step = jax.vmap(train_step, in_axes=(0, None))
    else:
        train_step = train_step

    # * start training
    metrics = []
    train_step = jax.jit(train_step)
    for _ in range(niters):
        batch = next(batch_iterator)
        ts, metric = train_step(ts, batch)
        metrics.append(metric)

    if get_param_trace:
        metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *metrics)

    return ts, metrics
