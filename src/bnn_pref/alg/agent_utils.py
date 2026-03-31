from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Int, Scalar
from tqdm import tqdm

from bnn_pref.data.data_env import PreferenceEnv, retrieve
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


class BatchNormTrainState(TrainState):
    batch_stats: jax.Array


def bt_loss_fn(params, logits_B2, labels_B2, l2_reg: float = 0.0):
    loss = optax.softmax_cross_entropy(logits_B2, labels_B2).mean()
    params_flat, _ = ravel_pytree(params)
    l2_loss = l2_reg * (params_flat**2).sum()
    return loss + l2_loss, logits_B2


def compute_acq_value(logprobs_M2: Float[Array, "M 2"], acq_fn: str) -> Scalar:
    """
    Takes logprobs_M2 (M, 2) for a single query, containing binary logprob from each model sample,
    and computes the value of the chosen acquisition function.

    """
    assert acq_fn in ["infogain", "disagreement", "entropy"], "Invalid acq_fn"
    if acq_fn == "infogain":
        return compute_infogain_acq(logprobs_M2)
    elif acq_fn == "disagreement":
        return compute_disagreement_acq(logprobs_M2)
    elif acq_fn == "entropy":
        return compute_entropy_acq(logprobs_M2)
    else:
        raise ValueError(f"Invalid acq_fn: {acq_fn}")


def compute_infogain_acq(logprobs_M2) -> Scalar:
    """
    Compute InfoGain for a single query, given binary logprob from each model of ensemble
    work in logspace for numerical stability
    """
    M = logprobs_M2.shape[0]
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


def compute_disagreement_acq(logprobs_M2) -> Scalar:
    """
    Compute disagreement for a single query, given binary logprob from each model of ensemble
    """
    probs_M2 = jnp.exp(logprobs_M2)
    pred_M = jnp.argmax(probs_M2, axis=1)
    value = jnp.var(pred_M, axis=0)
    return value


def compute_entropy_acq(logprobs_M2: Float[Array, "M 2"]) -> Scalar:
    """
    Shannon entropy (nats) of the mean predictive distribution over M models.
    logprobs_M2: (M, 2) per-model log-softmax for the binary preference query.
    High when the averaged belief over trajectories A vs B is uncertain.
    """
    # probs_M2: (M, 2) — per-model class probs; p_bar: (2,) — mean predictive
    probs_M2 = jnp.exp(logprobs_M2)
    p_bar = jnp.mean(probs_M2, axis=0)  # (2,)
    p_bar = jnp.clip(p_bar, 1e-12, 1.0)
    p_bar = p_bar / jnp.sum(p_bar)
    value = -jnp.sum(p_bar * jnp.log(p_bar))
    return value


def run_sgd(
    key,
    ts: TrainState,
    dataset: QueryData,
    *,
    loss_fn: Callable,
    niters: int,
    batch_size: int = -1,
    l2_reg: float = 0.0,
    get_param_trace: bool = False,  # only used for EKF's SVD subspace construction
    # * Bayesian methods kwargs
    n_models: int = 1,
    split_datastream: bool = False,
    use_vmap: bool = True,
    use_dropout: bool = False,
    use_batch_norm: bool = False,
    verbose: bool = False,
) -> Tuple[TrainState, Dict]:
    """
    Run SGD training for exactly niters steps, using for loop
    1. define the train_step function for either dropout or ensemble
    2. define iterator over batches
    3. if the trainstate is an ensemble, allow option for vmap or for loop over ensemble models

    supports:
    - single or batched trainstate for ensembles
    - same datastream for all models or different datastreams for each model

    """
    # todo: currently, only none-dropout models support batch norm
    if not use_dropout:

        def train_step(ts: TrainState, batch: QueryData) -> Tuple[TrainState, Dict]:
            """unbatched ts and a single batch of data"""
            contexts_B2TD, labels_B2 = batch.contexts, batch.labels

            def parameterized_loss(params):
                if use_batch_norm:
                    logits_B2, updates = ts.apply_fn(
                        {"params": params, "batch_stats": ts.batch_stats},
                        contexts_B2TD,
                        train=True,
                        mutable=["batch_stats"],
                    )
                else:
                    logits_B2 = ts.apply_fn({"params": params}, contexts_B2TD)
                    updates = None
                loss, _ = loss_fn(params, logits_B2, labels_B2, l2_reg)
                return loss, updates

            grad_fn = jax.value_and_grad(parameterized_loss, has_aux=True)
            (loss, updates), grads = grad_fn(ts.params)
            ts = ts.apply_gradients(grads=grads)
            if use_batch_norm:
                ts = ts.replace(batch_stats=updates["batch_stats"])
            flat_params = ravel_pytree(ts.params)[0] if get_param_trace else None
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
                    train=True,
                    rngs={"dropout": key_dropout},
                )

                loss, _ = loss_fn(params, logits_B2, labels_B2, l2_reg)
                return loss, None

            grad_fn = jax.value_and_grad(parameterized_loss, has_aux=True)
            (loss, _), grads = grad_fn(ts.params)
            ts = ts.apply_gradients(grads=grads)
            ts = ts.replace(dropout_key=key)
            flat_params = ravel_pytree(ts.params)[0] if get_param_trace else None
            return ts, {"loss": loss, "params": flat_params}

    # * data batch iterator for different model training cases:
    # (niters, bs) -> single model or multiple models with shared datastream
    # (niters, n_models, bs) -> multiple models with split datastream
    key, key_data = jr.split(key)
    batch_iterator = get_batch_iter(
        key_data,
        dataset=dataset,
        niters=niters,
        bs=batch_size,
        n_models=n_models,
        split_datastream=split_datastream,
    )
    # * training with vmap over ts
    if use_vmap or n_models == 1:
        if n_models > 1 and split_datastream:
            # (niters, n_models, bs): vmap over both ts, batch
            train_step = jax.vmap(train_step, in_axes=(0, 0))
        elif n_models > 1 and not split_datastream:
            # (niters, bs): vmap over ts
            train_step = jax.vmap(train_step, in_axes=(0, None))
        else:
            # (niters, bs): no vmap needed
            train_step = train_step

        # * start training
        metrics = []
        train_step = jax.jit(train_step)
        # for _ in range(niters):
        # pbar = tqdm(range(niters), desc="SGD steps", disable=not verbose, miniters=100)
        pbar = tqdm(range(niters), desc="SGD steps", disable=True, miniters=100)
        for _ in pbar:
            batch = next(batch_iterator)
            ts, metric = train_step(ts, batch)
            metrics.append(metric)

            postfix = {"train_loss": metric["loss"]}
            pbar.set_postfix(postfix)
        metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *metrics)

    # * training with for loop over ts
    elif n_models > 1 and not use_vmap:
        train_step = jax.jit(train_step)
        batches = []
        # get all batches upfront with optional split datastream
        # (niters, bs) or (niters, n_models, bs)
        for _ in range(niters):
            batches.append(next(batch_iterator))

        # train
        models, metrics = [], []
        for m in range(n_models):
            ts_single = get_nth_pytree(ts, m)
            for i in range(niters):
                batch = batches[i]
                batch = get_nth_pytree(batch, m) if split_datastream else batch
                ts_single, metric = train_step(ts_single, batch)
            models.append(ts_single)
            # metrics.append(metric)
        ts = jax.tree.map(lambda *xs: jnp.stack(xs), *models)
        # if get_param_trace:
        #     # metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *metrics)
        #     raise NotImplementedError("Not implemented")
    else:
        raise ValueError("Invalid use_vmap or n_models")

    return ts, metrics


def get_nth_pytree(ts, n: int):
    return jax.tree.map(lambda x: x[n], ts)


def get_batch_iter(
    key,
    dataset: QueryData,
    niters: int,
    bs: int,
    n_models: int,
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
        # (niters, n_models, bs)
        # retriever: (M, B, 2, T, D), (M, B, 2)
        batch_fn = jax.vmap(create_batches, out_axes=1)
        batch_idxs = batch_fn(jr.split(key_data, n_models))
        retriever_fn = jax.vmap(retrieve, in_axes=(None, 0))
    else:
        # (niters, bs)
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


def get_sgd_nsteps(niters: int, n: int) -> int:
    """
    For getting the number of SGD steps to run, specifiable as either iters or epochs
    but both are converted to steps (nonnegative).

    Args:
        niters (int): number of SGD iters or epochs
        n (int): number of data points

    For getting the number of SGD steps to run.
        if niters > 0: run for `niters` steps
        if niters < 0: run for `abs(niters)` epochs
    """
    if niters > 0:  # steps
        return niters
    elif niters < 0:  # epochs
        n_epochs = jnp.abs(niters)
        return int(n * n_epochs)
    else:
        return 0
