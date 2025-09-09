import warnings
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Key

from bnn_pref.alg.agent_dropout import DropoutAgent, DropoutBeliefState
from bnn_pref.alg.agent_ekf import EKFBeliefState, SubspaceEKF
from bnn_pref.alg.agent_ensemble import DeepEnsemble, EnsembleBeliefState
from bnn_pref.alg.agent_utils import Agent
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.network import RewardNet, RewardNet2
from bnn_pref.utils.type import QueryData

warnings.filterwarnings("ignore")

AgentState = Union[EKFBeliefState, EnsembleBeliefState, DropoutBeliefState]

"""
run_{ekf,ensemble}
    alg_pipeline -> run_updates
    evaluation
"""


def alg_pipeline(
    key,
    alg_cls: Agent,
    env: PreferenceEnv,
    alg_cfg: Dict,
    data_cfg: Dict,
) -> Tuple[AgentState, Agent]:
    nq_init, nsteps = data_cfg["nq_init"], data_cfg["nsteps"]
    bs = alg_cfg["bs"]
    if nq_init < bs:
        alg_cfg["bs"] = 1
        # print(f"WARNING: {nq_init=} < {bs=}, setting alg_cfg.bs = 1")

    traj_shape = env.get_traj_shape()
    # model = RewardNet(alg_cfg["hidden_sizes"], alg_cfg["n_splits"])
    model = RewardNet2(
        alg_cfg["hidden_sizes"],
        alg_cfg["n_splits"],
        alg_cfg["dropout_prob"],
    )
    opt = optax.adam(alg_cfg["learning_rate"])
    cls_kwargs = alg_cls.get_hydra_config(alg_cfg)
    bandit = alg_cls(model, opt, traj_shape=traj_shape, **cls_kwargs)

    key, key_warm, key_bel_init, key_run = jr.split(key, 4)
    warmup_data = env.warmup(key_warm, nq_init)
    bel_init = bandit.init_bel(key_bel_init, warmup_data)
    bel_trace = run_updates(
        key_run,
        bandit,
        bel_init,
        env,
        nq_init,
        nsteps,
        active=alg_cfg["active"],
    )

    # * prepend initial subspace belief (zero vector in subspace) to bel_trace
    bel_trace = jax.tree.map(
        lambda a, b: jnp.concat([a, b]),
        jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), bel_init),
        bel_trace,
    )

    return bel_trace, bandit


def run_updates_scan(
    key,
    bandit: Agent,
    bel: AgentState,
    env: PreferenceEnv,
    nq_init: int,
    nsteps: int,  # either (nq_train - nq_init) or nq_update
    active: bool = False,
) -> AgentState:
    """
    Run the bandit algorithm on the environment.
    Given `nq_train` queries, warmup sgd took `nq_init` queries
    Run EKF filtering on the remaining `nsteps` queries

    Note: the `t` here is the index into the query pool, excluding the `nq_init` warmup queries
    """
    # index into the dataset, get what's remaining after warmup
    pool_idxes = jnp.arange(nq_init, len(env))

    def update_step(bel: AgentState, key: Key) -> AgentState:
        key, key_query = jr.split(key)
        if not active:
            t = jr.choice(key_query, pool_idxes)
        else:
            t = bandit.compute_next_query(key_query, bel, env, pool_idxes)

        context = env.get_context(t)  # (2, T, D)
        label = env.get_label(t)  # (2,) one-hot preference
        batch = QueryData(context, label).add_leading_batch_dim()

        key, key_update = jr.split(key)
        bel = bandit.update_bel(key_update, bel, batch)

        return bel, (bel, t)

    keys = jr.split(key, nsteps)
    *_, (bel_trace, t_trace) = jax.lax.scan(
        update_step,
        init=bel,
        xs=keys,
    )

    return bel_trace


def run_updates(
    key,
    bandit: Agent,
    bel: AgentState,
    env: PreferenceEnv,
    nq_init: int,
    nsteps: int,  # either (nq_train - nq_init) or nq_update
    active: bool = False,
) -> AgentState:
    """
    Run the bandit algorithm on the environment.
    Given `nq_train` queries, warmup sgd took `nq_init` queries
    Run EKF filtering on the remaining `nsteps` queries
    """
    # index into the dataset, get what's remaining after warmup
    # pool_size = len(env) - nq_init  # pool for active learning, after warmup
    pool_idxs = jnp.arange(nq_init, len(env))

    bels = []
    for _ in range(nsteps):
        # retrieve query
        key, key_query = jr.split(key)
        if not active:
            t = jr.choice(key_query, jnp.arange(len(pool_idxs)))
        else:
            t = bandit.compute_next_query(key_query, bel, env, pool_idxs)
        batch = env.get_batched_query(t + nq_init)  # (1, 2, T, D)

        # update belief
        key, key_update = jr.split(key)
        bel = bandit.update_bel(key_update, bel, batch)
        bels.append(bel)

    bel_trace = jax.tree.map(lambda *xs: jnp.stack(xs), *bels)
    return bel_trace


def run_ekf(key, cfg, data_dict, env):
    ekf_cfg = cfg["ekf"]
    data_cfg = cfg["data"]
    test_trajs_obs = data_dict["test_trajs"]["observations"]
    test_prefs = data_dict["test_prefs"]

    # * build + run bandit alg
    key, key_pipe, key_bma = jr.split(key, 3)
    bel_trace, bandit = alg_pipeline(key_pipe, SubspaceEKF, env, ekf_cfg, data_cfg)

    # * compute metrics
    def eval_bel(_, bel: EKFBeliefState):
        # * sample model parameters
        key = jr.fold_in(key_bma, bel.t)
        prob_Q2 = bandit.compute_postpred(
            key, bel, test_trajs_obs, test_prefs.queries_Q2
        )

        # same as other algs
        pred_Q = prob_Q2.argmax(axis=1)
        test_acc = jnp.mean(pred_Q == test_prefs.responses_Q1.squeeze())
        prob_Q1 = jnp.take_along_axis(prob_Q2, test_prefs.responses_Q1, axis=1)
        test_logpdf = jnp.log(prob_Q1).mean()

        # all arrays of (1 + nq_updates, )
        result = {
            "test_logpdf": test_logpdf,
            "test_acc": test_acc,
        }
        return (), result

    *_, al_results = jax.lax.scan(eval_bel, init=(), xs=bel_trace)

    model = jax.tree.map(lambda x: x[-1], bel_trace)  # get only the final model
    results = {
        **al_results,  # (n_seeds, nq_update)
        "param_count": bandit.param_count,
        "subspace_param_count": bandit.subspace_param_count,
        "model": model,
    }

    return results


def run_ensemble(key, cfg, data_dict, env):
    data_cfg = cfg["data"]
    alg_cfg = cfg["sgd"]
    test_trajs_obs = data_dict["test_trajs"]["observations"]
    test_prefs = data_dict["test_prefs"]

    # * build + run ensemble alg
    key, key_pipe, key_eval = jr.split(key, 3)
    bel_trace, bandit = alg_pipeline(key_pipe, DeepEnsemble, env, alg_cfg, data_cfg)

    # * compute metrics
    def eval_bel(_, bel: EnsembleBeliefState):
        key = jr.fold_in(key_eval, bel.t)
        prob_Q2 = bandit.compute_postpred(
            key, bel, test_trajs_obs, test_prefs.queries_Q2
        )

        # same as other algs
        pred_Q = prob_Q2.argmax(axis=1)
        test_acc = jnp.mean(pred_Q == test_prefs.responses_Q1.squeeze())
        prob_Q1 = jnp.take_along_axis(prob_Q2, test_prefs.responses_Q1, axis=1)
        test_logpdf = jnp.log(prob_Q1).mean()

        # all arrays of (1 + nq_updates, )
        result = {
            "test_logpdf": test_logpdf,
            "test_acc": test_acc,
        }
        return (), result

    *_, al_results = jax.lax.scan(eval_bel, init=(), xs=bel_trace)

    model = jax.tree.map(lambda x: x[-1], bel_trace.ts)  # get only the final model
    results = {
        **al_results,  # (n_seeds, 1 + nq_update)
        "param_count": bandit.param_count,
        "ensemble_param_count": bandit.ensemble_param_count,
        "model": model,
    }
    return results


def run_dropout(key, cfg, data_dict, env):
    data_cfg = cfg["data"]
    alg_cfg = cfg["do"]
    test_trajs_obs = data_dict["test_trajs"]["observations"]
    test_prefs = data_dict["test_prefs"]

    # * build + run ensemble alg
    key, key_train, key_eval = jr.split(key, 3)
    bel_trace, bandit = alg_pipeline(key_train, DropoutAgent, env, alg_cfg, data_cfg)

    # * compute metrics
    def eval_bel(_, bel: DropoutBeliefState):
        key = jr.fold_in(key_eval, bel.t)
        prob_Q2 = bandit.compute_postpred(
            key, bel, test_trajs_obs, test_prefs.queries_Q2
        )

        # same as other algs
        pred_Q = prob_Q2.argmax(axis=1)
        test_acc = jnp.mean(pred_Q == test_prefs.responses_Q1.squeeze())
        prob_Q1 = jnp.take_along_axis(prob_Q2, test_prefs.responses_Q1, axis=1)
        test_logpdf = jnp.log(prob_Q1).mean()

        # all arrays of (1 + nq_updates, )
        result = {
            "test_logpdf": test_logpdf,
            "test_acc": test_acc,
        }
        return (), result

    *_, al_results = jax.lax.scan(eval_bel, init=(), xs=bel_trace)

    model = jax.tree.map(lambda x: x[-1], bel_trace.ts)  # get only the final model
    # remove dropout key if it exists in trainstate
    model = model.replace(dropout_key=jr.key_data(model.dropout_key))

    results = {
        **al_results,  # (n_seeds, 1 + nq_update)
        "param_count": bandit.param_count,
        "ensemble_param_count": bandit.ensemble_param_count,
        "model": model,
    }
    return results
