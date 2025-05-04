import warnings
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax.training.train_state import TrainState
from jax.lax import scan
from jax.random import split
from jaxtyping import Key

from bnn_pref.alg.agent_utils import Agent
from bnn_pref.alg.ekf_subspace import EKFBeliefState, SubspaceEKF
from bnn_pref.alg.ensemble import DeepEnsemble, EnsembleBeliefState
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.network import RewardNet
from bnn_pref.utils.type import QueryData

warnings.filterwarnings("ignore")

AgentState = Union[EKFBeliefState, EnsembleBeliefState]


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
    model = RewardNet(alg_cfg["hidden_sizes"], alg_cfg["n_splits"])
    opt = optax.adam(alg_cfg["learning_rate"])
    cls_kwargs = alg_cls.get_hydra_config(alg_cfg)
    bandit = alg_cls(model, opt, traj_shape=traj_shape, **cls_kwargs)

    key, key_warm, key_bel_init, key_run = split(key, 4)
    warmup_data = env.warmup(key_warm, nq_init)
    bel_init = bandit.init_bel(key_bel_init, warmup_data)
    bel_trace = run_update_loop(
        key_run, bandit, bel_init, env, nq_init, nsteps, active=alg_cfg["active"]
    )

    # * prepend initial belief (zero vector in subspace) to bel_trace
    bel_trace = jax.tree.map(
        lambda a, b: jnp.concat([a, b]),
        jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), bel_init),
        bel_trace,
    )

    return bel_trace, bandit


def run_update_loop(
    key,
    bandit: Agent,
    bel: AgentState,
    env: PreferenceEnv,
    nq_init: int,
    nsteps: int,  # either len(env) - nq_init or nq_update
    active: bool = False,
) -> AgentState:
    """
    Run the bandit algorithm on the environment.
    Given `nq_train` queries, warmup sgd took `nq_init` queries
    Run EKF filtering on the remaining `nsteps` queries
    """
    # index into the dataset, get what's remaining after warmup
    pool_size = len(env) - nq_init  # active learning
    pool_idxes = jnp.arange(nq_init, len(env))

    def update_step(
        curr: Tuple[AgentState, int],
        key: Key,
    ) -> Tuple[AgentState, int]:
        bel, t = curr
        t_offset = t + nq_init  # offset by nq_init to index into query pool

        context = env.get_context(t_offset)  # (2, T, D)
        label = env.get_label(t_offset)  # (2,) one-hot preference
        batch = QueryData(context, label).add_leading_batch_dim()

        key, key_update = split(key)
        bel = bandit.update_bel(key_update, bel, batch)
        q = env.get_pref_indices(t_offset)

        key, key_query = split(key)
        if not active:
            t_next = jr.randint(key_query, (), 0, pool_size)
        else:
            t_next = bandit.acquire_next_query(key_query, bel, env, pool_idxes)

        return (bel, t_next), (bel, t, q)

    keys = split(key, nsteps)
    *_, (bel_trace, t_trace, q_trace) = scan(update_step, init=(bel, 0), xs=keys)

    # print(q_trace)
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
        prob_Q2 = bandit.compute_predictive(
            key, bel, test_trajs_obs, test_prefs.queries_Q2
        )
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
    key, key_pipe = jr.split(key, 2)
    bel_trace, bandit = alg_pipeline(key_pipe, DeepEnsemble, env, alg_cfg, data_cfg)
    ts_trace = bel_trace.ts

    # * compute metrics
    def eval_bel(_, ts: TrainState):
        prob_Q2 = bandit.compute_predictive(ts, test_trajs_obs, test_prefs.queries_Q2)
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

    *_, al_results = jax.lax.scan(eval_bel, init=(), xs=ts_trace)

    model = jax.tree.map(lambda x: x[-1], ts_trace)  # get only the final model
    results = {
        **al_results,  # (n_seeds, 1 + nq_update)
        "param_count": bandit.param_count,
        "ensemble_param_count": bandit.ensemble_param_count,
        "model": model,
    }
    return results
