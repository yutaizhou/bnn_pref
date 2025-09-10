import warnings
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from bnn_pref.alg.agent_dropout import DropoutAgent, DropoutBeliefState
from bnn_pref.alg.agent_ekf import EKFAgent, EKFBeliefState
from bnn_pref.alg.agent_ensemble import EnsembleAgent, EnsembleBeliefState
from bnn_pref.alg.agent_utils import Agent
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.network import RewardNet, RewardNet2

warnings.filterwarnings("ignore")

AgentState = Union[EKFBeliefState, EnsembleBeliefState, DropoutBeliefState]

"""
run_alg -> alg_pipeline -> run_updates
"""


def run_alg(key, alg: str, cfg, data_dict, env):
    # print(f"Running {alg}...")
    assert alg in ["ekf", "sgd", "do"]
    alg_cls_dict = {
        "ekf": EKFAgent,
        "sgd": EnsembleAgent,
        "do": DropoutAgent,
    }
    alg_cfg = cfg[alg]
    alg_cls = alg_cls_dict[alg]
    data_cfg = cfg["data"]
    test_trajs_obs = data_dict["test_trajs"]["observations"]
    test_prefs = data_dict["test_prefs"]

    # * build + train
    key, key_train, key_eval = jr.split(key, 3)
    bel_trace, bandit = alg_pipeline(key_train, alg_cls, env, alg_cfg, data_cfg)

    # * evaluation
    def eval_bel(_, bel: AgentState):
        # compute posterior predictive
        key_postpred = jr.fold_in(key_eval, bel.t)
        prob_Q2 = bandit.compute_postpred(
            key_postpred, bel, test_trajs_obs, test_prefs.queries_Q2
        )

        # compute metrics
        pred_Q = prob_Q2.argmax(axis=1)
        test_acc = jnp.mean(pred_Q == test_prefs.responses_Q1.squeeze())
        prob_Q1 = jnp.take_along_axis(prob_Q2, test_prefs.responses_Q1, axis=1)
        prob_Q1 = jnp.clip(prob_Q1, a_min=1e-7, a_max=1 - 1e-7)  # numerical stability
        test_logpdf = jnp.log(prob_Q1).mean()

        # all arrays of (1 + nq_updates, )
        result = {
            "test_logpdf": test_logpdf,
            "test_acc": test_acc,
        }
        return (), result

    *_, al_results = jax.lax.scan(eval_bel, init=(), xs=bel_trace)

    model_trace = bel_trace if alg == "ekf" else bel_trace.ts
    model = jax.tree.map(lambda x: x[-1], model_trace)  # get only the final model
    if alg == "do":
        model = model.replace(dropout_key=jr.key_data(model.dropout_key))

    results = {
        **al_results,  # (n_seeds, 1 + nq_update)
        "param_count": bandit.param_count,
        # "ensemble_param_count": bandit.ensemble_param_count,
        "model": model,
    }
    if alg == "ekf":
        results["subspace_param_count"] = bandit.subspace_param_count
    else:
        results["ensemble_param_count"] = bandit.ensemble_param_count

    return results


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
