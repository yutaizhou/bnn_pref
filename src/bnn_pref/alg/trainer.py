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
from bnn_pref.alg.ekf_subspace import EKFBeliefState
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.network import RewardNet
from bnn_pref.utils.type import CARL

warnings.filterwarnings("ignore")

AgentState = Union[TrainState, EKFBeliefState]


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

    model = RewardNet(alg_cfg["hidden_sizes"])
    opt = optax.adam(alg_cfg["learning_rate"])
    cls_kwargs = alg_cls.get_hydra_config(alg_cfg)
    bandit = alg_cls(model, opt, **cls_kwargs)

    key, key_warm, key_bel_init, key_run = split(key, 4)
    warmup_data = env.warmup(key_warm, nq_init)
    bel_init = bandit.init_bel(key_bel_init, warmup_data)
    bel_trace = run_bandit(
        key_run, bandit, bel_init, env, nq_init, nsteps, active=alg_cfg["active"]
    )

    # * prepend initial belief (zero vector in subspace) to bel_trace
    bel_trace = jax.tree.map(
        lambda a, b: jnp.concat([a, b]),
        jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), bel_init),
        bel_trace,
    )

    return bel_trace, bandit


def run_bandit(
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

    def filter_onestep(
        curr: Tuple[AgentState, int],
        key: Key,
    ) -> Tuple[AgentState, int]:
        bel, t = curr
        t_offset = t + nq_init  # offset by nq_init to index into query pool

        context = env.get_context(t_offset)
        label = env.get_label(t_offset)  # one-hot pref, always [0,1] if noiseless
        batch = CARL(context, None, None, label)
        bel = bandit.update_bel(bel, batch)
        q = env.get_pref_indices(t_offset)

        key, subkey = split(key)
        if not active:
            t_next = jr.randint(subkey, (), 0, pool_size)
        else:
            t_next = bandit.acquire_next_query(subkey, bel, env, pool_idxes)

        return (bel, t_next), (bel, t, q)

    keys = split(key, nsteps)
    *_, (bel_trace, t_trace, q_trace) = scan(filter_onestep, init=(bel, 0), xs=keys)

    # print(q_trace)
    return bel_trace
