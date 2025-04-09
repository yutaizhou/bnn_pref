import warnings
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax.lax import scan
from jax.random import split
from jaxtyping import Key

from bnn_pref.alg.ekf_subspace import EKFBeliefState, SubspaceNeuralEKF
from bnn_pref.alg.ensemble import DeepEnsemble
from bnn_pref.data.ekf_env import DataEnvironment
from bnn_pref.utils.network import RewardNet
from bnn_pref.utils.type import CAR, CARL

warnings.filterwarnings("ignore")


def ensemble_pipeline(
    key,
    env: DataEnvironment,
    sgd_cfg: Dict,
    data_cfg: Dict,
):
    nq_init, nsteps = data_cfg["nq_init"], data_cfg["nsteps"]
    bs = sgd_cfg["cls"]["batch_size"]
    if nq_init < bs:
        sgd_cfg["cls"]["batch_size"] = 1
        # print(f"WARNING: {nq_init=} < {bs=}, setting ekf.cls.batch_size = 1")

    model = RewardNet(sgd_cfg["hidden_sizes"])
    opt = optax.adam(sgd_cfg["learning_rate"])
    bandit = DeepEnsemble(model, opt, **sgd_cfg["cls"])

    key, key_warmup, key_belief_init = split(key, 3)
    warmup_data = env.warmup(key_warmup, nq_init)
    bel_init = bandit.init_bel(key_belief_init, warmup_data)
    bel_trace = run_bandit(
        key, bandit, bel_init, env, warmup_data, nsteps, active=sgd_cfg["active"]
    )

    # * prepend initial belief (zero vector in subspace) to bel_trace
    bel_trace = jax.tree.map(
        lambda a, b: jnp.concat([a, b]),
        jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), bel_init),
        bel_trace,
    )

    return bel_trace, bandit


def bandit_pipeline(
    key,
    env: DataEnvironment,
    bandit_cfg: Dict,
    data_cfg: Dict,
):
    """
    Train a bandit on an environment.
    1. Initialize the bandit
    2. Run warmups in the env to collect data (round-robin actions)
    3. Init belief on warmup data: sgd + PCA/random projection + EKF init
    4. Run trials: interact with the env and run EKF filtering

    Returns:
        bel_trace: list of BeliefState, length = nq_update + 1 (including sgd init)
        bandit: SubspaceNeuralEKF class constructed with bandit_kw
    """
    nq_init, nsteps = data_cfg["nq_init"], data_cfg["nsteps"]
    bs = bandit_cfg["cls"]["batch_size"]
    if nq_init < bs:
        bandit_cfg["cls"]["batch_size"] = 1
        # print(f"WARNING: {nq_init=} < {bs=}, setting ekf.cls.batch_size = 1")

    model = RewardNet(bandit_cfg["hidden_sizes"])
    opt = optax.adam(bandit_cfg["learning_rate"])
    bandit = SubspaceNeuralEKF(model, opt, **bandit_cfg["cls"])

    key, key_warmup, key_belief_init = split(key, 3)
    warmup_data = env.warmup(key_warmup, nq_init)
    bel_init = bandit.init_bel(key_belief_init, warmup_data)
    bel_trace = run_bandit(
        key, bandit, bel_init, env, warmup_data, nsteps, active=bandit_cfg["active"]
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
    bandit: SubspaceNeuralEKF,
    bel: EKFBeliefState,
    env: DataEnvironment,
    warmup_data: CARL,
    nsteps: int,  # either len(env) - nq_init or nq_update
    active: bool = False,
) -> Tuple[EKFBeliefState, CAR]:
    """
    Run the bandit algorithm on the environment.
    Given `nq_train` queries, warmup sgd took `nq_init` queries
    Run EKF filtering on the remaining `nsteps` queries
    """
    # index into the dataset, get what's remaining after warmup
    nq_init = len(warmup_data.rewards)
    pool_size = len(env) - nq_init  # active learning
    active_contexts, _ = env.get_n(jnp.arange(nq_init, len(env)))
    assert pool_size == len(active_contexts)

    def filter_onestep(
        curr: Tuple[EKFBeliefState, int],
        key: Key,
    ) -> Tuple[EKFBeliefState, CAR]:
        bel, t = curr
        t_offset = t + nq_init  # offset by nq_init

        context = env.get_context(t_offset)
        label = env.get_label(t_offset)  # one-hot pref, always [0,1] cuz traj 2 > 1
        batch = CARL(context, None, None, label)
        bel = bandit.update_bel(bel, batch)

        key, subkey = split(key)
        if not active:  # get a random query
            # t_next = t + 1
            t_next = jr.randint(subkey, (), 0, pool_size)
        else:  # get a query that maximizes acquisition fn
            t_next = bandit.acquire_next_query(subkey, bel, active_contexts)

        return (bel, t_next), (bel, t, batch)

    keys = split(key, nsteps)
    *_, (bel_trace, ts, _) = scan(filter_onestep, init=(bel, 0), xs=keys)

    # print(ts)
    return bel_trace
