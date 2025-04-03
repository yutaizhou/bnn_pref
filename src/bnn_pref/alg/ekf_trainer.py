import warnings
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax.lax import scan
from jax.random import split
from jaxtyping import Key

from bnn_pref.alg.ekf_subspace import SubspaceNeuralEKF
from bnn_pref.data.ekf_env import EKFEnvironment
from bnn_pref.utils.network import RewardNet
from bnn_pref.utils.type import CAR, CARL, BeliefState

warnings.filterwarnings("ignore")


def bandit_pipeline(
    key,
    env: EKFEnvironment,
    bandit_kw: Dict,
):
    """
    Train a bandit on an environment.
    1. Initialize the bandit
    2. Run warmups in the env to collect data (round-robin actions)
    3. Init belief on warmup data: sgd + PCA/random projection + EKF init
    4. Run trials: interact with the env and run EKF filtering
    5. Return rewards from: warmup, trials, and (optionally) opt_rewards
    """
    n_samples, *_, n_feats = env.contexts.shape
    nq_init = bandit_kw["nq_init"]
    bs = bandit_kw["cls"]["batch_size"]
    if nq_init < bs:
        bandit_kw["cls"]["batch_size"] = 1
        print(f"WARNING: {nq_init=} < {bs=}, setting ekf.cls.batch_size = 1")
    nq_updates = bandit_kw["nq_updates"]
    # end_idx = n_samples if nq_updates == -1 else nq_init + nq_updates
    nsteps = len(env) - nq_init if nq_updates == -1 else nq_updates

    model = RewardNet(bandit_kw["hidden_sizes"])
    opt = optax.adam(bandit_kw["learning_rate"])
    bandit = SubspaceNeuralEKF(n_feats, model, opt, **bandit_kw["cls"])

    key, key_warmup, key_belief_init = split(key, 3)
    warmup_data = env.warmup(key_warmup, nq_init)
    bel_init = bandit.init_bel(key_belief_init, warmup_data)
    bel_trace, batches = run_bandit(
        key, bandit, bel_init, env, warmup_data, nsteps, active=bandit_kw["active"]
    )

    # prepend initial belief (zero vector in subspace) to bel_trace
    bel_trace = jax.tree.map(
        lambda a, b: jnp.concat([a, b]),
        jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), bel_init),
        bel_trace,
    )

    rewards_info = (warmup_data.rewards, batches.rewards)  # all 1D
    return rewards_info, bel_trace, bandit


def run_bandit(
    key,
    bandit: SubspaceNeuralEKF,
    bel: BeliefState,
    env: EKFEnvironment,
    warmup_data: CARL,
    nsteps: int,  # either nq_train or nq_init + nq_updates
    active: bool = False,
) -> Tuple[BeliefState, CAR]:
    """
    Run the bandit algorithm on the environment.
    Given `nq_train` queries, warmup sgd took `nq_init` queries
    Run EKF filtering on the remaining `nq_updates = nq_train - nq_init` queries
    """
    # index into the dataset, get what's remaining after warmup
    nq_init = len(warmup_data.rewards)
    # nq_updates = end_idx - nq_init  # not cfg.ekf.nq_updates! based on end_idx
    keys = split(key, nsteps)
    pool_size = len(env) - nq_init  # active learning
    query_idxs = jnp.arange(nq_init, len(env))
    active_contexts, _ = env.get_n(query_idxs)

    def filter_onestep(
        curr: Tuple[BeliefState, int],
        key: Key,
    ) -> Tuple[BeliefState, CAR]:
        bel, t = curr
        t_offset = t + nq_init  # offset by nq_init

        context = env.get_context(t_offset)
        action = bandit.choose_action(key, bel, context)
        reward = env.get_reward(t_offset, action)
        label = env.get_label(t_offset)
        batch = CAR(context, action, reward)

        bel = bandit.update_bel(bel, batch)

        key, subkey = split(key)
        if not active:  # get a random query
            # t_next = t + 1
            t_next = jr.randint(subkey, (1,), 0, pool_size)[0]
        else:  # get a query that maximizes acquisition fn
            t_next = bandit.acquire_next_query(subkey, bel, active_contexts)

        return (bel, t_next), (bel, batch, t)

    (final_bel, _), (bel_trace, batches, ts) = scan(
        filter_onestep, init=(bel, 0), xs=keys
    )

    warmup_contexts, warmup_actions, warmup_rewards, _ = warmup_data
    contexts = jnp.vstack([warmup_contexts, batches.contexts])
    actions = jnp.append(warmup_actions, batches.actions)
    rewards = jnp.append(warmup_rewards, batches.rewards)

    print(ts)
    return bel_trace, CAR(contexts, actions, rewards)


def summarize_results(warmup_rewards, rewards):
    """
    Print a summary of running a Bandit algorithm for a number of runs
    """
    warmup_reward = warmup_rewards.sum()
    rewards = rewards.sum(axis=-1)
    r_mean = rewards.mean()
    r_std = rewards.std()
    r_total = r_mean + warmup_reward

    print(f"Expected Reward : {r_total:0.2f} ± {r_std:0.2f}")
    return r_total, r_std
