import warnings
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from jax.lax import scan
from jax.random import split
from jaxtyping import Key

from bnn_pref.data.ekf_env import EKFEnvironment
from bnn_pref.utils.network import RewardNet
from bnn_pref.utils.type import CAR, CARL, BeliefState

warnings.filterwarnings("ignore")


def bandit_pipeline(
    key,
    bandit_cls,
    env: EKFEnvironment,
    bandit_kw: Dict,
):
    """
    Train a bandit on an environment.
    1. Initialize the bandit
    2. Run warmups in the env to collect data (round-robin actions)
    3. Init belief on warmup data: sgd + PCA/random projection + EKF init
    4. Run trials: interact with the env and run EKF filtering
    5. Return rewards from: warmup, trials, and opt_rewards
    """
    n_samples, *_, n_feats = env.contexts.shape
    warmup_obs = bandit_kw["warm_obs"]
    nsteps = n_samples if bandit_kw["n_steps"] == -1 else bandit_kw["n_steps"]
    model = RewardNet(bandit_kw["hidden_sizes"])
    opt = optax.adam(bandit_kw["learning_rate"])
    bandit = bandit_cls(n_feats, model, opt, **bandit_kw["cls"])

    key, key_warmup, key_belief_init = split(key, 3)
    warmup_data = env.warmup(key_warmup, warmup_obs)
    bel_init = bandit.init_bel(key_belief_init, warmup_data)
    bel_trace, batches = run_bandit(key, bandit, bel_init, env, warmup_data, nsteps)

    # prepend initial belief (zero vector) to bel_trace
    bel_trace = jax.tree.map(
        lambda a, b: jnp.concat([a, b]),
        jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), bel_init),
        bel_trace,
    )

    rewards_info = (warmup_data.rewards, batches.rewards, env.opt_rewards)  # all 1D
    return rewards_info, bel_trace, bandit


def run_bandit(
    key,
    bandit,
    bel: BeliefState,
    env: EKFEnvironment,
    warmup_data: CARL,
    nsteps: int,
) -> Tuple[BeliefState, CAR]:
    warmup_contexts, warmup_actions, warmup_rewards, _ = warmup_data
    nwarmup = len(warmup_rewards)

    steps = jnp.arange(nwarmup, nsteps)
    keys = split(key, nsteps - nwarmup)

    def filter_onestep(
        bel: BeliefState,
        curr: Tuple[int, Key],
    ) -> Tuple[BeliefState, CAR]:
        t, mykey = curr

        context = env.get_context(t)
        action = bandit.choose_action(mykey, bel, context)
        reward = env.get_reward(t, action)
        batch = CAR(context, action, reward)

        bel = bandit.update_bel(bel, batch)

        return bel, (bel, batch)

    final_bel, (bel_trace, batches) = scan(filter_onestep, init=bel, xs=(steps, keys))

    contexts = jnp.vstack([warmup_contexts, batches.contexts])
    actions = jnp.append(warmup_actions, batches.actions)
    rewards = jnp.append(warmup_rewards, batches.rewards)

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


def print_ekf_cfg(seed, cfg, length=None, n_feats=None):
    data_kw = cfg["data"]
    task_kw = cfg["task"]  # only synthetic task has T and D in task_kw
    ekf_kw = cfg["ekf"]
    ekf_cls_cfg = ekf_kw["cls"]

    n_demos, n_queries = data_kw["n_demos"], data_kw["n_queries"]
    n_feats = n_feats if n_feats is not None else task_kw["n_feats"]
    length = length if length is not None else task_kw["length"]

    warm_obs = ekf_kw["warm_obs"]
    n_steps = ekf_kw["n_steps"]
    n_updates = (n_queries - warm_obs) if n_steps == -1 else n_steps

    n_iterates = ekf_cls_cfg["n_iterates"]
    batch_size = ekf_cls_cfg["batch_size"]
    warm_burns = ekf_cls_cfg["warm_burns"]
    thinning = ekf_cls_cfg["thinning"]
    sub_dim = ekf_cls_cfg["sub_dim"]
    rnd_proj = ekf_cls_cfg["rnd_proj"]
    n_eff_iterates = (n_iterates - warm_burns) // thinning

    if task_kw["ds_type"] == "synthetic":
        # todo fix this fhat thing
        task_str = f"{task_kw['ds_type']}: f={task_kw['f']}, fhat={task_kw['fhat']} (fhat ignored for ekf runs)"
    else:
        task_str = f"{task_kw['ds_type']}: {task_kw['name']}"

    print(
        f"Seed: {seed}\n"
        f"Data:\n"
        f"  {task_str}\n"
        f"  N={n_demos}, Q={n_queries}, T={length}, D={n_feats} (Q is Train)\n"
        f"  Samples for init/update = {warm_obs}/{n_updates}\n"
        f"EKF:\n"
        f"  init: bs={batch_size}, n_iterates={n_iterates}[{warm_burns}::{thinning}] ({n_eff_iterates} eff), {sub_dim=}, {rnd_proj=}\n"
    )
