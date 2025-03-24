import warnings
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
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
    5. Return rewards from: warmup, trials, and opt_rewards
    """
    n_samples, *_, n_feats = env.contexts.shape
    warmup_obs = bandit_kw["warm_obs"]
    n_steps = bandit_kw["n_steps"]
    end_idx = n_samples if n_steps == -1 else n_samples - n_steps

    model = RewardNet(bandit_kw["hidden_sizes"])
    opt = optax.adam(bandit_kw["learning_rate"])
    bandit = SubspaceNeuralEKF(n_feats, model, opt, **bandit_kw["cls"])

    key, key_warmup, key_belief_init = split(key, 3)
    warmup_data = env.warmup(key_warmup, warmup_obs)
    bel_init = bandit.init_bel(key_belief_init, warmup_data)
    bel_trace, batches = run_bandit(key, bandit, bel_init, env, warmup_data, end_idx)

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
    bandit: SubspaceNeuralEKF,
    bel: BeliefState,
    env: EKFEnvironment,
    warmup_data: CARL,
    end_idx: int,
) -> Tuple[BeliefState, CAR]:
    warmup_contexts, warmup_actions, warmup_rewards, _ = warmup_data
    nwarmup = len(warmup_rewards)

    # index into the dataset, get what's remaining after warmup
    steps = jnp.arange(nwarmup, end_idx)
    keys = split(key, end_idx - nwarmup)

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
