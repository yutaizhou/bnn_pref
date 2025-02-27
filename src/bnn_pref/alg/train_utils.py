import warnings
from typing import Dict, Tuple

import flax.linen as nn
import jax.numpy as jnp
from jax import vmap
from jax.lax import scan
from jax.random import split
from jaxtyping import Array, Float, Int, Key

from bnn_pref.alg.bandit_env import BanditEnvironment
from bnn_pref.utils.type import CAR, CARL, BeliefState

warnings.filterwarnings("ignore")


def bandit_pipeline(
    key,
    bandit_cls,
    env: BanditEnvironment,
    npulls: int,
    ntrials: int,
    bandit_kwargs: Dict,
    neural: bool = True,
):
    """
    Train a bandit on an environment.
    1. Initialize the bandit
    2. Run warmups in the env to collect data (round-robin actions)
    3. Init belief on warmup data: sgd + PCA/random projection + EKF init
    4. Run trials: interact with the env and run EKF filtering
    5. Return rewards from: warmup, trials, and opt_rewards

    Parameters
    ----------
    npulls: int
        The number of pulls (per arm) to be used for the warmup phase.
    ntrials: int
        The number of trials to be used for the training phase.
    bandit_kwargs: Dict
        The keyword arguments to be used for the bandit.
    neural: bool
        Whether to use a neural network for the bandit.
    nsteps: int
        The number of steps to be used for the training phase.
    """
    nsteps, nfeatures = env.contexts_ND.shape
    _, narms = env.labels_onehot_NA.shape
    bandit = bandit_cls(nfeatures, narms, **bandit_kwargs)

    # npulls * n_arms worth of data, (contexts, states, actions, rewards)
    key, key_belief_init = split(key)
    warmup_data = env.warmup(npulls)
    _, _, warmup_rewards, _ = warmup_data
    bel = bandit.init_bel(key_belief_init, warmup_data)

    def single_trial(key):
        _, _, rewards = run_bandit(
            key, bandit, bel, env, warmup_data, nsteps=nsteps, neural=neural
        )
        return rewards

    if ntrials > 1:
        keys = split(key, ntrials)
        rewards_trace = vmap(single_trial)(keys)
    else:
        rewards_trace = single_trial(key)

    return warmup_rewards, rewards_trace, env.opt_rewards_NA


def run_bandit(
    key,
    bandit,
    bel: BeliefState,
    env: BanditEnvironment,
    warmup_data: CARL,
    nsteps: int,
):
    warmup_contexts, warmup_actions, warmup_rewards, _ = warmup_data
    nwarmup = len(warmup_rewards)

    # start from t=nwarmup, end at t=nsteps
    steps = jnp.arange(nsteps - nwarmup) + nwarmup
    keys = split(key, nsteps - nwarmup)

    def step(
        bel: BeliefState,
        curr: Tuple[Key, int],
    ) -> Tuple[BeliefState, CAR]:
        mykey, t = curr

        context = env.get_context(t)
        action = bandit.choose_action(mykey, bel, context)
        reward = env.get_reward(t, action)
        batch = CAR(context, action, reward)

        bel = bandit.update_bel(bel, batch)

        return bel, (context, action, reward)

    _, data = scan(step, init=bel, xs=(keys, steps))

    contexts = jnp.vstack([warmup_contexts, data.contexts])
    actions = jnp.append(warmup_actions, data.actions)
    rewards = jnp.append(warmup_rewards, data.rewards)

    return CAR(contexts, actions, rewards)


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
