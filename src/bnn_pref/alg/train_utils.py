import warnings
from typing import Dict, Tuple

import flax.linen as nn
import jax.numpy as jnp
import optax
from jax import vmap
from jax.lax import scan
from jax.random import split
from jaxtyping import Array, Float, Int, Key
from omegaconf import OmegaConf

from bnn_pref.alg.bandit_env import BanditEnvironment
from bnn_pref.utils.network import RewardNet
from bnn_pref.utils.type import CAR, CARL, BeliefState

warnings.filterwarnings("ignore")


def bandit_pipeline(
    key,
    bandit_cls,
    env: BanditEnvironment,
    warmup_obs: int,
    n_trials: int,
    bandit_kw: Dict,
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
    n_trials: int
        The number of trials to be used for the training phase.
    bandit_kwargs: Dict
        The keyword arguments to be used for the bandit.
    neural: bool
        Whether to use a neural network for the bandit.
    nsteps: int
        The number of steps to be used for the training phase.
    """
    nsteps, _, nfeatures = env.contexts.shape
    _, narms = env.labels_onehot.shape
    model = RewardNet(bandit_kw["hidden_sizes"])
    opt = optax.sgd(bandit_kw["learning_rate"])
    bandit = bandit_cls(nfeatures, narms, model, opt, **bandit_kw["cls"])

    # npulls * n_arms worth of data, (contexts, states, actions, rewards)
    key, key_warmup, key_belief_init = split(key, 3)
    warmup_data = env.warmup(key_warmup, warmup_obs)
    _, _, warmup_rewards, _ = warmup_data
    bel = bandit.init_bel(key_belief_init, warmup_data)

    def single_trial(key):
        final_bel, batch = run_bandit(key, bandit, bel, env, warmup_data, nsteps=nsteps)
        return final_bel, batch.rewards

    keys = split(key, n_trials)
    final_bel, rewards_trace = vmap(single_trial)(keys)

    rewards_info = (warmup_rewards, rewards_trace, env.opt_rewards)

    return rewards_info, final_bel, bandit


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

        return bel, batch

    final_bel, data = scan(step, init=bel, xs=(keys, steps))

    contexts = jnp.vstack([warmup_contexts, data.contexts])
    actions = jnp.append(warmup_actions, data.actions)
    rewards = jnp.append(warmup_rewards, data.rewards)

    return final_bel, CAR(contexts, actions, rewards)


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
