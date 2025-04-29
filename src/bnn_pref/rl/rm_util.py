import os
import warnings
from functools import partial
from typing import Callable

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import d4rl
import distrax
import gym
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import orbax
import orbax.checkpoint
from einops import rearrange
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PRNGKeyArray
from omegaconf import OmegaConf

from bnn_pref.alg.agent_utils import subspace2full_params
from bnn_pref.alg.ekf_subspace import EKFBeliefState
from bnn_pref.alg.ensemble import init_model
from bnn_pref.utils.network import RewardNet


def load_reward_model(
    key: PRNGKeyArray,
    run_dir: str,
    task_name: str = "halfcheetah-random-v2",
    alg: str = "ekf",
    is_al: bool = False,
) -> Callable[
    [Float[Array, "T D"]],
    Float[Array, "T"],
]:
    """
    run_dir: hydra run output directory, e.g. "../results/2025***"
    key: used for flax.linen.init and EKF subspace parameter sampling
    model ts shape: (n_seeds, ensemble_size, ...)
        DEPRECATED: n_seeds axis is now removed, only best seed per run is saved

    Assumes ckpts are named as:
        <run_dir>/ckpts/<task_name>_<alg>_al=<is_al>
    """

    cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")
    ckpts_dir = f"{run_dir}/ckpts"
    ckpt_fp = f"{ckpts_dir}/{task_name}_{alg}_al={is_al}"

    cktper = orbax.checkpoint.PyTreeCheckpointer()
    empty_stats = cktper.restore(ckpt_fp)

    obs_shape = gym.make(task_name).observation_space.shape
    if alg == "sgd":
        key, *keys = jr.split(key, 1 + cfg["sgd"]["M"])
        keys = jnp.array(keys)
        model = RewardNet(cfg["sgd"]["hidden_sizes"])
        dummy_item = jax.vmap(init_model, in_axes=(0, None, None, None))(
            keys, model, (2, 50, *obs_shape), optax.adam(cfg["sgd"]["learning_rate"])
        )
        ts = cktper.restore(ckpt_fp, item=dummy_item)
        params = {"params": ts.params}

    elif alg == "ekf":
        model = RewardNet(cfg["ekf"]["hidden_sizes"])
        dummy_item = EKFBeliefState(
            mean=empty_stats["mean"],
            cov=empty_stats["cov"],
            t=empty_stats["t"],
            proj_matrix=empty_stats["proj_matrix"],
            offset_ts=init_model(
                key, model, (2, 50, *obs_shape), optax.adam(cfg["ekf"]["learning_rate"])
            ),
        )
        bel = cktper.restore(ckpt_fp, item=dummy_item)
        ts = bel.offset_ts

        params_offset, unravel_fn = ravel_pytree(ts.params)

        distr = distrax.MultivariateNormalFullCovariance(bel.mean, bel.cov)
        ss_params = distr.sample(seed=key, sample_shape=(cfg["ekf"]["M"],))
        full_params_flattened = jax.vmap(subspace2full_params, in_axes=(0, None, None))(
            ss_params, bel.proj_matrix, params_offset
        )
        params = jax.vmap(unravel_fn)(full_params_flattened)
        params = {"params": params}

    else:
        raise ValueError(f"Algorithm {alg} not supported")

    def reward_fn(obs: Float[Array, "T D"]) -> Float[Array, "T"]:
        """M = ensemble size"""
        apply_fn = partial(ts.apply_fn, method=model.predict_traj_rewards)
        out_MT = jax.vmap(apply_fn, in_axes=(0, None))(params, obs)
        return out_MT.mean(axis=0)

    return reward_fn


def relabel_rewards(
    reward_fn: Callable[[Float[Array, "T D"]], Float[Array, "T"]],
    obs: Float[Array, "N obs_dim"],
) -> Float[Array, "N"]:
    def fn(obs_D: Float[Array, "D"]) -> Float[Array, "1 "]:
        obs_D = rearrange(obs_D, "D -> 1 D")
        reward = reward_fn(obs_D)
        return reward

    out = jax.lax.map(fn, obs).squeeze(axis=1)
    return out


if __name__ == "__main__":
    run_dir = "/Users/yutai/dev/projects/bnn_pref/results/20250428_171621"
    task_name = "halfcheetah-random-v2"
    alg = "sgd"
    is_al = False

    key = jr.key(0)
    for alg in ["ekf", "sgd"]:
        reward_fn = load_reward_model(
            key=key,
            run_dir=run_dir,
            task_name=task_name,
            alg=alg,
            is_al=is_al,
        )  # (T,D) -> (T,)

        obs = jnp.zeros((50, 17))
        reward = reward_fn(obs)
        print(reward.shape)
