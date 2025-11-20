import os
import warnings
from functools import partial
from typing import Callable, Tuple

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
import orbax.checkpoint as ocp
from einops import rearrange
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PRNGKeyArray
from omegaconf import OmegaConf

from bnn_pref.alg.agent_dropout import init_model as init_model_dropout
from bnn_pref.alg.agent_ekf import EKFBeliefState
from bnn_pref.alg.agent_ensemble import init_model as init_model_ensemble
from bnn_pref.alg.projection_matrix import sub2full_params_flat
from bnn_pref.utils.network import RewardNet, count_params


def find_ckpt_fp(
    run_dir: str,
    task_choice: str = "cheetahRandom",
    task_name: str = "halfcheetah-random-v2",
    alg: str = "ekf",
    is_al: bool = False,
) -> str:
    """
    only used for finding ckpts of slurm sweeped RM
    """
    for dirname in os.listdir(run_dir):
        if f"task={task_choice}" in dirname:
            subdir = f"{run_dir}/{dirname}"
            ckpt_fp = f"{subdir}/ckpts/{task_name}_{alg}_al={is_al}"
            cfg_path = f"{subdir}/.hydra/config.yaml"
            return ckpt_fp, cfg_path
    raise ValueError(f"CKPT for {task_choice}_{alg}_al={is_al} not found")


def load_reward_model(
    key: PRNGKeyArray,
    run_dir: str,
    task_name: str = "halfcheetah-random-v2",
    alg: str = "ekf",
    is_al: bool = False,
    sweeped_rm: bool = True,
    task_choice: str = "cheetahRandom",
) -> Tuple[Callable[[Float[Array, "T D"]], Float[Array, "T"]], str]:
    """
    run_dir: hydra run output directory, e.g. "../results/2025***"
    key: used for flax.linen.init and EKF subspace parameter sampling

    Assumes ckpts are named as follows, and exists in <run_dir>/ckpts/
        <run_dir>/ckpts/<task_name>_<alg>_al=<is_al>

    """

    if sweeped_rm:
        # find hydra choice override folder path
        ckpt_fp, cfg_path = find_ckpt_fp(run_dir, task_choice, task_name, alg, is_al)
        cfg = OmegaConf.load(cfg_path)
    else:  # deprecated convention
        ckpts_dir = f"{run_dir}/ckpts"
        ckpt_fp = f"{ckpts_dir}/{task_name}_{alg}_al={is_al}"
        cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")

    cktper = ocp.PyTreeCheckpointer()
    sharding = jax.sharding.PositionalSharding(jax.local_devices())

    obs_shape = gym.make(task_name).observation_space.shape
    traj_shape = (50, *obs_shape)

    if alg == "sgd":
        sgd_cfg = cfg["sgd"]
        model = RewardNet(sgd_cfg["hidden_sizes"])
        key, key_init = jr.split(key)
        dummy_item = jax.vmap(init_model_ensemble, in_axes=(0, None, None, None))(
            jr.split(key_init, sgd_cfg["M"]),
            model,
            optax.adam(sgd_cfg["learning_rate"]),
            traj_shape,
        )
        dummy_items = jax.tree.map(lambda x: jax.device_put(x, sharding), dummy_item)
        restore_kw = {
            "restore_args": ocp.checkpoint_utils.construct_restore_args(
                dummy_items, jax.tree.map(lambda _: sharding, dummy_items)
            )
        }
        ts = cktper.restore(ckpt_fp, item=dummy_items, **restore_kw)
        params = {"params": ts.params}

    elif alg == "do":
        do_cfg = cfg["do"]
        model = RewardNet(do_cfg["hidden_sizes"], dropout_prob=do_cfg["dropout_prob"])
        key, key_init = jr.split(key)
        dummy_item = init_model_dropout(
            key_init,
            model,
            optax.adam(do_cfg["learning_rate"]),
            traj_shape,
        )
        dummy_item = dummy_item.replace(dropout_key=jr.key_data(dummy_item.dropout_key))
        dummy_items = jax.tree.map(lambda x: jax.device_put(x, sharding), dummy_item)
        restore_kw = {
            "restore_args": ocp.checkpoint_utils.construct_restore_args(
                dummy_items, jax.tree.map(lambda _: sharding, dummy_items)
            )
        }
        ts = cktper.restore(ckpt_fp, item=dummy_items, **restore_kw)
        params = {"params": ts.params}

        key, *key_dropout = jr.split(key, cfg["do"]["M"] + 1)

        def reward_fn(obs: Float[Array, "T D"]) -> Float[Array, "T"]:
            obs = rearrange(obs, "T D -> 1 T D")
            apply_fn = lambda key, obs: ts.apply_fn(
                {"params": ts.params},
                obs,
                method=model.predict_traj_rewards,
                train=True,
                rngs={"dropout": key},
            )
            out_MT = jax.vmap(apply_fn, in_axes=(0, None))(jnp.array(key_dropout), obs)
            return out_MT.mean(axis=0)

        return reward_fn, ckpt_fp

    elif alg == "ekf":
        ekf_cfg = cfg["ekf"]
        model = RewardNet(ekf_cfg["hidden_sizes"])
        key, key_init = jr.split(key)
        dummy_ts = init_model_ensemble(
            key_init,
            model,
            optax.sgd(ekf_cfg["learning_rate"], momentum=ekf_cfg["momentum"]),
            traj_shape,
        )
        full_dim = count_params(dummy_ts.params)
        sub_dim = ekf_cfg["sub_dim"]

        dummy_item = EKFBeliefState(
            mean=jnp.zeros((sub_dim,)),
            cov=jnp.eye(sub_dim),
            t=0,
            proj_matrix=jnp.zeros((sub_dim, full_dim)),
            offset_ts=dummy_ts,
        )
        dummy_items = jax.tree.map(lambda x: jax.device_put(x, sharding), dummy_item)
        restore_kw = {
            "restore_args": ocp.checkpoint_utils.construct_restore_args(
                dummy_items, jax.tree.map(lambda _: sharding, dummy_items)
            )
        }
        bel = cktper.restore(ckpt_fp, item=dummy_items, **restore_kw)
        ts = bel.offset_ts

        key, key_sample = jr.split(key)
        distr = distrax.MultivariateNormalFullCovariance(bel.mean, bel.cov)
        ss_params = distr.sample(
            seed=key_sample, sample_shape=(ekf_cfg["M"],)
        )  # (M, subdim)
        params_offset_flat, unravel_fn = ravel_pytree(ts.params)
        # projection type option was added later on
        proj_type = ekf_cfg.get("proj_type", "dense")
        params_flat = jax.vmap(
            partial(sub2full_params_flat, type=proj_type),
            in_axes=(0, None, None),
        )(ss_params, bel.proj_matrix, params_offset_flat)  # (M, full_dim)
        params = jax.vmap(unravel_fn)(params_flat)
        params = {"params": params}

    else:
        raise ValueError(f"Algorithm {alg} not supported")

    def reward_fn(obs: Float[Array, "T D"]) -> Float[Array, "T"]:
        apply_fn = partial(ts.apply_fn, method=model.predict_traj_rewards)  # (T,D -> T)
        out_MT = jax.vmap(apply_fn, in_axes=(0, None))(params, obs)
        return out_MT.mean(axis=0)

    return reward_fn, ckpt_fp


def relabel_rewards(
    reward_fn: Callable[[Float[Array, "T D"]], Float[Array, "T"]],
    obs: Float[Array, "N obs_dim"],
) -> Float[Array, "N"]:
    def fn(obs_D: Float[Array, "D"]) -> Float[Array, "1 "]:
        obs_D = rearrange(obs_D, "D -> 1 D")
        reward = reward_fn(obs_D)
        return reward

    out = jax.lax.map(fn, obs).squeeze(axis=1)  # (N,)
    return out


if __name__ == "__main__":
    run_dir = "/Users/yutai/dev/projects/bnn_pref/results/20250428_171621"
    task_name = "halfcheetah-random-v2"
    alg = "sgd"
    is_al = False

    key = jr.key(0)
    for alg in ["ekf", "sgd"]:
        reward_fn, ckpt_fp = load_reward_model(
            key=key,
            run_dir=run_dir,
            task_name=task_name,
            alg=alg,
            is_al=is_al,
        )  # (T,D) -> (T,)

        obs = jnp.zeros((50, 17))
        reward = reward_fn(obs)
        print(reward.shape)
        print(ckpt_fp)
