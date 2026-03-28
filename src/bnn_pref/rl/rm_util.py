import os
import warnings
from typing import Callable, Tuple

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import d4rl
import gym
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray
from omegaconf import OmegaConf

from bnn_pref.alg import alg_classes


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
    agg_type: str = "mean",
) -> Tuple[Callable, str]:
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

    obs_shape = gym.make(task_name).observation_space.shape
    traj_shape = (50, *obs_shape)

    alg_class = alg_classes[alg]
    reward_fn = alg_class.load_reward_model(
        key=key,
        cfg=cfg,
        traj_shape=traj_shape,
        ckpt_fp=ckpt_fp,
    )  # Callable: (T,D) -> (M,T)

    def agg_reward_fn(obs: Float[Array, "T D"]) -> Float[Array, "T"]:
        """
        reward_fn computes ensemble rewards for each trajectory step. This function
        aggregates the ensemble rewards into a single reward for each trajectory step.

        Args:
            obs: (T,D)
            agg_type: "mean" or "max" or "min"
            reward_fn: (T,D) -> (M,T)
        Returns:
            Callable: (T,D) -> (T,)
        reward_fn: (T,D) -> (M,T)
        """
        rewards = reward_fn(obs)  # (M,T)
        mu = jnp.mean(rewards, axis=0)
        if agg_type == "mean":
            return mu
        elif agg_type == "median":
            return jnp.median(rewards, axis=0)
        elif agg_type == "std_penalty":
            std = jnp.std(rewards, axis=0)  # (T,)
            std = jnp.nan_to_num(std, nan=0, posinf=0, neginf=0)
            return mu - std * 0.2
        else:
            raise ValueError(f"Invalid aggregation type: {agg_type}")

    return agg_reward_fn, ckpt_fp


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
