from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import RecordVideo

import ogbench
from bnn_pref.utils.utils import get_random_seed

seed = get_random_seed()
key = jr.key(seed)
key, key1, key2 = jr.split(key, 3)

task_name = "antmaze-medium-navigate-singletask-v0"
env, train_trajs, val_trajs = ogbench.make_env_and_datasets(
    task_name,
    compact_dataset=False,
    render_mode="rgb_array",
)


def process_ogbench(ds: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    """
    observations (5000000, 29) -> (10000, 500, 29)
    actions (5000000, 8) -> (10000, 500, 8)
    terminals (5000000,) -> (10000, 500)
    next_observations (5000000, 29) -> (10000, 500, 29)
    rewards (5000000,) -> (10000, 500)
    masks (5000000,) -> (10000, 500)
    returns (5000000,) -> (10000,)

    Returns only obs, actions, returns

    Number of trajectories: 10000
    """
    # seperate trajectories via terminals field
    starts = jnp.where(ds["terminals"])[0]
    ends = jnp.concatenate([jnp.array([-1]), starts[:-1]])
    ds = jax.tree.map(
        lambda x: jnp.array([x[s + 1 : e + 1] for s, e in zip(ends, starts)]),
        ds,
    )
    # sum rewards to get returns, keep only obs, actions, returns
    ds["returns"] = ds["rewards"].sum(axis=-1)
    # ds = {k: ds[k] for k in ["observations", "actions", "returns"]}

    # filter out low return trajectories
    n_traj, traj_len, _ = ds["observations"].shape
    ds = jax.tree.map(lambda x: x[ds["returns"] > -traj_len], ds)

    # sort trajectories by return (ascending)
    sorted_idxes = jnp.argsort(ds["returns"])
    ds = jax.tree.map(lambda x: x[sorted_idxes], ds)

    return ds


for k, v in train_trajs.items():
    print(f"{k}: {v.shape}")
print()
train_trajs = process_ogbench(train_trajs)
val_trajs = process_ogbench(val_trajs)
for k, v in train_trajs.items():
    print(f"{k}: {v.shape}")


def render_trajectory(env, actions):
    """
    Render a trajectory as a sequence of frames.

    Args:
        env: The environment
        actions: Array of shape (T, action_dim)

    Returns:
        frames: List of RGB arrays
    """
    # frames = []
    env = RecordVideo(env, "video.mp4")
    env.reset()
    for action in actions:
        _, _, _, _, _ = env.step(action)
        # frames.append(env.render().copy())
    # return frames
    env.close()


# env = ogbench.make_env_and_datasets(task_name, compact_dataset=False, env_only=True)
render_trajectory(env, train_trajs["actions"][-1])
# print("Frame shape:", frames[0].shape)
# print("Frame min/max:", frames[0].min(), frames[0].max())
