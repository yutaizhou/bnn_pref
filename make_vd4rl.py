import os
from pathlib import Path

import ipdb
import jax.random as jr

os.environ["MUJOCO_GL"] = "egl"


import numpy as np
from dm_env import specs

from bnn_pref.data.make_d4rl import make_d4rl_data
from bnn_pref.data.vd4rl_utils import (
    EfficientReplayBuffer,
    load_offline_dataset_into_buffer,
    make,
)

if __name__ == "__main__":
    vd4rl_path = Path("/scr/shared/datasets/vd4rl")
    task_name = "cheetah_run"
    dataset_name = "medium_expert"
    fdir = vd4rl_path / "main" / task_name / dataset_name / "84px"

    cfg = {
        "task_name": task_name,
        "dataset_name": dataset_name,
        "fdir": fdir,
        "frame_stack": 10,
        "action_repeat": 1,
        "seed": 0,
        "buffer_size": int(1e6),
        "batch_size": 64,
        "nstep": 1,
        "discount": 0.99,
        "sarsa": False,
    }

    env = make(
        name=task_name,
        frame_stack=cfg["frame_stack"],
        action_repeat=cfg["action_repeat"],
        seed=cfg["seed"],
    )

    data_specs = (
        env.observation_spec(),
        env.action_spec(),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
    )

    replay_buffer = EfficientReplayBuffer(
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        nstep=cfg["nstep"],
        discount=cfg["discount"],
        frame_stack=cfg["frame_stack"],
        data_specs=data_specs,
        sarsa=cfg["sarsa"],
    )

    # trajs = make_vd4rl_data(
    #     offline_dir=fdir,
    # )
    # print(f"Loaded {fdir} into replay buffer")
    # print(f"Replay buffer size: {len(replay_buffer)}")

    # batch = next(replay_buffer)
    # for k, v in batch.items():
    #     print(k, v.shape)

    from hydra import compose, initialize

    from bnn_pref.utils.utils import get_random_seed

    with initialize(
        version_base=None, config_path="/scr/yutaizho/code/p-prefEKF/bnn_pref/src/cfg"
    ):
        cfg = compose(config_name="config", overrides=["task=vcheetahMediumReplay"])

    key = jr.key(get_random_seed())
    data = make_d4rl_data(key, cfg)
    train_trajs, test_trajs = data["train_trajs"], data["test_trajs"]
    train_prefs, test_prefs = data["train_prefs"], data["test_prefs"]
    print(f"{train_trajs['observations'].shape=}")
    print(f"{test_trajs['observations'].shape=}")
    print(f"{train_prefs.queries_Q2.shape=}")
    print(f"{test_prefs.queries_Q2.shape=}")
