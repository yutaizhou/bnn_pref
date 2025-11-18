import os
from pathlib import Path

import ipdb

os.environ["MUJOCO_GL"] = "egl"


import numpy as np
from dm_env import specs

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

    load_offline_dataset_into_buffer(
        offline_dir=fdir,
        replay_buffer=replay_buffer,
        frame_stack=cfg["frame_stack"],
        replay_buffer_size=cfg["buffer_size"],
    )
    print(f"Loaded {fdir} into replay buffer")
    print(f"Replay buffer size: {len(replay_buffer)}")

    batch = next(replay_buffer)
    for k, v in batch.items():
        print(k, v.shape)
