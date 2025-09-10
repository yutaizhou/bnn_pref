#!/bin/bash

TASKS=(
    "cheetahRandom"
    # "cheetahMediumReplay"
    # "cheetahMediumExpert"
    # "hopperRandom"
    # "hopperMediumReplay"
    # "hopperMediumExpert"
    # "walkerRandom"
    # "walkerMediumReplay"
    # "walkerMediumExpert"
    # "penHuman"
    # "penExpert"
    # "penCloned"
    # "kitchenComplete"
    # "kitchenPartial"
    # "kitchenMixed"
    # "mazeUDense"
    # "mazeMediumDense"
    # "mazeLargeDense"
)
TASK_LIST=$(IFS=,; echo "${TASKS[*]}")

# * submitit
python bnn_pref/rl/iql.py \
    -m task=$TASK_LIST \
    rl.reward=pref \
    rl.pref_alg=ekf,sgd \
    rl.pref_is_al=True,False \
    rl.normalize_reward=True \
    rl.clip_reward=True \
    rl.n_updates=1000000 \
    rl.eval_interval=25000 \
    rl.run_dir='"/scr/yutaizho/projects/bnn_pref/_runs/pref/20250501_002234_rm_d4rl_18tasks"' \
    wandb.tags=nq30 \
    rl.use_wandb=True \
    wandb.group=rl_pref_reward_norm_clip \
    hydra/launcher=slurm 

# * local
# CUDA_VISIBLE_DEVICES=6 python bnn_pref/rl/iql.py \
#     -m task=$TASK_LIST \
#     rl.reward=pref \
#     rl.pref_alg=ekf,sgd \
#     rl.pref_is_al=True,False \
#     rl.n_updates=1000000 \
#     rl.eval_interval=25000 \
#     rl.use_wandb=False \
#     wandb.group=rl_pref \
