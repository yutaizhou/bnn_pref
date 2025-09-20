#!/bin/bash

TASKS=(
    "cheetahRandom"
    "cheetahMediumReplay"
    "cheetahMediumExpert"
    "hopperRandom"
    "hopperMediumReplay"
    "hopperMediumExpert"
    "walkerRandom"
    "walkerMediumReplay"
    "walkerMediumExpert"
    "penHuman"
    # "penExpert"
    # "penCloned"
    # "kitchenComplete"
    # "kitchenPartial"
    # "kitchenMixed"
    # "mazeUDense"
    "mazeMediumDense"
    "mazeLargeDense"
)
TASK_LIST=$(IFS=,; echo "${TASKS[*]}")

# * submitit
python bnn_pref/rl/iql.py \
    -m task=$TASK_LIST \
    rl.n_updates=1000000 \
    rl.eval_interval=25000 \
    rl.reward=pref \
    rl.pref_alg=ekf,sgd,do \
    rl.pref_is_al=True,False \
    rl.normalize_reward=True \
    rl.clip_reward=True \
    rl.run_dir='"/scr/yutaizho/projects/bnn_pref/_runs/pref/20250919_004004_nitersUpdate=10_lr=0.001_ekfM=100_iekf=5_ekfLr=0.003_ekfDn=0.0001_ekfPn=0.07_ekfOn=0.07_ekfAcq=infogain_acq=infogain"' \
    wandb.tags=nitersUpdate10_infogain \
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
