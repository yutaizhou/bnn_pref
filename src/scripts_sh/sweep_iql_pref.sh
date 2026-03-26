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
python src/bnn_pref/rl/iql.py \
    -m task=$TASK_LIST \
    rl.n_updates=1000000 \
    rl.eval_interval=50000 \
    rl.reward=pref \
    rl.pref_alg=ekf,sgd,do,laplace,llmcmc \
    rl.pref_is_al=True,False \
    rl.normalize_reward=True \
    rl.clip_reward=True \
    rl.run_dir='"/scr/yutaizho/code/p-prefEKF/bnn_pref/_runs/pref/20251125_070829_nitersUpdate=10_lr=0.003_acq=infogain_M=100"' \
    wandb.tags=lr0.003_nitersUpdate10_infogain \
    rl.use_wandb=True \
    wandb.group=rl_pref_reward_norm_clip \
    hydra/launcher=slurm \
    hydra.launcher.gres=shard:6 \
    hydra.launcher.mem_gb=30

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
