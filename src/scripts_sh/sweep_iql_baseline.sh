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
    "penExpert"
    "penCloned"
    "kitchenComplete"
    "kitchenPartial"
    "kitchenMixed"
    "mazeUDense"
    "mazeMediumDense"
    "mazeLargeDense"
)
TASK_LIST=$(IFS=,; echo "${TASKS[*]}")

# submitit
python bnn_pref/rl/iql.py \
    -m task=$TASK_LIST \
    rl.reward=zero,gt \
    rl.n_updates=1000000 \
    rl.eval_interval=25000 \
    rl.use_wandb=True \
    wandb.group=rl_baseline \
    hydra/launcher=slurm 



# local
# KITCHEN_TASKS=("kitchenComplete" "kitchenPartial" "kitchenMixed")

# for task in "${KITCHEN_TASKS[@]}"; do
#     echo "Running task: $task"
#     CUDA_VISIBLE_DEVICES=6 python bnn_pref/rl/iql.py \
#         -m task=$task \
#         rl.reward=zero,gt \
#         rl.n_updates=1000000 \
#         rl.eval_interval=25000 \
#         rl.use_wandb=True \
#         wandb.group=rl_baseline
# done

