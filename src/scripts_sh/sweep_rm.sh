#!/bin/bash

# TASKS=(
#     "cheetahRandom"
#     "cheetahMediumReplay"
#     "cheetahMediumExpert"
#     "hopperRandom"
#     "hopperMediumReplay"
#     "hopperMediumExpert"
#     "walkerRandom"
#     "walkerMediumReplay"
#     "walkerMediumExpert"
#     "penHuman"
#     "penExpert"
#     "penCloned"
#     "mazeUDense"
#     "mazeMediumDense"
#     "mazeLargeDense"
#     "kitchenComplete"
#     "kitchenPartial"
#     "kitchenMixed"
# )
# TASK_LIST=$(IFS=,; echo "${TASKS[*]}")

# # * submitit
# python bnn_pref/rl/iql.py \
#     -m task=$TASK_LIST \
#     rl.reward=pref \
#     rl.pref_alg=ekf,sgd \
#     rl.pref_is_al=True,False \
#     rl.normalize_reward=True \
#     rl.clip_reward=True \
#     rl.n_updates=1000000 \
#     rl.eval_interval=25000 \
#     rl.use_wandb=True \
#     wandb.group=rl_pref_reward_norm_clip \
#     hydra/launcher=slurm 

# this python script runs through all tasks, alg, is_al, in sequence
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/sweep_tasks_alg.py \
    -m seed=-1 seeds=5 \
    data.nq_train=50000 \
    data.nq_update=60,150 \
    sgd.n_epochs=0,1,5 \
    hydra/launcher=slurm
