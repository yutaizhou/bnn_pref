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

# this script runs through all product(task, alg, is_al) in sequence
python scripts/sweep_tasks_alg.py \
    -m seed=-1 seeds=5 seed_vmap=False \
    data.nq_train=50000 \
    data.nq_update=60 \
    sgd.max_buffer_size=500 \
    sgd.n_epochs=0 \
    hydra/launcher=slurm
