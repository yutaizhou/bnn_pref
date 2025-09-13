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


# this runs through product (alg, is_al) in sequence, for each task
# lr=0.0001 for update_all=True, lr=0.001 for update_all=False
python scripts/run_rm.py \
    -m seed=-1 seeds=5 \
    task=$TASK_LIST \
    data.nq_train=100000 \
    data.nq_update=60 \
    update_all=True \
    niters_update=-3 \
    learning_rate=0.0001 \
    ekf.acq=infogain \
    acq=infogain \
    hydra/launcher=slurm
