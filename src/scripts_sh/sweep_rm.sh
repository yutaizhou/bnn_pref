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
    # "kitchenComplete"
    # "kitchenPartial"
    # "kitchenMixed"
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
    update_all=True \
    niters_update=10 \
    sgd.split_datastream=True \
    sgd.M=5 \
    learning_rate=0.001 \
    bs=8 \
    ekf.learning_rate=0.003 \
    ekf.bs=1 \
    ekf.iekf=5 \
    ekf.acq=infogain \
    acq=infogain \
    hydra/launcher=slurm