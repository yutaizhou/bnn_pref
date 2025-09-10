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
python scripts/run_rm.py \
    -m seed=-1 seeds=5 \
    task=$TASK_LIST \
    data.nq_train=100000 \
    data.nq_update=60 \
    sgd.max_buffer_size=500 \
    hydra/launcher=slurm
