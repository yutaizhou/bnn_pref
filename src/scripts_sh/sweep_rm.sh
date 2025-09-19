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
    data.nq_train=150000 \
    data.nq_test=3000 \
    data.nq_update=60 \
    data.nq_init=8 \
    update_all=True \
    niters_update=10 \
    sgd.split_datastream=True \
    learning_rate=0.001 \
    bs=8 \
    ekf.learning_rate=0.003 \
    ekf.bs=1 \
    ekf.dynamics_noise=0.0001 \
    ekf.prior_noise=0.07 \
    ekf.obs_noise=0.07 \
    ekf.iekf=5 \
    ekf.acq=infogain \
    acq=infogain \
    hydra/launcher=slurm