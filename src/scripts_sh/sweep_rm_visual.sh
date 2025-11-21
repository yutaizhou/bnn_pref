#!/bin/bash

# * iclr tasks
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
#     # "penExpert"
#     # "penCloned"
#     # "kitchenComplete"
#     # "kitchenPartial"
#     # "kitchenMixed"
#     # "mazeUDense"
#     "mazeMediumDense"
#     "mazeLargeDense"
# )

#* vd4rl
TASKS=(
    # "vcheetahRandom"
    # "vcheetahMediumReplay"
    "vcheetahMediumExpert"
    # 'vhumanoidRandom'
    # 'vhumanoidMediumReplay'
    # 'vhumanoidMediumExpert'
    # 'vwalkerRandom'
    # 'vwalkerMediumReplay'
    # 'vwalkerMediumExpert'
)

TASK_LIST=$(IFS=,; echo "${TASKS[*]}")


# this runs through product (alg, is_al) in sequence, for each task
# lr=0.0001 for update_all=True, lr=0.001 for update_all=False
python src/scripts/run_rm.py \
    -m seed=-1 seeds=1 \
    verbose=True \
    task=$TASK_LIST \
    network=resnet18 \
    data.nq_init=200 \
    data.nq_update=250 \
    acq=infogain \
    ekf.rnd_proj=True \
    ekf.proj_type=dense \
    ekf.sub_dim=500 \
    ekf.niters_init=7000 \
    ekf.learning_rate=0.0001 \
    ekf.bs=16 \
    ekf.iekf=5 \
    ekf.use_vmap=False \
    hydra/launcher=slurm \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.mem_gb=80