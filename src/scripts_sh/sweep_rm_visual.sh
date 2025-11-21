#!/bin/bash

# * iclr tasks
TASKS=(
    # "cheetahRandom"
    # "vcheetahMediumReplay"
    "vcheetahMediumExpert"
    # "hopperRandom"
    # "hopperMediumReplay"
    # "hopperMediumExpert"
    # "walkerRandom"
    # "walkerMediumReplay"
    # "walkerMediumExpert"
    # "penHuman"
    # "penExpert"
    # "penCloned"
    # "kitchenComplete"
    # "kitchenPartial"
    # "kitchenMixed"
    # "mazeUDense"
    # "mazeMediumDense"
    # "mazeLargeDense"
)

#* vd4rl
# TASKS=(
#     "vcheetahRandom"
#     # "vcheetahMediumReplay"
#     # "vcheetahMediumExpert"
#     # 'vhumanoidRandom'
#     # 'vhumanoidMediumReplay'
#     # 'vhumanoidMediumExpert'
#     # 'vwalkerRandom'
#     # 'vwalkerMediumReplay'
#     # 'vwalkerMediumExpert'
# )

TASK_LIST=$(IFS=,; echo "${TASKS[*]}")


# this runs through product (alg, is_al) in sequence, for each task
# lr=0.0001 for update_all=True, lr=0.001 for update_all=False
python src/scripts/run_rm.py \
    -m seed=-1 seeds=1 \
    task=$TASK_LIST \
    data.nq_init=100 \
    data.nq_update=120 \
    acq=infogain \
    update_all=True \
    network=resnet18 \
    ekf.rnd_proj=True \
    ekf.proj_type=dense \
    ekf.sub_dim=500 \
    ekf.niters_init=5000 \
    ekf.learning_rate=0.0001 \
    ekf.bs=16 \
    ekf.iekf=5 \
    ekf.use_vmap=False \
    hydra/launcher=slurm \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.mem_gb=80