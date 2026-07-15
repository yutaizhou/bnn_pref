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
    'vhumanoidMediumExpert'
    'vwalkerRandom'
    'vwalkerMediumReplay'
    'vwalkerMediumExpert'
)

TASK_LIST=$(IFS=,; echo "${TASKS[*]}")
ALG_LIST="ekf,sgd,do"
DIR_EXTRA="tmlr_vd4rl_baselines"

# this runs through product (alg, is_al) in sequence, for each task
python src/scripts/run_rm.py \
    -m seed=-1 seeds=5 \
    "algs=[${ALG_LIST}]" \
    verbose=True \
    task=$TASK_LIST \
    network=resnet18 \
    data.nq_init=150 \
    data.nq_update=100 \
    data.vd4rl_64=False \
    data.vd4rl_segment_size=10 \
    acq=infogain \
    M=20 \
    sgd.M=5 \
    bs=16 \
    learning_rate=0.003 \
    niters_init=420 \
    niters_update=10 \
    sgd.split_datastream=True \
    laplace.prior_prec=1000 \
    laplace.curv_type=diagonal \
    llmcmc.mcmc_warmups_init=500 \
    llmcmc.mcmc_warmups_update=20 \
    llmcmc.mcmc_steps=1000 \
    ekf.rnd_proj=True \
    ekf.proj_type=dense \
    ekf.sub_dim=500 \
    ekf.niters_init=3000 \
    ekf.learning_rate=0.0001 \
    ekf.bs=16 \
    ekf.iekf=5 \
    ekf.M=20 \
    ekf.use_vmap=False \
    dir_extra="'${DIR_EXTRA}'" \
    hydra/launcher=slurm \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.mem_gb=80
