#!/bin/bash

# * iclr tasks
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
    # "penExpert"
    # "penCloned"
    # "kitchenComplete"
    # "kitchenPartial"
    # "kitchenMixed"
    # "mazeUDense"
    "mazeMediumDense"
    "mazeLargeDense"
)

# * testout state exps
# TASKS=(
#     "cheetahMediumReplay"
#     "hopperMediumReplay"
#     "walkerMediumReplay"
#     "penHuman"
#     "mazeMediumDense"
# )

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
    -m seed=-1 seeds=5 \
    task=$TASK_LIST \
    data.nq_init=8 \
    data.nq_update=60 \
    acq=infogain \
    bs=8 \
    M=100 \
    sgd.M=5 \
    learning_rate=0.003 \
    niters_init=420 \
    niters_update=10 \
    ekf.learning_rate=0.003 \
    ekf.bs=1 \
    ekf.iekf=5 \
    sgd.split_datastream=True \
    laplace.prior_prec=1000 \
    laplace.niters_update=10 \
    llmcmc.mcmc_warmups_init=500 \
    llmcmc.mcmc_warmups_update=20 \
    llmcmc.mcmc_steps=1000 \
    hydra/launcher=slurm \
    hydra.launcher.gres=shard:12 \
    hydra.launcher.mem_gb=50