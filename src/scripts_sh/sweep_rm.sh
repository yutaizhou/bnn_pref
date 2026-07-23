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

# * uniRLHF
# TASKS=(
#     "uniCheetahMedium"
#     "uniCheetahMediumReplay"
#     "uniCheetahMediumExpert"
#     "uniHopperMedium"
#     "uniHopperMediumReplay"
#     "uniHopperMediumExpert"
#     "uniWalkerMedium"
#     "uniWalkerMediumReplay"
#     "uniWalkerMediumExpert"
#     "uniPenHuman"
#     "uniPenCloned"
# )

# * testout state exps
# TASKS=(
#     "cheetahMediumReplay"
#     "hopperMediumReplay"
#     "walkerMediumReplay"
#     "penHuman"
#     "mazeMediumDense"
# )

# * soar (>=100 trajs, >=45% success)
TASKS=(
    "soarEggplantPutIn"
    "soarBlueBlockPutIn"
    "soarCloseDrawer"
    "soarMushroomRemove"
    "soarSpoonMoveLeft"
    "soarCarrotRemove"
    "soarEggplantRemove"
)

TASK_LIST=$(IFS=,; echo "${TASKS[*]}")
ALG_LIST="ekf,sgd,do,laplace,llmcmc"

DIR_EXTRA="tmlr_seeds=5_soar"


# this runs through product (alg, is_al) in sequence, for each task
python src/scripts/run_rm.py \
    -m seed=-1 seeds=5 \
    "algs=[${ALG_LIST}]" \
    task=$TASK_LIST \
    data.nq_init=8 \
    data.nq_update=60 \
    acq=infogain \
    M=100 \
    sgd.M=5 \
    bs=8 \
    learning_rate=0.003 \
    niters_init=420 \
    niters_update=10 \
    ekf.rnd_proj=True \
    ekf.niters_init=1020 \
    ekf.sub_dim=500 \
    ekf.learning_rate=0.003 \
    ekf.bs=1 \
    ekf.prior_noise=0.07 \
    ekf.obs_noise=0.07 \
    ekf.iekf=2 \
    sgd.split_datastream=True \
    laplace.prior_prec=1000 \
    laplace.curv_type=full \
    llmcmc.mcmc_warmups_init=500 \
    llmcmc.mcmc_warmups_update=20 \
    llmcmc.mcmc_steps=500 \
    dir_extra="'${DIR_EXTRA}'" \
    hydra/launcher=slurm \
    hydra.launcher.gres=shard:12 \
    hydra.launcher.mem_gb=50