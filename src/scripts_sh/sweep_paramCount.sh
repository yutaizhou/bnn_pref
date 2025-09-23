#!/bin/bash

# Ms=(5 15 30)
# Ms=(5 15 30 50 100 150 200 250)
# M_LIST=$(IFS=,; echo "${Ms[*]}")

NETS=(
    "64x2"
    "64x3"
    "128x3"
    "256x3"
    "512x2"
    "512x3"
    "1024x2"
    "1024x3"
)
NET_LIST=$(IFS=,; echo "${NETS[*]}")


#* sweep over network sizes
# python script runs over {alg, is_al} in sequence, for one task
# keep niters_update = ekf.iekf for fair compute comparison

JAX_PLATFORMS=cpu JAX_DISABLE_JIT=1 python scripts/scale_dims_alg.py \
    -m seed=-1 seeds=1,1,1 seed_vmap=False \
    task=walkerMediumExpert \
    active=True \
    network=${NET_LIST} \
    M=3 \
    data.nq_train=50000 \
    data.nq_update=60 \
    update_all=True \
    niters_update=5 \
    sgd.split_datastream=True \
    learning_rate=0.001 \
    bs=8 \
    ekf.learning_rate=0.003 \
    ekf.bs=1 \
    ekf.iekf=5 \
    ekf.acq=infogain \
    sgd.acq=infogain \
    sgd.use_vmap=False \
    ekf.use_vmap=False \
    ekf.acq=infogain \
    sgd.acq=infogain \
    dir_extra=param_infogain_effi_epochs3_d4rl_cpu_partial \
    hydra/launcher=slurm