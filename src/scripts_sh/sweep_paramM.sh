#!/bin/bash

Ms=(5 30 50)
# Ms=(5 15 30 50 100 150)
M_LIST=$(IFS=,; echo "${Ms[*]}")

# NETS=(
#     "32x2"
#     "64x3"
#     "128x3"
#     "256x3"
#     "512x2"
#     "512x3"
#     "1024x2"
#     "1024x3"
# )
# NET_LIST=$(IFS=,; echo "${NETS[*]}")


#* sweep over M; always active, not random querying
# python script runs over {alg, is_al} in sequence, for one task
# keep niters_update = ekf.iekf for fair compute comparison

JAX_PLATFORM_NAME=cpu python scripts/scale_dims_alg.py \
    -m seed=-1 seeds=1 seed_vmap=False \
    task=walkerMediumExpert \
    active=True \
    network=64x2 \
    M=${M_LIST} \
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
    dir_extra=M_infogain_effi_d4rl_cpu_partial \
    hydra/launcher=slurm