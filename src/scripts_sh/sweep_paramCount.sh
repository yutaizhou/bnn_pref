#!/bin/bash

# Ms=(5 15 30)
# Ms=(5 15 30 50 100 150 200 250)
# M_LIST=$(IFS=,; echo "${Ms[*]}")

NETS=(
    "32x2"
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

JAX_PLATFORM_NAME=cpu python scripts/scale_dims_alg.py \
    -m seed=-1 seeds=1 seed_vmap=False \
    task=acrobot \
    active=True \
    network=${NET_LIST} \
    M=5 \
    data.nq_train=50000 \
    data.nq_update=60 \
    sgd.max_buffer_size=500 \
    sgd.n_epochs=0 \
    sgd.use_vmap=False \
    ekf.use_vmap=False \
    dir_extra=scale_param_active \
    data.segment_size=-1 \
    network.n_splits=5 \
    hydra/launcher=slurm