#!/bin/bash

# Ms=(5 15)
Ms=(5 15 30 50 75 100 150)
M_LIST=$(IFS=,; echo "${Ms[*]}")

# NETS=(
#     "64x2"
#     "64x3"
#     "128x3"
#     "256x3"
#     "512x2"
#     "512x3"
#     "1024x2"
#     "1024x3"
# )
# NET_LIST=$(IFS=,; echo "${NETS[*]}")


#* sweep over M;
# python script runs over {alg, is_al} in sequence, for one task
# keep niters_update = ekf.iekf for fair compute comparison
# python script is meant to be ran with 1 seed at a time. if multiple needed, pass in `seeds=1,1,1` as hydra syntax

# JAX_DISABLE_JIT=1 
JAX_PLATFORMS=cpu python scripts/run_scaling.py \
    -m seed=-1 seeds=1,1,1 seed_vmap=False \
    task=walkerMediumExpert \
    active=True \
    network=64x2 \
    M=${M_LIST} \
    acq=infogain \
    use_vmap=False \
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
    dir_extra=M_infogain_d4rl_cpu \
    hydra/launcher=slurm \
    hydra.launcher.gres=null \
    hydra.launcher.cpus_per_task=6