#!/bin/bash

# Ms=(5 10)
Ms=(5 15 30 50 75 100)
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
JAX_PLATFORMS=cpu python src/scripts/run_scaling.py \
    -m seed=-1 seeds=1,1,1 seed_vmap=False \
    task=walkerMediumExpert \
    active=True \
    data.nq_train=100000 \
    data.nq_update=60 \
    network=64x2 \
    M=${M_LIST} \
    acq=infogain \
    use_vmap=False \
    bs=8 \
    learning_rate=0.003 \
    niters_init=420 \
    niters_update=10 \
    ekf.learning_rate=0.003 \
    ekf.bs=1 \
    ekf.iekf=5 \
    ekf.use_vmap=True \
    sgd.split_datastream=True \
    laplace.prior_prec=1000 \
    llmcmc.mcmc_warmups_init=500 \
    llmcmc.mcmc_warmups_update=20 \
    llmcmc.mcmc_steps=1000 \
    dir_extra=M_infogain_cpu \
    hydra/launcher=slurm \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.gres=null \
    hydra.launcher.mem_gb=120