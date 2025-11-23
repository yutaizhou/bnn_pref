#!/bin/bash

subdims=(10 25 50 75 100 125 150 200 250 400 500)
subdim_list=$(IFS=,; echo "${subdims[*]}")


# this runs through product (alg, is_al) in sequence, for each task
python scripts/run_ekf_subdim.py \
    -m seed=-1 seeds=3 seed_vmap=True \
    task=walkerMediumExpert \
    active=True \
    network=64x2 \
    data.nq_train=150000 \
    data.nq_test=3000 \
    data.nq_update=60 \
    ekf.rnd_proj=False,True \
    ekf.sub_dim=${subdim_list} \
    ekf.niters_init=420 \
    ekf.warm_burns=20 \
    ekf.thinning=2 \
    ekf.learning_rate=0.003 \
    ekf.bs=1 \
    ekf.iekf=5 \
    ekf.acq=infogain \
    hydra/launcher=slurm \
    hydra.launcher.cpus_per_task=3 \