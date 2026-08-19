# test out state experiments
CUDA_VISIBLE_DEVICES=0 python src/scripts/run_rm.py \
    -m seed=-1 seeds=1 \
    task=cheetahRandom \
    data.nq_init=8 \
    data.nq_update=2 \
    acq=infogain \
    niters_init=10 \
    ekf.rnd_proj=True

# test out pixel experiments
CUDA_VISIBLE_DEVICES=0 python src/scripts/run_rm.py \
    -m seed=-1 seeds=1 \
    task=vcheetahRandom \
    network=resnet18 \
    data.nq_train=10000 \
    data.nq_init=8 \
    data.nq_update=2 \
    acq=infogain \
    niters_init=10 \
    ekf.rnd_proj=True


# test out scaling M 
JAX_PLATFORMS=cpu python src/scripts/run_scaling.py \
    -m seed=-1 seeds=1 \
    task=walkerMediumExpert \
    active=True \
    data.nq_train=50000 \
    data.nq_update=10 \
    network=1024x3 \
    M=5 \
    acq=infogain \
    use_vmap=False \
    bs=8 \
    learning_rate=0.003 \
    niters_init=10 \
    niters_update=10 \
    ekf.learning_rate=0.003 \
    ekf.bs=1 \
    ekf.iekf=5 \
    ekf.rnd_proj=True \
    sgd.split_datastream=True \
    laplace.prior_prec=1000 \
    llmcmc.mcmc_warmups_init=500 \
    llmcmc.mcmc_warmups_update=20 \
    llmcmc.mcmc_steps=1000 \
    dir_extra=M_infogain_cpu

# test out iql pref 
python bnn_pref/rl/iql.py \
    -m task=cheetahRandom \
    rl.n_updates=1000000 \
    rl.eval_interval=50000 \
    rl.reward=pref \
    rl.pref_alg=ekf \
    rl.pref_is_al=True \
    rl.normalize_reward=True \
    rl.clip_reward=True \
    rl.run_dir='results_sweep/pref/<your_pref_run_dir>' \
    wandb.tags=lr=0.003_nitersUpdate=10_infogain \
    rl.use_wandb=False \
    wandb.group=rl_pref_reward_norm_clip 