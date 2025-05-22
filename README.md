# Setup
Install d4rl locally from source:
```bash
git clone git@github.com:Farama-Foundation/D4RL.git
cd D4RL
pip install -e .
```

Install a custom version of dynamax that uses Joseph form EKF update implementation:
```bash
git clone git@github.com:preferenceEKF2025/dynamax_preferenceEKF.git
cd dynamax
pip install -e .
```

Install the rest of the dependencies:
```bash
pip install -r requirements.txt
```

## hydra config
We use the [hydra framework](https://github.com/facebookresearch/hydra) for configuration management.
Our codebase expects a file `.../bnn_pref/src/cfg/local/default.yaml` with the following content:
```yaml
# @package _global_

paths:
  root_dir: PATH/TO/YOUR/bnn_pref

wandb:
  entity: "foo"
  project: "bar"

  # name: null # set in code, display name of run in GUI
  group: null
  # job_type: null # set in code
  tags: null
```
# Commands
In many instances across the codebase, we use "PATH/TO/YOUR/bnn_pref" to refer to the root directory of the project, and "CHANGEME" to refer to the directory that hydra creates after a run.

## Reward learning experiments
To train preference-based reward models over 18 chosen D4RL tasks, each task running 4 algs
(SubspaceEKF (Random), DeepEnsemble (Random), SubspaceEKF (Active), DeepEnsemble (Active))
for 5 seeds each:

```python
python scripts/sweep_tasks_alg.py \
    seed=-1 seeds=5 seed_vmap=False \
    data.nq_train=50000 \
    data.nq_update=60 \
    sgd.max_buffer_size=500 \
    sgd.n_epochs=0 \
```

This will produce 18 tasks * 4 algs * 5 seeds = 360 model checkpoints, and will output a hydra directory in `PATH/TO/YOUR/bnn_pref/results/pref/{DUMMY}` with all the results, where `{DUMMY}` will be replaced by the hydra run name.

A slurm version of the above command is included in the `scripts_sh/sweep_rm.sh` bash script, and will produce results in `PATH/TO/YOUR/bnn_pref/results_sweep/pref/{DUMMY}`. It's good for sweeping over config values, each seperate sweep run will run over all 18 tasks.

For visualization, run `python bnn_pref/src/scripts/viz_logpdf.py` with the `save_dir` variable set to the desired output directory. Set `dirp` variable to the desired reward learning hydra run directory, e.g. `PATH/TO/YOUR/bnn_pref/results/pref/{DUMMY}`. 

## Offline RL experiments
To train offline RL agent using the reward models trained above, on the `cheetahRandom` task, run the following command:

```python
python bnn_pref/rl/iql.py \
    task=cheetahRandom \
    rl.reward=pref \
    rl.pref_alg=ekf,sgd \
    rl.pref_is_al=True,False \
    rl.normalize_reward=True \
    rl.clip_reward=True \
    rl.n_updates=1000000 \
    rl.eval_interval=25000 \
    rl.run_dir='"PATH/TO/YOUR/bnn_pref/results/pref/{DUMMY}"' \
```

This will produce 4 runs (for the 4 reward model algs) in `PATH/TO/YOUR/bnn_pref/results/offline_rl/{DUMMY}`. The 4 models corresponding to cheetahRandom will be chosen automatically by the function `bnn_pref.rl.rm_util.load_reward_model()`. 
To train IQL policies using the environment or zeroed out rewards simply set `rl.reward=gt` or `rl.reward=zero` respectively.

A slurm version of the above command is included in the `scripts_sh/sweep_iql_pref.sh` bash script, and will produce results in `PATH/TO/YOUR/bnn_pref/results_sweep/offline_rl/{DUMMY}`. It also runs over 18 tasks from D4RL, each for 1 seed.

For visualization, run `python bnn_pref/src/scripts/viz_offlineRL.py` with the `save_dir` variable set to the desired output directory. Set `ref_dirp` variable to the desired hydra run directory from training the IQL agent on the GT and zeroed out rewards. Set `pref_dirp` variable to the desired hydra run directory from training the IQL agent on the learned reward models, e.g. `PATH/TO/YOUR/bnn_pref/results/offline_rl/{DUMMY}`.

## Scalability experiments
To run the scalability experiments, use `scripts_sh/sweep_paramCount.sh` to sweep over network sizes with `M` fixed, and use `scripts_sh/sweep_paramM.sh` to sweep over `M` with network size fixed. Both are done only in the active querying setting. This is done over slurm with `jax.vmap` disabled for vectorized network training and prediction. 

For visualization, run `python bnn_pref/src/scripts/viz_scale.py` with the `save_dir` variable set to the desired output directory. Set `M_dirp` variable to the desired hydra run directory from running `sweep_paramCount.sh`. Set `net_dirp` variable to the desired hydra run directory from running `sweep_paramM.sh`.