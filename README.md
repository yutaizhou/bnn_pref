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

## Reward learning experiments
To train preference-based reward models over chosen D4RL tasks, each task running 6 algs
(`{SubspaceEKF, DeepEnsemble, Dropout} x {Random, Active)}`)
for 5 seeds each:

```bash
bash scripts_sh/sweep_rm.sh
```

This will produce `num_tasks * 6 * num_seeds` model checkpoints, and will output a hydra directory in `PATH/TO/YOUR/bnn_pref/results_sweep/pref/{DUMMY}` with all the results, where `{DUMMY}` will be replaced by the hydra run name. Each seperate sweep run will run over all tasks.

For visualization, 
```bash
python bnn_pref/src/scripts/viz_logpdf_sweep.py {save_dir}
```

With the `save_dir` variable set to the desired output directory, e.g. `PATH/TO/YOUR/bnn_pref/results_sweep/pref/{DUMMY}`.

## Offline RL experiments
To train offline RL agent using the reward models trained above, on the `cheetahRandom` task, run the following command:

```python
python bnn_pref/rl/iql.py \
    task=cheetahRandom \
    rl.reward=pref \
    rl.pref_alg=ekf,sgd,do \
    rl.pref_is_al=True,False \
    rl.normalize_reward=True \
    rl.clip_reward=True \
    rl.n_updates=1000000 \
    rl.eval_interval=50000 \
    rl.run_dir='"PATH/TO/YOUR/bnn_pref/results/pref/{DUMMY}"' \
```

This will produce 6 runs (for the 6 reward model algs) in `PATH/TO/YOUR/bnn_pref/results/offline_rl/{DUMMY}`. The 6 models corresponding to cheetahRandom will be chosen automatically by the function `bnn_pref.rl.rm_util.load_reward_model()`. 
To train IQL policies using the environment or zeroed out rewards simply set `rl.reward=gt` or `rl.reward=zero` respectively.

A slurm version of the above command is included in the `scripts_sh/sweep_iql_pref.sh` bash script, and will produce results in `PATH/TO/YOUR/bnn_pref/results_sweep/offline_rl/{DUMMY}`. It also runs over all tasks from D4RL, each for 1 seed.

For visualization, run `python bnn_pref/src/scripts/viz_offlineRL.py` with the `save_dir` variable set to the desired output directory. Set `ref_dirp` variable to the desired hydra run directory from training the IQL agent on the GT and zeroed out rewards. Set `pref_dirp` variable to the desired hydra run directory from training the IQL agent on the learned reward models, e.g. `PATH/TO/YOUR/bnn_pref/results/offline_rl/{DUMMY}`.

## Scalability experiments
To run the scalability experiments, use `scripts_sh/sweep_scalingParam.sh` to sweep over network sizes with `M` fixed, and use `scripts_sh/sweep_scalingM.sh` to sweep over `M` with network size fixed. Both are done only in the active querying setting. This is done over slurm with `jax.vmap` disabled for vectorized network training and prediction. Results are saved in `PATH/TO/YOUR/bnn_pref/results_sweep/scaling/{DUMMY}`.

For visualization, run `python bnn_pref/src/scripts/viz_scaling.py` with the `save_dir` variable set to the desired output directory. Set `M_dirp` variable to the desired hydra run directory from running `sweep_paramCount.sh`. Set `net_dirp` variable to the desired hydra run directory from running `sweep_paramM.sh`.


## Subdimension ablation experiments
```bash
 bash scripts_sh/sweep_ekf_subdim.sh
```

This runs over one task, `walkerMediumExpert`, with 3 seeds for each each combination of subdimension size and whether to use random projection (11 x 2 = 22 runs). Results are saved in `PATH/TO/YOUR/bnn_pref/results_sweep/subdim/{DUMMY}`.

For visualization, run `python bnn_pref/src/scripts/viz_subdim_ablation.py` with the `save_dir` variable set to the desired output directory. Set `dirp` variable to the desired hydra run directory from running `sweep_ekf_subdim.sh`.
