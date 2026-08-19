# PreferenceEKF (`bnn_pref`)

Code for training preference-based reward models with **PreferenceEKF** and baselines, plus downstream offline RL experiments. The repository uses [Hydra](https://github.com/facebookresearch/hydra) for configuration and [JAX/Flax](https://github.com/google/flax) for model training.

## Algorithms

Reward-learning methods (configured via `algs=` in sweep scripts):

| Config key | Method |
|---|---|
| `ekf` | PreferenceEKF (subspace EKF) |
| `sgd` | Deep ensemble (SGD) |
| `do` | MC dropout |
| `laplace` | Laplace approximation |
| `llmcmc` | Last-layer MCMC |

Each method is run in **random** and **active** query settings (`is_al=False/True`).

Task groupings for plotting live in `src/bnn_pref/utils/task_sets.py` (e.g. `main`, `soar`, `visual`, `visual3`).

---

## Installation

### 1. Clone and install this repo

```bash
git clone <your-repo-url>
cd bnn_pref
pip install -e .
pip install -r requirements.txt
```

### 2. Install D4RL (state-based MuJoCo / Adroit / Maze tasks)

D4RL datasets are downloaded automatically on first use.

```bash
git clone https://github.com/Farama-Foundation/D4RL.git
cd D4RL
pip install -e .
cd ..
```

You also need a MuJoCo license and the legacy `mujoco-py` / gym stack expected by your D4RL version.

### 3. Install custom dynamax (Joseph-form EKF)

```bash
git clone git@github.com:preferenceEKF2025/dynamax_preferenceEKF.git dynamax
cd dynamax
pip install -e .
cd ..
```

### 4. Local Hydra config

Create `src/cfg/local/default.yaml` (this file is gitignored):

```yaml
# @package _global_

paths:
  root_dir: /absolute/path/to/bnn_pref

wandb:
  entity: your-wandb-entity
  project: your-project
  group: null
  tags: null
```

All dataset paths below are relative to `paths.root_dir` unless noted.

---

## Data preparation

### State-based D4RL tasks (`cheetah*`, `hopper*`, `walker*`, `pen*`, `maze*`)

No extra download step beyond D4RL installation. Task configs are in `src/cfg/task/` with `ds_type: d4rl`.

Example:

```bash
python src/scripts/run_rm.py task=cheetahMediumReplay seed=0
```

### VD4RL pixel tasks (`vcheetah*`, `vwalker*`, `vhumanoid*`)

1. Download pixel datasets:

```bash
bash download_d4rl.sh
```

This populates `data/vd4rl/main/<env>/<dataset>/84px/`.

2. Confirm `paths.vd4rl_dir` in `src/cfg/config.yaml` points to `${paths.root_dir}/data/vd4rl` (default).

3. Train with the ResNet encoder:

```bash
python src/scripts/run_rm.py task=vcheetahMediumExpert network=resnet18 seed=0
```

See `src/scripts_sh/sweep_rm_visual.sh` for the full pixel-task sweep.

### SOAR robot tasks (`soar*`)

The postprocessed SOAR proprio dataset is **included in this repo** at `data/soar/soar_pref_proprio/`. After cloning, you can run SOAR tasks directly — no download or preprocessing required.

```bash
python src/scripts/run_rm.py task=soarEggplantPutIn seed=0
```

Each `soar*` task config points at this directory via `paths.soar.pref_proprio_dir` in `src/cfg/config.yaml`. SOAR tasks use binary success/failure preferences (`data/soar.yaml`).

<details>
<summary>Regenerating from raw SOAR (optional)</summary>

Only needed if you want to rebuild `soar_pref_proprio` from scratch:

1. Download the raw numpy release from the [SOAR dataset page](https://rail.eecs.berkeley.edu/datasets/soar_release/numpy_source/) into `data/soar/soar-dataset-local/` (gitignored).
2. Run:

```bash
python -m bnn_pref.data.make_soar_dataset \
    --soar-numpy-root data/soar/soar-dataset-local \
    --out-dir data/soar/soar_pref_proprio
```

</details>

### UniRLHF crowd-sourced labels (optional, `uni*` tasks)

For human crowd-sourced preference labels on D4RL tasks:

1. Download processed labels from the [Clean-Offline-RLHF repo](https://github.com/pickxiguapi/Clean-Offline-RLHF) (`crowdsource_human_labels/`).
2. Place them at:

```
data/unirlhf_crowdsource_human_labels/
  <task_name>_human_labels/
    ...
```

3. Run e.g. `task=uniCheetahMediumExpert`.

---

## Slurm / cluster launches

Sweep scripts under `src/scripts_sh/` submit jobs via [Hydra + Submitit](https://hydra.cc/docs/plugins/submitit_launcher/). Each script ends with `hydra/launcher=slurm` (and often per-job resource overrides like `hydra.launcher.gres=...`).

### Setup

```bash
pip install hydra-submitit-launcher submitit
```

Edit `src/cfg/hydra/launcher/slurm.yaml` for your cluster:

| Field | Typical use |
|---|---|
| `partition` | Slurm partition name |
| `account` | Billing account |
| `gres` | GPU/CPU resources (e.g. `gpu:1`, `shard:12`) |
| `mem_gb` | Memory per job |
| `cpus_per_task` | CPU cores per job |
| `timeout_min` | Wall-clock limit (default 2400 = 40 h) |
| `array_parallelism` | Max concurrent sweep jobs (default 256) |

Job logs and Submitit metadata land under `<sweep_dir>/.submitit/`.

### Slurm vs local

**Slurm sweep** (default in `scripts_sh/`):

```bash
bash src/scripts_sh/sweep_rm.sh
# submits one Slurm array job per Hydra sweep combination
```

**Local / single-machine** — drop the Slurm launcher and run serially on one GPU:

```bash
python src/scripts/run_rm.py \
    -m seed=-1 seeds=1 \
    task=cheetahRandom \
    hydra/launcher=local
```

For a single config with no sweep, omit `-m`:

```bash
CUDA_VISIBLE_DEVICES=0 python src/scripts/run_rm.py task=cheetahRandom seed=0
```

Local launcher config: `src/cfg/hydra/launcher/local.yaml`.

---

## Running experiments

All commands assume the repo root as the working directory. Slurm sweeps use `hydra/launcher=slurm`; see [Slurm / cluster launches](#slurm--cluster-launches) above for setup and local alternatives.

### Reward learning (main benchmark)

State-based tasks from the `main` task set (15 D4RL environments):

```bash
bash src/scripts_sh/sweep_rm.sh
```

Edit the `TASKS=(...)` block at the top of the script to choose tasks. The default script sweeps SOAR tasks; uncomment the `main` D4RL block for state experiments.

Each task runs all algorithms × {random, active} × `seeds` seeds. Outputs go to:

```
results_sweep/pref/<timestamp>_.../<job_id>_task=<task>/
```

Aggregate plots:

```bash
python src/scripts/viz_logpdf_sweep.py \
    --dirp results_sweep/pref/<run_dir> \
    --task_set main
```

Other useful `--task_set` values: `soar`, `soar3`, `visual`, `visual3`, `unirlhf`.

### Reward learning (pixel / VD4RL)

```bash
bash src/scripts_sh/sweep_rm_visual.sh
```

Uses `network=resnet18` and VD4RL image observations.

### Offline RL with learned rewards

First train reward models (above), then point IQL at the sweep directory:

```bash
python src/bnn_pref/rl/iql.py \
    task=cheetahRandom \
    rl.reward=pref \
    rl.pref_alg=ekf,sgd,do \
    rl.pref_is_al=True,False \
    rl.normalize_reward=True \
    rl.clip_reward=True \
    rl.n_updates=1000000 \
    rl.eval_interval=50000 \
    rl.run_dir='"results_sweep/pref/<your_pref_run_dir>"'
```

Slurm sweep over all main tasks:

```bash
export PREF_RUN_DIR=results_sweep/pref/<your_pref_run_dir>
bash src/scripts_sh/sweep_iql_pref.sh
```

Ground-truth and zero-reward baselines:

```bash
bash src/scripts_sh/sweep_iql_baseline.sh
```

Visualize offline RL curves:

```bash
python src/scripts/viz_offlineRL.py \
    --pref_dirp results_sweep/offline_rl/<pref_run> \
    --ref_dirp results_sweep/offline_rl/<baseline_run> \
    --task_set main
```

### Scalability experiments

Sweep network width/depth with fixed ensemble size `M`:

```bash
bash src/scripts_sh/sweep_scalingParam.sh
```

Sweep ensemble size `M` with fixed network:

```bash
bash src/scripts_sh/sweep_scalingM.sh
```

Both use active querying on `walkerMediumExpert` with `use_vmap=False`. Results: `results_sweep/scaling/`.

Plot:

```bash
python src/scripts/viz_scaling.py \
    --M-dirp results_sweep/scaling/<M_sweep_dir> \
    --net-dirp results_sweep/scaling/<net_sweep_dir>
```

### Subdimension ablation

```bash
bash src/scripts_sh/sweep_ekf_subdim.sh
```

Sweeps `ekf.sub_dim` and random projection on/off for `walkerMediumExpert` (3 seeds). Results: `results_sweep/subdim/`.

Plot:

```bash
python src/scripts/viz_subdim_ablation.py \
    --dirp results_sweep/subdim/<run_dir>
```

---

## Visualization scripts

Most plotting scripts take paths via CLI (tyro). No hard-coded machine paths.

| Script | Required args |
|---|---|
| `viz_logpdf_sweep.py` | `--dirp`, `--task_set`, `--alg_set` |
| `viz_offlineRL.py` | `--pref-dirp`, `--ref-dirp` |
| `viz_scaling.py` | `--M-dirp`, `--net-dirp` |
| `viz_subdim_ablation.py` | `--dirp` |
| `viz_rho_projection.py` | set `BNN_PREF_PREF_RUN_DIR` env var |

Optional: set `BNN_PREF_FONT_DIR` to a directory containing `palatinolinotype_roman.ttf` for paper-style plots (defaults to matplotlib fonts otherwise).

---

## Quick local smoke tests

```bash
# state task
CUDA_VISIBLE_DEVICES=0 python src/scripts/run_rm.py \
    -m seed=-1 seeds=1 task=cheetahRandom data.nq_update=2 niters_init=10

# pixel task
CUDA_VISIBLE_DEVICES=0 python src/scripts/run_rm.py \
    -m seed=-1 seeds=1 task=vcheetahRandom network=resnet18 data.nq_update=2 niters_init=10
```

More examples in `src/scripts_sh/commands.sh`.

---

## Repository layout

```
src/
  bnn_pref/
    alg/          # PreferenceEKF and baseline agents
    data/         # dataset loaders (D4RL, VD4RL, SOAR, UniRLHF)
    rl/           # IQL offline RL
    utils/
      resnet/     # vendored Flax ResNet18 encoder (ImageNet pretrained)
      task_sets.py
  cfg/            # Hydra configs (tasks, networks, algorithms)
  scripts/        # training and visualization entry points
  scripts_sh/     # Slurm sweep scripts (hydra/launcher=slurm)
data/
  soar/soar_pref_proprio/  # postprocessed SOAR trajectories (committed)
```

---

## ResNet encoder

Image-based tasks use a vendored Flax ResNet18 (`src/bnn_pref/utils/resnet/`), adapted from [flaxmodels](https://github.com/matthias-wright/flaxmodels) (MIT License). ImageNet weights are downloaded automatically on first use.

---

## Citation

If you use this code, please cite the PreferenceEKF paper (bibtex TBD).
