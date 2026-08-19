#!/bin/bash
# Download VD4RL pixel datasets (cheetah, walker, humanoid) into data/vd4rl/.
# Requires: pip install gdown
#
# Expected layout after extraction:
#   data/vd4rl/main/cheetah_run/{random,medium,medium_replay,medium_expert}/84px/
#   data/vd4rl/main/walker_walk/...
#   data/vd4rl/main/humanoid_walk/...

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${ROOT_DIR}/data/vd4rl"
mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

echo "Downloading VD4RL datasets to ${OUT_DIR}"

gdown -O vd4rl.tar.gz 1F4LIH_khOFw1asVvXo82OMa2tZ0Ax5Op # walker_walk
tar xf vd4rl.tar.gz
gdown -O vd4rl.tar.gz 1WR2LfK0y94C_1r2e1ps1dg6zSMHlVY_e # cheetah_run
tar xf vd4rl.tar.gz
gdown -O vd4rl.tar.gz 1zTBL8KWR3o07BQ62jJR7CeatN7vb-vjd # humanoid_walk
tar xf vd4rl.tar.gz
rm vd4rl.tar.gz

echo "Done. Set paths.vd4rl_dir=${OUT_DIR} in your Hydra config (default in config.yaml)."
