#!/bin/bash
#SBATCH --job-name=qiu_full
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=/weka/bethge/bkr578/projects/open-retina/openretina_assets/slurm/qiu_full_%j.log
#
# Full qiu_2026 DoD training run: all 10 sessions, default batch_size=32, non-debug
# trainer (auto-selects the H100), TensorBoard + CSV logging (config defaults).
# Thread pools capped to the CPU allocation to avoid the oversubscription that hung the
# node when run bare on a shared box. Submit with:  sbatch run_qiu_full_train.sh
set -euo pipefail

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

mkdir -p /weka/bethge/bkr578/projects/open-retina/openretina_assets/slurm
cd /weka/bethge/bkr578/projects/open-retina

uv run openretina train \
  --config-name qiu_2026_core_readout \
  trainer=default_deterministic
