#!/bin/bash
#SBATCH --job-name=qiu_seed
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-12:00:00
#SBATCH --output=/mnt/lustre/work/bethge/bkr578/projects/open-retina/openretina_assets/slurm/qiu_seed_%j.log
#
# Train one additional qiu_2026 model at a given seed, for MEI ensembling.
#
# Why: Qiu 2026 generate dynamic MEIs by gradient ascent on an *ensemble of three models trained
# with different random seeds*. We have one (seed 42), and our single-model MEIs come out spatially
# non-smooth with a cross-seed stimulus correlation of only +0.23. If the model's high-frequency
# spatial filters are fitting pixel-scale noise, that noise is idiosyncratic to one fit -- three
# models will not share it, so the ensemble gradient cancels it. See mei_optimization_reference.md.
#
#   sbatch run_qiu_train_seed.sh 43
#   sbatch run_qiu_train_seed.sh 44
#
# `seed_everything(cfg.seed)` runs immediately before model construction (openretina/cli/train.py:87),
# so the seed does control weight init -- two runs at different seeds are genuinely different models.
#
# Run directory: `exp_name` carries the seed, so runs are identifiable after the fact and two
# simultaneous jobs cannot collide on Hydra's timestamped directory
# (`configs/hydra/default.yaml` names it by `${now:...}` alone, to the second).
#
#   openretina_assets/runs/core_readout_qiu_2026_mouse/<ts>/            <- seed 42, the baseline
#   openretina_assets/runs/core_readout_qiu_2026_mouse_seed43/<ts>/     <- this script
#
# Note the asymmetry: the baseline predates this script and keeps the unsuffixed name. That also
# means notebooks/qiu_2026_insilico.ipynb, which globs the unsuffixed directory for the newest
# checkpoint, keeps picking the seed-42 model and will not silently pick up an ensemble member.
#
# Otherwise identical to run_qiu_full_train_galvani.sh -- same thread caps, deterministic trainer,
# locally staged data dir, and save_last. Those settings are load-bearing; see that script's header
# and qiu_2026_history.md for the node-hang fix.
set -euo pipefail

SEED="${1:-}"
if [ -z "$SEED" ]; then
  echo "usage: sbatch $0 <seed>    (e.g. 43)" >&2
  exit 2
fi
case "$SEED" in
  *[!0-9]*) echo "ABORT: seed must be an integer, got '$SEED'" >&2; exit 2 ;;
esac
if [ "$SEED" = "42" ]; then
  echo "ABORT: seed 42 is the existing baseline run; training it again would just duplicate it." >&2
  echo "       Its checkpoint: openretina_assets/runs/core_readout_qiu_2026_mouse/*/checkpoints/*_final.ckpt" >&2
  exit 2
fi

EXP_NAME="core_readout_qiu_2026_mouse_seed${SEED}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

export PATH="$HOME/.local/bin:$PATH"
export OPENRETINA_CACHE_DIRECTORY=/mnt/lustre/work/bethge/bkr578/openretina_cache

cd /mnt/lustre/work/bethge/bkr578/projects/open-retina
mkdir -p openretina_assets/slurm

echo "host=$(hostname)  start=$(date -Is)"
echo "seed=$SEED  exp_name=$EXP_NAME"
nvidia-smi -L || true

# Refuse to start if this seed already has a finished run -- 3.5 GPU-hours is not worth spending
# twice by accident.
if compgen -G "openretina_assets/runs/${EXP_NAME}/*/checkpoints/*_final.ckpt" > /dev/null; then
  echo "ABORT: ${EXP_NAME} already has a finished checkpoint:" >&2
  ls -1 openretina_assets/runs/"${EXP_NAME}"/*/checkpoints/*_final.ckpt >&2
  exit 1
fi

uv run openretina train \
  --config-name qiu_2026_core_readout \
  seed="$SEED" \
  exp_name="$EXP_NAME" \
  trainer=default_deterministic \
  paths.data_dir=/mnt/lustre/work/bethge/bkr578/openretina_cache/franke_lab/qiu_2026 \
  training_callbacks.model_checkpoint.save_last=true

echo "done=$(date -Is)"
ls -1 openretina_assets/runs/"${EXP_NAME}"/*/checkpoints/ || true
