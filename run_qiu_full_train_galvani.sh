#!/bin/bash
#SBATCH --job-name=qiu_full
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-12:00:00
#SBATCH --output=/mnt/lustre/work/bethge/bkr578/projects/open-retina/openretina_assets/slurm/qiu_full_%j.log
#
# Phase 0b: full qiu_2026 training run on Galvani -- all 10 sessions, batch_size=32, non-debug
# trainer, TensorBoard + CSV logging (config defaults).
#
# Galvani port of run_qiu_full_train.sh (which targets the retired h100-ferranti cluster and its
# /weka paths; kept for the record of the run that produced the published 0.402 checkpoint).
# Differences: a100-galvani instead of h100-ferranti, /mnt/lustre paths instead of /weka, and:
#   * OPENRETINA_CACHE_DIRECTORY is MANDATORY -- configs/qiu_2026_core_readout.yaml resolves
#     paths.cache_dir from ${oc.env:...}, so Hydra dies at config resolution if it is unset.
#   * PATH must include ~/.local/bin -- sbatch does not source ~/.bashrc, so uv is otherwise absent.
#   * paths.data_dir points at the LOCAL pre-staged copy rather than the HuggingFace URL, so the GPU
#     node never needs outbound network (and cannot re-trigger the 132 GB download).
#   * mem 128G / 8 CPU, not 256G / 16: measured construction peak with release_movies is 58.9 GB, and
#     16 GB/CPU exceeds the node's 14.8 GB/CPU ratio, making the job needlessly hard to schedule.
#   * time 2-12:00:00 -- A100 is ~1.5-2x slower than H100 on fp32 convs (precision: "32-true").
#   * save_last=true, cheap insurance: train.py calls trainer.fit() without ckpt_path, so there is no
#     epoch-level resume; last.ckpt at least allows a weight-level restart via paths.load_model_path.
# The thread caps and trainer=default_deterministic are carried over unchanged -- those were the fix
# for the node hang documented in qiu_2026_history.md.
#
#   sbatch --dependency=afterok:<download_jobid> run_qiu_full_train_galvani.sh
set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

export PATH="$HOME/.local/bin:$PATH"
export OPENRETINA_CACHE_DIRECTORY=/mnt/lustre/work/bethge/bkr578/openretina_cache

cd /mnt/lustre/work/bethge/bkr578/projects/open-retina
mkdir -p openretina_assets/slurm

echo "host=$(hostname)  start=$(date -Is)"
nvidia-smi -L || true

uv run openretina train \
  --config-name qiu_2026_core_readout \
  trainer=default_deterministic \
  paths.data_dir=/mnt/lustre/work/bethge/bkr578/openretina_cache/franke_lab/qiu_2026 \
  training_callbacks.model_checkpoint.save_last=true

echo "done=$(date -Is)"
