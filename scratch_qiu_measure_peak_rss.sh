#!/bin/bash
#SBATCH --job-name=qiu_rss
#SBATCH --partition=cpu-ferranti
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=01:00:00
#SBATCH --output=/weka/bethge/bkr578/projects/open-retina/openretina_assets/slurm/qiu_rss_%j.log
#
# A/B the peak RSS of qiu_2026 dataloader construction with and without `release_movies`, to decide
# what --mem the sweep tasks actually need. CPU-only (no model, no GPU) and each arm is a SEPARATE
# process, so arm 1's resident memory cannot inflate arm 2's high-water mark.
#
# --mem=250G is deliberately generous: the point is to MEASURE the peak, not to be killed at it. The
# no-release arm is expected around 60-70 GB.

set -euo pipefail

REPO=/weka/bethge/bkr578/projects/open-retina
export OPENRETINA_CACHE_DIRECTORY="${OPENRETINA_CACHE_DIRECTORY:-/weka/bethge/bkr578/openretina_cache}"
NTHREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OMP_NUM_THREADS="$NTHREADS" MKL_NUM_THREADS="$NTHREADS" OPENBLAS_NUM_THREADS="$NTHREADS"

cd "$REPO"
echo "=== cache : $OPENRETINA_CACHE_DIRECTORY"
echo "=== threads: $NTHREADS"

for arm in false true; do
  echo
  echo "############ arm: release_movies=$arm ############"
  uv run python scratch_qiu_measure_peak_rss.py --release "$arm"
done
