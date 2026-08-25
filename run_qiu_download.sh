#!/bin/bash
#SBATCH --job-name=qiu_dl
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/lustre/work/bethge/bkr578/projects/open-retina/openretina_assets/slurm/qiu_dl_%j.log
#
# Phase 0a: pre-stage the ~132 GB qiu_2026 dataset from HuggingFace into the Lustre cache.
#
# Run on cpu-galvani (30-day limit, no GPU) rather than inside the training allocation: the loaders
# would otherwise trigger the download lazily on their first call and burn GPU hours on I/O.
#
# Resumable: download_file_from_huggingface() checks is_target_present() first and skips any session
# already downloaded or already unzipped (file_utils.py:198-202), so re-running after a timeout picks
# up where it left off. The verification block below is what gates the dependent training job.
#
#   sbatch run_qiu_download.sh
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"                  # sbatch does not source ~/.bashrc; uv lives here
export OPENRETINA_CACHE_DIRECTORY=/mnt/lustre/work/bethge/bkr578/openretina_cache

cd /mnt/lustre/work/bethge/bkr578/projects/open-retina

echo "host=$(hostname)  cache=$OPENRETINA_CACHE_DIRECTORY"
echo "start: $(date -Is)"

# --- quota preflight -------------------------------------------------------
# The first attempt (job 2778962) died at 88 GB with OSError 122 "Disk quota exceeded" mid-extraction.
# `lfs quota` cannot show this user's limit (it reports the default as 0/0 and `lfs quota -U` is
# permission-denied), so the cap below is the empirically observed wall: writes began failing at
# 454.4 G. Fail fast and loudly here rather than halfway through unpacking a 24 GB session.
QUOTA_CAP_GB=450
TOTAL_TARGET_GB=132          # full extracted dataset
TRANSIENT_GB=24              # largest single session, present as zip + extraction simultaneously

used_gb=$(lfs quota -u "$USER" /mnt/lustre 2>/dev/null | awk 'NR==3{printf "%d", $2/1024/1024}')
have_gb=$(du -sm "$OPENRETINA_CACHE_DIRECTORY/franke_lab" 2>/dev/null | awk '{printf "%d", $1/1024}')
have_gb=${have_gb:-0}
need_gb=$(( TOTAL_TARGET_GB - have_gb + TRANSIENT_GB ))
[ "$need_gb" -lt 0 ] && need_gb=0

echo "quota preflight: used=${used_gb}G have=${have_gb}G need=${need_gb}G cap=${QUOTA_CAP_GB}G"
if [ $(( used_gb + need_gb )) -gt "$QUOTA_CAP_GB" ]; then
  echo "ABORT: need ${need_gb}G but only $(( QUOTA_CAP_GB - used_gb ))G headroom under the ~${QUOTA_CAP_GB}G user quota." >&2
  echo "       Free space or request a quota increase, then resubmit (the download resumes)." >&2
  exit 1
fi
# ---------------------------------------------------------------------------

uv run python -c "
from openretina.utils.file_utils import huggingface_download
print(huggingface_download('franke_lab/qiu_2026'), flush=True)
"

echo "download returned: $(date -Is)"

# Gate the dependent training job on a genuinely complete dataset, not just a zero exit code.
QIU_DIR="$OPENRETINA_CACHE_DIRECTORY/franke_lab/qiu_2026"
n_sessions=$(find "$QIU_DIR" -maxdepth 1 -type d -name 'dynamic*' | wc -l)
n_masks=$(find "$QIU_DIR/data-quality" -name '*_neurons_fluor_good.npy' 2>/dev/null | wc -l)
n_zips=$(find "$QIU_DIR" -maxdepth 1 -name '*.zip' | wc -l)
echo "sessions=$n_sessions masks=$n_masks leftover_zips=$n_zips"
du -sh "$QIU_DIR" 2>/dev/null || true

if [ "$n_sessions" -ne 10 ] || [ "$n_masks" -ne 10 ]; then
  echo "INCOMPLETE: expected 10 session dirs and 10 quality masks. Re-run this script to resume." >&2
  exit 1
fi
echo "COMPLETE: $(date -Is)"
