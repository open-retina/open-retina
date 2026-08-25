#!/bin/bash
#SBATCH --job-name=lnp_grey
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=openretina_assets/slurm/lnp_grey_%j.log
#
# Greyscale LNP on hoefling_2024. See hoefling_2024_lnp_greyscale_plan.md for the full write-up.
#
# The hoefling stimulus is 2-channel (green, UV) and the LNP readout is a per-neuron full-field
# Conv3d((1, 18, 16)), so colour costs 2*18*16 = 576 weights per neuron. Squashing the channels
# to their mean inside DummyCore halves that to 288.
#
# Two arms, differing only in the readout smoothness weight:
#   grey_base      smooth_weight=1     numerically inert (~2e-4 of the Poisson term), so this is
#                                      the like-for-like counterpart of the colour baseline
#                                      (val 0.0860 / test 0.2503, early-stopped at epoch 13).
#   grey_smooth3e4 smooth_weight=3e4   best colour arm (val 0.0983 / test 0.2594, peak epoch 83).
#                                      LaplaceL2norm is the ratio sum(laplace(w)^2)/sum(w^2), so
#                                      it is scale- and channel-count-free and 3e4 should carry
#                                      over -- this arm is what confirms that rather than assuming.
#
# Caveat when reporting: sparse_weight is NOT channel-count invariant. weights_l1() averages over
# in_channels*288 weights, so at a fixed sparse_weight the grey model feels ~2x the per-weight L1
# pressure, and more once the kernel grows to compensate for the mean-not-sum input scaling. The
# earlier sweep found sparse_weight a near no-op between 1 and 1e5 for the colour model, so this
# is expected to be second order -- but it is the one term that does not transfer cleanly.
#
# ---------------------------------------------------------------------------------------------
# PORTABILITY: no absolute paths and no partition/GPU request are baked in, because those are
# site-specific. Submit from the repository root and supply the cluster-specific bits yourself:
#
#   sbatch --partition=<your-gpu-partition> --gres=gpu:1 run_hoefling_lnp_grey.sh
#
# Override any of these via the environment if the defaults do not match your setup:
#   OPENRETINA_CACHE_DIRECTORY  where the movies/responses archives are cached (REQUIRED)
#   OPENRETINA_BIN              path to the `openretina` entry point   (default .venv/bin/openretina)
#   PYTHON_BIN                  path to the python interpreter          (default .venv/bin/python)
#   MAX_EPOCHS                  training epoch cap                      (default 300)
#
# The model is tiny (<1M trainable weights). The colour reference runs took under 3 minutes each
# on one H100 with a measured peak RSS of 4.5 GB, hence --mem=16G and the short walltime; a small,
# short request also backfills far more readily on a busy partition. It runs on CPU too, just
# slower -- drop --gres in that case.
# ---------------------------------------------------------------------------------------------

set -uo pipefail

# Repo root = the directory this script lives in, so the script is location-independent.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO" || exit 1

if [[ -z "${OPENRETINA_CACHE_DIRECTORY:-}" ]]; then
  echo "ERROR: set OPENRETINA_CACHE_DIRECTORY to the data cache directory before submitting." >&2
  exit 1
fi
export OPENRETINA_CACHE_DIRECTORY

OPENRETINA_BIN="${OPENRETINA_BIN:-$REPO/.venv/bin/openretina}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"

for bin in "$OPENRETINA_BIN" "$PYTHON_BIN"; do
  [[ -x "$bin" ]] || { echo "ERROR: $bin not found or not executable." >&2; exit 1; }
done

# Cap BLAS/OMP pools to the allocation; oversubscription on a shared node has frozen it before.
NTHREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OMP_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"

mkdir -p openretina_assets/slurm
BASE="$REPO/openretina_assets/runs/lnp_grey/${SLURM_JOB_ID:-manual}"

# arm_name : smooth_weight
ARMS=(
  "grey_base:1"
  "grey_smooth3e4:3e4"
)

for arm in "${ARMS[@]}"; do
  IFS=':' read -r NAME SMOOTH <<< "$arm"
  OUT="$BASE/$NAME"
  mkdir -p "$OUT"

  echo "=== arm '$NAME' (smooth_weight=$SMOOTH) on $(hostname) with $NTHREADS threads"
  echo "=== output dir: $OUT"

  if ! "$OPENRETINA_BIN" train \
      --config-name hoefling_2024_core_readout_low_res_lnp_grey \
      exp_name="lnp_grey_$NAME" \
      hydra.run.dir="$OUT" \
      trainer.max_epochs="$MAX_EPOCHS" \
      model.readout.smooth_weight="$SMOOTH"; then
    echo "!!! arm '$NAME' failed (continuing)"
    continue
  fi

  # Receipt that the run really was greyscale. The readout channel count is derived by probing
  # the core, so hparams.yaml is the only durable per-run record of it: in_shape [1, 120, 18, 16]
  # means grey, [2, ...] means the squash silently did not happen.
  "$PYTHON_BIN" - "$OUT" "$NAME" <<'RECEIPT'
import pathlib
import sys

import yaml

out_dir, arm = pathlib.Path(sys.argv[1]), sys.argv[2]
hparams_path = out_dir / "csv" / "hparams.yaml"
if not hparams_path.exists():
    print(f"=== arm {arm}: {hparams_path} missing, cannot confirm greyscale")
    raise SystemExit(0)

hparams = yaml.safe_load(hparams_path.read_text())
in_shape = hparams["readout"]["in_shape"]
weights = hparams["core"].get("color_squashing_weights")
verdict = "GREYSCALE" if in_shape[0] == 1 else "*** NOT GREYSCALE ***"
print(f"=== arm {arm}: readout in_shape={in_shape} color_squashing_weights={weights} -> {verdict}")
RECEIPT
done

# Optional convenience aggregation. scratch_lnp_reg_summarize.py is local, untracked tooling from
# the earlier colour sweep; if it is not present, read $OUT/csv/metrics.csv directly (take the max
# of val_evaluation_loss for the peak, and the CorrelationLoss3d/dataloader_idx_2 row for test).
if [[ -f "$REPO/scratch_lnp_reg_summarize.py" ]]; then
  "$PYTHON_BIN" "$REPO/scratch_lnp_reg_summarize.py" "$BASE"/* | tee "$BASE/SUMMARY.txt"
else
  echo "=== scratch_lnp_reg_summarize.py not present; per-arm metrics are in $BASE/*/csv/metrics.csv"
fi
