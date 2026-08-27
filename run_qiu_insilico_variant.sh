#!/bin/bash
#SBATCH --job-name=qiu_insilico_var
# a100 and v100 only. `bethge` and `2080-galvani` are both entirely rtx2080ti, and a 2080 Ti is too
# slow for this notebook now that the MEI cells run ~1e5 forward/backward passes.
#SBATCH --partition=a100-galvani,v100-galvani
# Generic gpu:1, not gpu:a100:1 -- the partition list spans a100 and v100, and a typed gres would
# silently restrict the job to a100-galvani.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00
#SBATCH --output=/mnt/lustre/work/bethge/bkr578/projects/open-retina/openretina_assets/slurm/qiu_insilico_var_%j.log
#
# Execute notebooks/qiu_2026_insilico.ipynb at non-default MEI settings, writing results to a
# *variant* subdirectory so runs at different settings sit side by side rather than overwriting each
# other -- comparing them is the point.
#
#   ./run_qiu_insilico_variant.sh                    # smoothness 1000, variant "smooth1000"
#   ./run_qiu_insilico_variant.sh 1000 smooth1000    # the same, explicit
#   ./run_qiu_insilico_variant.sh 0.5 lowcontrast MEI_RMS_FACTOR=0.5
#
# Unlike run_qiu_insilico_notebook.sh this does NOT write executed outputs back over the tracked
# notebook: that one documents the baseline run, and a variant must not silently replace it. The
# executed copy lands next to its own results instead.
#
# Usage: [smoothness_spatial] [variant_name] [extra KEY=VALUE env overrides...]
set -euo pipefail

SMOOTHNESS="${1:-1000}"
# 1000 -> smooth1000, 0.5 -> smooth0p5 (a bare `%.*` would collapse both 0.5 and 0.1 to "smooth0").
VARIANT="${2:-smooth${SMOOTHNESS//./p}}"
if [ $# -ge 1 ]; then shift; fi
if [ $# -ge 1 ]; then shift; fi

export MEI_SMOOTHNESS_SPATIAL="$SMOOTHNESS"
export MEI_VARIANT="$VARIANT"
# Any remaining KEY=VALUE arguments become environment overrides (MEI_RMS_FACTOR, MEI_SMOOTHNESS_TEMPORAL).
for override in "$@"; do
  case "$override" in
    MEI_*=*) export "$override" ;;
    *) echo "ABORT: unrecognised argument '$override' (expected MEI_KEY=VALUE)" >&2; exit 2 ;;
  esac
done

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PATH="$HOME/.local/bin:$PATH"
export OPENRETINA_CACHE_DIRECTORY=/mnt/lustre/work/bethge/bkr578/openretina_cache
# Deliberately NOT setting MPLBACKEND: Agg renders off-screen, so plt.show() emits nothing and the
# executed notebook comes back with zero figures. The notebook selects the inline backend itself.

REPO=/mnt/lustre/work/bethge/bkr578/projects/open-retina
cd "$REPO"

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
echo "variant=$MEI_VARIANT  smoothness_spatial=$MEI_SMOOTHNESS_SPATIAL"
echo "env overrides: $(env | grep -E '^MEI_' | sort | tr '\n' ' ')"
echo "start: $(date -Is)"

# The newest checkpoint must be the finished one, not a mid-training save.
CKPT=$(ls -t openretina_assets/runs/core_readout_qiu_2026_mouse/*/checkpoints/*.ckpt | head -1)
echo "newest checkpoint: $CKPT"
case "$CKPT" in
  *_final.ckpt) ;;
  *) echo "ABORT: newest checkpoint is not a *_final.ckpt; training may still be running." >&2; exit 1;;
esac

NB=notebooks/qiu_2026_insilico.ipynb
WORK="notebooks/.qiu_2026_insilico.${MEI_VARIANT}.executing.ipynb"

cp "$NB" "$WORK"
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=14400 \
  --ExecutePreprocessor.kernel_name=python3 \
  "$WORK"

echo "executed: $(date -Is)"

# Fail loudly if any cell recorded an error output despite nbconvert's exit code.
uv run python - "$WORK" <<'PYCHECK'
import json, sys

notebook = json.load(open(sys.argv[1]))
errors = [
    (i, output.get("ename"), (output.get("evalue") or "")[:200])
    for i, cell in enumerate(notebook["cells"])
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
executed = sum(1 for c in notebook["cells"] if c["cell_type"] == "code" and c.get("execution_count"))
total = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
figures = sum(
    1
    for cell in notebook["cells"]
    for output in cell.get("outputs", [])
    if "image/png" in output.get("data", {})
)
print(f"executed {executed}/{total} code cells, {figures} figures captured")
if errors:
    for i, name, value in errors:
        print(f"  cell {i}: {name}: {value}", file=sys.stderr)
    sys.exit(1)
if executed != total:
    print(f"ABORT: only {executed}/{total} cells ran", file=sys.stderr)
    sys.exit(1)
MIN_FIGURES = 8
if figures < MIN_FIGURES:
    print(f"ABORT: only {figures} figures captured, expected >= {MIN_FIGURES} "
          "(is a non-inline matplotlib backend active?)", file=sys.stderr)
    sys.exit(1)
print("no error outputs")
PYCHECK

# Park the executed notebook with its results rather than over the tracked baseline. The results
# directory is chosen by the notebook itself, so find it rather than reconstructing the path.
RESULTS=$(ls -d openretina_assets/insilico/qiu_2026/*/"$MEI_VARIANT" 2>/dev/null | head -1 || true)
if [ -n "$RESULTS" ]; then
  mv "$WORK" "$RESULTS/executed_notebook.ipynb"
  echo "executed notebook -> $RESULTS/executed_notebook.ipynb"
  echo "results:"
  ls -la "$RESULTS"
else
  echo "WARNING: no results directory matched variant '$MEI_VARIANT'; leaving $WORK in place" >&2
fi

echo "COMPLETE: $(date -Is)"
