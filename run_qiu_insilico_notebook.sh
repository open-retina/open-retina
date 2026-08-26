#!/bin/bash
#SBATCH --job-name=qiu_insilico_nb
# a100 and v100 only. `bethge` and `2080-galvani` are both entirely rtx2080ti, and a 2080 Ti is
# too slow for this notebook now that the MEI cells run ~1e5 forward/backward passes -- an earlier
# run landed on one and was heading for the 8 h wall. The trade is queue time: dropping `bethge`
# gives up the lab's own nodes for shared ones.
#SBATCH --partition=a100-galvani,v100-galvani
# Generic gpu:1, not gpu:a100:1 -- the partition list above spans a100 and v100, and a typed gres
# would silently restrict the job to a100-galvani.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00
#SBATCH --output=/mnt/lustre/work/bethge/bkr578/projects/open-retina/openretina_assets/slurm/qiu_insilico_nb_%j.log
#
# Phase 4: execute notebooks/qiu_2026_insilico.ipynb end-to-end against the trained checkpoint,
# writing the outputs back into the notebook so the figures are committed with it.
#
# The wall clock and the per-cell timeout are generous because the MEI cells dominate: 24 boutons x
# 2 states x 1000 iterations, plus a 13-setting knob sweep over 4 boutons. That is roughly 1e5
# forward/backward passes in two single cells, so a 1-hour per-cell limit is not enough.
#
# Runs in-place on a copy first, then swaps, so a mid-run failure cannot leave a half-executed
# notebook in the working tree.
set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PATH="$HOME/.local/bin:$PATH"
export OPENRETINA_CACHE_DIRECTORY=/mnt/lustre/work/bethge/bkr578/openretina_cache
# Deliberately NOT setting MPLBACKEND: Agg renders to an off-screen buffer, so plt.show() emits
# nothing and the executed notebook comes back with zero figures. The notebook selects the
# inline backend itself via `%matplotlib inline`, which is what captures plots as cell outputs.

REPO=/mnt/lustre/work/bethge/bkr578/projects/open-retina
cd "$REPO"

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
echo "start: $(date -Is)"

NB=notebooks/qiu_2026_insilico.ipynb
WORK=notebooks/.qiu_2026_insilico.executing.ipynb

# The newest checkpoint must be the finished one, not a mid-training save.
CKPT=$(ls -t openretina_assets/runs/core_readout_qiu_2026_mouse/*/checkpoints/*.ckpt | head -1)
echo "newest checkpoint: $CKPT"
case "$CKPT" in
  *_final.ckpt) ;;
  *) echo "ABORT: newest checkpoint is not a *_final.ckpt; training may still be running." >&2; exit 1;;
esac

cp "$NB" "$WORK"
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=14400 \
  --ExecutePreprocessor.kernel_name=python3 \
  "$WORK"

mv "$WORK" "$NB"
echo "executed: $(date -Is)"

# Fail loudly if any cell recorded an error output despite nbconvert's exit code.
uv run python - "$NB" <<'PYCHECK'
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
# A notebook whose whole purpose is figures must not come back empty. A non-interactive matplotlib
# backend swallows every plot silently, leaving all cells "successful" with nothing to show.
MIN_FIGURES = 8
if figures < MIN_FIGURES:
    print(f"ABORT: only {figures} figures captured, expected >= {MIN_FIGURES} "
          "(is a non-inline matplotlib backend active?)", file=sys.stderr)
    sys.exit(1)
print("no error outputs")
PYCHECK

echo "COMPLETE: $(date -Is)"
