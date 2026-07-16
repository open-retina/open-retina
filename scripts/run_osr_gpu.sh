#!/usr/bin/env bash
set -euo pipefail

# Run a full GPU OSR analysis for one registered temporal OpenRetina model.
#
# Usage:
#   ./scripts/run_osr_gpu.sh hoefling_2024_high_res
#   ./scripts/run_osr_gpu.sh karamanlis_2024_mouse
#   ./scripts/run_osr_gpu.sh all
#
# Set OVERWRITE=1 to replace an existing result directory.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL="${1:-}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/openretina-mpl}"

usage() {
    echo "Usage: $0 MODEL"
    echo
    echo "GPU-ready models:"
    echo "  hoefling_2024_low_res"
    echo "  hoefling_2024_high_res"
    echo "  hoefling_2024_base_low_res"
    echo "  karamanlis_2024_mouse"
    echo "  karamanlis_2024_marmoset"
    echo "  maheswaranathan_2023"
    echo "  all"
}

if [[ -z "${MODEL}" || "${MODEL}" == "-h" || "${MODEL}" == "--help" ]]; then
    usage
    exit 0
fi

cd "${REPO_ROOT}"

uv run python - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    print("CUDA is not available in this environment.", file=sys.stderr)
    raise SystemExit(1)

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
PY

COMMON_ARGS=(
    --device cuda
    --auto-time-steps
    --n-flashes 0,1,2,3,4,5,7,10
    --flash-amplitudes 0.25,0.5,0.75
    --flash-frames 1
    --polarity dark
    --channel-mode all
    --value-space model
    --n-jitter-controls 8
    --include-embedded-omission
    --include-sustained-controls
    --top-k-plots 12
)

if [[ "${OVERWRITE:-0}" == "1" ]]; then
    COMMON_ARGS+=(--overwrite)
fi

run_model() {
    local model="$1"
    local batch_size
    local periods

    case "${model}" in
        hoefling_2024_low_res)
            batch_size=16
            periods="2,3,4,5"
            ;;
        hoefling_2024_high_res)
            batch_size=2
            periods="2,3,4,5"
            ;;
        hoefling_2024_base_low_res)
            batch_size=16
            periods="2,3,4,5"
            ;;
        karamanlis_2024_mouse)
            batch_size=2
            periods="5,8,10,13"
            ;;
        karamanlis_2024_marmoset)
            batch_size=4
            periods="5,8,10,13"
            ;;
        maheswaranathan_2023)
            batch_size=16
            periods="2,3,4,5"
            ;;
        *)
            echo "Unsupported model: ${model}" >&2
            usage >&2
            return 2
            ;;
    esac

    echo "Starting full OSR analysis for ${model}"
    uv run python scripts/analyze_osr.py \
        --model "${model}" \
        --batch-size "${batch_size}" \
        --period-frames "${periods}" \
        --output-dir "results/osr_${model}_gpu_full" \
        "${COMMON_ARGS[@]}"
}

if [[ "${MODEL}" == "all" ]]; then
    for model in \
        hoefling_2024_low_res \
        hoefling_2024_high_res \
        hoefling_2024_base_low_res \
        karamanlis_2024_mouse \
        karamanlis_2024_marmoset \
        maheswaranathan_2023
    do
        run_model "${model}"
    done
else
    run_model "${MODEL}"
fi
