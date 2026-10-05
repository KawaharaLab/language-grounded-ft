#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
    echo "Usage: $0 OBJECT TARGET INPUT_DATASET OUTPUT_DATASET" >&2
    echo "Example: $0 001_chips_can gentle_simple eval_08142026 smoke_one_episode" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_project_root="$(cd -- "${script_dir}/../.." && pwd)"
workspace_root="${FT_WORKSPACE_ROOT:-$(dirname -- "${default_project_root}")}"
project_root="${FT_GENESIS_ROOT:-${workspace_root}/Genesis}"
python_bin="${FT_PYTHON:-${project_root}/.venv/bin/python}"
repo_root="$(dirname -- "${workspace_root}")"

export GS_CACHE_FILE_PATH="${GS_CACHE_FILE_PATH:-${repo_root}/.cache/genesis}"

export GENESIS_SINGLE_OBJECT="$1"
export GENESIS_SINGLE_TARGET="$2"
export DATASET="$3"
export OUTPUT_DATASET="$4"
export OUTPUT_VARIANT_LABEL="${OUTPUT_VARIANT_LABEL:-single}"
export GENESIS_CAPTURE_IMAGES="${GENESIS_CAPTURE_IMAGES:-0}"

if [[ "${GENESIS_DEVICE:-cpu}" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "GENESIS_DEVICE is not cpu, but CUDA_VISIBLE_DEVICES is empty" >&2
    exit 2
fi

exec "${python_bin}" "${project_root}/src/generation/run_genesis_sim.py"
