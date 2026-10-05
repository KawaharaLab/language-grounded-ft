#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_project_root="$(cd -- "${script_dir}/../.." && pwd)"
workspace_root="${FT_WORKSPACE_ROOT:-$(dirname -- "${default_project_root}")}"
project_root="${FT_GENESIS_ROOT:-${workspace_root}/Genesis}"
python_bin="${FT_PYTHON:-${project_root}/.venv/bin/python}"
repo_root="$(dirname -- "${workspace_root}")"
runner="${project_root}/src/generation/run_genesis_sim.py"
batch_log="${GENESIS_BATCH_LOG:-${project_root}/data/rigid_two_rounds_08142026.log}"

cd "${project_root}"

export GS_CACHE_FILE_PATH="${GS_CACHE_FILE_PATH:-${repo_root}/.cache/genesis}"
export TARGET_CHOICES="${TARGET_CHOICES:-drop_simple,drop,inclined,step,gentle_simple,firm_simple}"
export GENESIS_GPU_IDS="${GENESIS_GPU_IDS:-0,1}"
export REQUIRED_RUNS_PER_TASK="${REQUIRED_RUNS_PER_TASK:-2}"
# The launcher balances these workers evenly across both GPUs. A local probe
# found ten total workers faster than 2/4/6/8/12 on this 12-core host.
export MAX_PARALLEL_PROCESSES="${MAX_PARALLEL_PROCESSES:-10}"
# Reach the selected concurrency before short simulations finish; retaining a
# small stagger avoids launching all Genesis compilers in the same instant.
export WORKER_SPAWN_DELAY_SECONDS="${WORKER_SPAWN_DELAY_SECONDS:-1}"
GENESIS_DATASETS="${GENESIS_DATASETS:-eval_08142026,train_08142026}"

exec > >(tee -a "${batch_log}") 2>&1

echo "[$(date --iso-8601=seconds)] Starting rigid two-round batch"
echo "Datasets: ${GENESIS_DATASETS}"
echo "Targets: ${TARGET_CHOICES} (inclined and step expand to three variants each)"
echo "Required completed runs per object/target: ${REQUIRED_RUNS_PER_TASK}"
echo "GPUs: ${GENESIS_GPU_IDS}; parallel workers: ${MAX_PARALLEL_PROCESSES}; spawn delay: ${WORKER_SPAWN_DELAY_SECONDS}s"

batch_status=0
IFS=',' read -r -a dataset_names <<< "${GENESIS_DATASETS}"
for dataset_name in "${dataset_names[@]}"; do
    dataset_name="${dataset_name//[[:space:]]/}"
    [[ -n "${dataset_name}" ]] || continue
    # Each pass queues at most one missing run per object/target. Two passes are
    # enough for a new dataset and are idempotent when this script is restarted.
    for reconciliation_pass in 1 2; do
        echo "[$(date --iso-8601=seconds)] START dataset=${dataset_name} reconciliation_pass=${reconciliation_pass}"
        DATASET="${dataset_name}" "${python_bin}" "${runner}"
        run_status=$?
        echo "[$(date --iso-8601=seconds)] END dataset=${dataset_name} reconciliation_pass=${reconciliation_pass} status=${run_status}"
        if (( run_status != 0 )); then
            batch_status=1
        fi
    done
done

echo "[$(date --iso-8601=seconds)] Batch finished status=${batch_status}"
exit "${batch_status}"
