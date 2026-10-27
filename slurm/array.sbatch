#!/bin/bash
# Generic array task for make-dconn. Do not sbatch this directly: run
#   make-dconn submit <config> --stage {dconn,zscore,both}
# which freezes the run directory and passes resources on the command line.
#
# Usage (set by make-dconn submit): array.sbatch <run_dir> <venv_python>
set -euo pipefail

RUN_DIR="$1"
PYTHON="$2"

echo "job ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} on $(hostname) at $(date --iso-8601=seconds)"

ml purge
# shellcheck disable=SC2086  # module list is space-separated on purpose
ml ${MAKE_DCONN_MODULES}
ml list 2>&1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"${PYTHON}" -m make_dconn.cli run-array "${RUN_DIR}"

echo "finished at $(date --iso-8601=seconds)"
