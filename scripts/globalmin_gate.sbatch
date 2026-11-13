#!/bin/bash
#SBATCH --job-name=gaab-globalmin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/globalmin-%j.out
#SBATCH --error=logs/globalmin-%j.err
#
# Full global-minimum gate (K=64) over the curated suite.
#
# NOTE ON THE GPU: this workload is pure NumPy and runs entirely on CPU -- the
# requested GPU is NOT used.  It is here only because it was asked for; drop the
# `--gres=gpu:1` line above to avoid waiting for a GPU and to free it for jobs
# that need one.  Nothing else changes.
#
# The gate is embarrassingly parallel over (spec, instance) pairs and spreads
# them across the allocated CPUs (--jobs below == --cpus-per-task).  Each worker
# pins BLAS to a single thread (the tiny matmuls make threaded BLAS pure
# overhead and oversubscription actively harmful), so the exports below just
# make that explicit at the job level too.
#
# Submit from the repository root:
#     sbatch scripts/globalmin_gate.sbatch
# Override a knob at submit time, e.g. a shorter smoke run:
#     RESTARTS=8 sbatch scripts/globalmin_gate.sbatch

set -euo pipefail

# --- knobs (override via environment at submit time) -----------------------
SUITE="${SUITE:-suites/grokking_v1.json}"
OUT_SUITE="${OUT_SUITE:-suites/grokking_v1_globalmin.json}"
OUT_DIR="${OUT_DIR:-results/globalmin}"
RESTARTS="${RESTARTS:-64}"
MEMORIZERS="${MEMORIZERS:-4}"
STEPS="${STEPS:-60000}"
MARGIN="${MARGIN:-0.0}"

# --- environment -----------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs "${OUT_DIR}"

# One BLAS thread per process; the pool supplies the parallelism.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

JOBS="${SLURM_CPUS_PER_TASK:-16}"

echo "host=$(hostname)  cpus=${JOBS}  started=$(date -Is)"
echo "suite=${SUITE}  restarts=${RESTARTS}  memorizers=${MEMORIZERS}  steps=${STEPS}"
echo "descents/instance = 1 + ${RESTARTS} + 2*${MEMORIZERS}"

# `uv run` uses the project's locked .venv (Python 3.11); no manual activation.
srun uv run python scripts/globalmin_gate.py \
    --suite        "${SUITE}" \
    --out          "${OUT_SUITE}" \
    --out-dir      "${OUT_DIR}" \
    --restarts     "${RESTARTS}" \
    --memorizers   "${MEMORIZERS}" \
    --steps        "${STEPS}" \
    --margin       "${MARGIN}" \
    --jobs         "${JOBS}"

echo "finished=$(date -Is)"
