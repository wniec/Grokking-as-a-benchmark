#!/bin/bash
#SBATCH --job-name=gaab-bench-test
#SBATCH --array=0-2                 # 3 algorithms only
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=hpc/logs/%x_%A_%a.out
#SBATCH --error=hpc/logs/%x_%A_%a.err
#
# Small smoke test of the whole pipeline: 3 algorithms, tiny budget, a couple of
# BBOB functions. Use this first to confirm the environment and paths work
# before launching the full run_benchmarks.slurm.
#   sbatch hpc/run_benchmarks_test.slurm     (or: bash hpc/submit.sh --test)

set -euo pipefail
mkdir -p hpc/logs

ALGOS=(cma_es lm_cma_es sep_nes)
ALGO=${ALGOS[$SLURM_ARRAY_TASK_ID]}

# module load python/3.11
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "TEST task ${SLURM_ARRAY_TASK_ID}: algorithm=${ALGO}"

python scripts/run_one_algorithm.py \
    --algorithm "${ALGO}" \
    --benchmark both \
    --output-dir results/hpc_test \
    --jobs "${SLURM_CPUS_PER_TASK}" \
    --budget 5000 \
    --reps 2 \
    --bbob-fids 1 8 15 \
    --bbob-dims 5 \
    --bbob-instances 1 \
    --bbob-reps 2 \
    --bbob-budget-factor 500

echo "TEST task ${SLURM_ARRAY_TASK_ID} (${ALGO}) done."
