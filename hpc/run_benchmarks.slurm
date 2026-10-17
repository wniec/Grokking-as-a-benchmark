#!/bin/bash
#SBATCH --job-name=gaab-bench
#SBATCH --array=0-14                # one array task per algorithm (see `--list`); submit.sh sizes this for you
#SBATCH --cpus-per-task=16          # each task parallelizes over problems on its node
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/%x_%A_%a.out
#SBATCH --error=hpc/logs/%x_%A_%a.err
#
# Run one optimizer (this array index) on BOTH the grokking suite and BBOB.
# Submit with:  sbatch hpc/run_benchmarks.slurm
# or via the helper:  bash hpc/submit.sh
#
# The `#SBATCH --array` above must cover every algorithm. `bash hpc/submit.sh`
# sets the range automatically from the list below; if you run `sbatch` by hand,
# set `--array=0-$(($(python scripts/run_one_algorithm.py --list | wc -l)-1))`.

set -euo pipefail
mkdir -p hpc/logs

# --- environment ------------------------------------------------------------
# Adjust to your cluster: load a Python module or activate your environment.
# module load python/3.11
source .venv/bin/activate

# --- one algorithm per array index (single source of truth: the registry) ---
mapfile -t ALGOS < <(python scripts/run_one_algorithm.py --list)
ALGO=${ALGOS[$SLURM_ARRAY_TASK_ID]}

# One BLAS thread per worker; the script fans out over $SLURM_CPUS_PER_TASK cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# --- run --------------------------------------------------------------------
echo "Task ${SLURM_ARRAY_TASK_ID}: algorithm=${ALGO} on $(hostname), ${SLURM_CPUS_PER_TASK} cpus"

python scripts/run_one_algorithm.py \
    --algorithm "${ALGO}" \
    --benchmark both \
    --output-dir results/hpc \
    --jobs "${SLURM_CPUS_PER_TASK}" \
    --budget 60000 \
    --reps 15 \
    --bbob-fids 1-24 \
    --bbob-dims 5 20 40 \
    --bbob-instances 1 2 3 \
    --bbob-reps 5 \
    --bbob-budget-factor 2000

echo "Task ${SLURM_ARRAY_TASK_ID} (${ALGO}) done."
