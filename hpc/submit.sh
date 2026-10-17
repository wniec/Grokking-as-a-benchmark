#!/bin/bash
# Submit the GaaB-vs-BBOB optimizer sweep.
#
#   bash hpc/submit.sh            # submit the full array job (sbatch)
#   bash hpc/submit.sh --test     # submit the 3-algorithm smoke test
#   bash hpc/submit.sh --local    # run locally, no scheduler (loops over algorithms)
#   bash hpc/submit.sh --test --local
#
# On a Slurm cluster it calls sbatch; with --local (or when sbatch is missing)
# it runs the same work sequentially on this machine -- handy for a laptop check.

set -euo pipefail
cd "$(dirname "$0")/.."      # repo root

TEST=0; LOCAL=0
for arg in "$@"; do
    case "$arg" in
        --test) TEST=1 ;;
        --local) LOCAL=1 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

if [[ "$TEST" == 1 ]]; then
    SLURM=hpc/run_benchmarks_test.slurm
    ALGOS=(cma_es lm_cma_es sep_nes)
    OUT=results/hpc_test
    ARGS=(--budget 5000 --reps 2 --bbob-fids 1 8 15 --bbob-dims 5 --bbob-instances 1 --bbob-reps 2 --bbob-budget-factor 500)
else
    SLURM=hpc/run_benchmarks.slurm
    mapfile -t ALGOS < <(python scripts/run_one_algorithm.py --list)
    OUT=results/hpc
    ARGS=(--budget 60000 --reps 15 --bbob-fids 1-24 --bbob-dims 5 20 40 --bbob-instances 1 2 3 --bbob-reps 5 --bbob-budget-factor 2000)
fi

if [[ "$LOCAL" == 0 ]] && command -v sbatch >/dev/null 2>&1; then
    N=${#ALGOS[@]}
    echo "Submitting ${SLURM} (${N} algorithms, array 0-$((N-1))) via sbatch..."
    sbatch --array=0-$((N-1)) "${SLURM}"
    echo "When all tasks finish:  python scripts/aggregate_hpc.py --hpc-dir ${OUT}"
else
    [[ "$LOCAL" == 0 ]] && echo "sbatch not found -- running locally instead."
    JOBS=$(python -c "import os;print(os.cpu_count() or 4)")
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    for algo in "${ALGOS[@]}"; do
        echo "=== ${algo} ==="
        python scripts/run_one_algorithm.py --algorithm "${algo}" --benchmark both \
            --output-dir "${OUT}" --jobs "${JOBS}" "${ARGS[@]}"
    done
    echo "Aggregating..."
    python scripts/aggregate_hpc.py --hpc-dir "${OUT}"
fi
