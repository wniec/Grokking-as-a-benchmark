# HPC sweep: every optimizer on GaaB **and** BBOB

This runs the whole optimizer roster on the curated grokking suite and on
IOHexperimenter's BBOB functions, one algorithm per Slurm array task, so you can
compare methods on both benchmarks and see whether the grokking ranking differs
from the BBOB ranking.

## Files

| file | what it is |
|---|---|
| `../scripts/run_one_algorithm.py` | runs **one** optimizer on gaab and/or BBOB (the array-task unit; runnable on its own) |
| `run_benchmarks.slurm` | Slurm **array** job — one task per algorithm, `--benchmark both` |
| `run_benchmarks_test.slurm` | 3-algorithm smoke test (tiny budget) |
| `submit.sh` | submit helper — `sbatch` on a cluster, or a local loop off-cluster |
| `../scripts/aggregate_hpc.py` | stitches all per-algorithm files → leaderboard + gaab-vs-BBOB rank correlation |

## Quick start

```bash
# 0. one algorithm, right now, no cluster (this is the unit of work):
python scripts/run_one_algorithm.py --algorithm cma_es --benchmark both

# 1. smoke-test the whole pipeline locally (3 algorithms, tiny budget):
bash hpc/submit.sh --test --local

# 2. on the cluster — smoke test, then the full run:
sbatch hpc/run_benchmarks_test.slurm        # or: bash hpc/submit.sh --test
sbatch hpc/run_benchmarks.slurm             # or: bash hpc/submit.sh

# 3. when the array finishes, aggregate + rank:
python scripts/aggregate_hpc.py --hpc-dir results/hpc
```

## Before you submit

Edit the two `#SBATCH` headers and the environment block in the `.slurm` files
for your cluster:

- **Environment.** Replace `source .venv/bin/activate` with your `module load
  python/3.11` / conda activation. The venv must have this package installed
  (`pip install -e .`); add `pip install cma` if you want the `pycma_es` baseline.
- **Resources.** `--cpus-per-task` is how many cores each algorithm gets to
  parallelize over problems (`--jobs "$SLURM_CPUS_PER_TASK"` is passed through);
  tune `--mem` and `--time` to your queue.
- **Array size.** `run_benchmarks.slurm` builds its algorithm list from
  `python scripts/run_one_algorithm.py --list` (currently 15 methods, including
  the optional Nevergrad ones), and `bash hpc/submit.sh` sets `--array=0-<N-1>`
  to match automatically. If you `sbatch` by hand, size the array yourself:
  `sbatch --array=0-$(($(python scripts/run_one_algorithm.py --list | wc -l)-1)) hpc/run_benchmarks.slurm`.

Each task writes only its own files, so tasks never collide:

```
results/hpc/gaab/<algo>/runs.csv     full grokking metrics (AOCC, ERT, grokked, ...)
results/hpc/bbob/<algo>.csv          one row per (function, dim, instance, rep)
results/hpc/comparison.csv           written by aggregate_hpc.py
```

## What you get

`aggregate_hpc.py` prints a per-optimizer leaderboard on each benchmark and the
**Kendall tau / Spearman** correlation between the gaab ranking and the BBOB
ranking. A low or negative correlation is the headline evidence that the grokking
suite measures an optimizer capability that BBOB does not — the empirical hook
for a "novel benchmark" paper.

## Notes

- `adam` (the gradient reference) runs on gaab only; it is skipped on BBOB.
- Optional methods — `pycma_es` (needs `cma`) and the Nevergrad `ng_*` methods
  (need `nevergrad`) — are skipped cleanly wherever their package is absent, so a
  missing optional dependency never fails an array task. `pip install nevergrad
  cma` on the cluster to include them.
- BBOB defaults: functions f1–f24, dims {5, 20, 40}, instances {1,2,3}, 5 reps,
  budget `2000 × dim`. Grokking defaults: the shipped 56-problem suite, budget
  60k, 15 reps. All are flags on `run_one_algorithm.py`.
