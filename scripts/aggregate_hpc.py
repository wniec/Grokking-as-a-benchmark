#!/usr/bin/env python
"""Stitch the per-algorithm HPC outputs into one comparison, and rank methods.

Each Slurm array task writes its own files (per algorithm); this collects them,
builds a per-optimizer leaderboard on each benchmark, and -- the point of running
both -- reports how the grokking-suite ranking compares to the BBOB ranking. A
low rank correlation is evidence the grokking benchmark measures something BBOB
does not.

    python scripts/aggregate_hpc.py --hpc-dir results/hpc
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np  # noqa: E402


def kendall_tau(a, b) -> float:
    """Kendall's tau-b between two rankings (small n; no scipy needed)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    concordant = discordant = ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    denom = np.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return float((concordant - discordant) / denom) if denom else float("nan")


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def load_gaab(hpc_dir):
    import pandas as pd

    files = sorted(glob.glob(os.path.join(hpc_dir, "gaab", "*", "runs.csv")))
    if not files:
        return None
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return runs.groupby("optimizer").agg(
        gaab_aocc_objective=("aocc_objective", "mean"),
        gaab_aocc_test_accuracy=("aocc_test_accuracy", "mean"),
        gaab_best_test_accuracy=("best_test_accuracy", "mean"),
        gaab_grokked_fraction=("grokked", "mean"),
        gaab_runs=("run", "count"),
    ).reset_index()


def load_bbob(hpc_dir):
    import pandas as pd

    files = sorted(glob.glob(os.path.join(hpc_dir, "bbob", "*.csv")))
    if not files:
        return None
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return runs.groupby("optimizer").agg(
        bbob_aocc_objective=("aocc_objective", "mean"),
        bbob_final_gap_median=("final_gap", "median"),
        bbob_runs=("optimizer", "count"),
    ).reset_index()


def main(argv=None) -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hpc-dir", default="results/hpc")
    parser.add_argument("--out", default=None, help="comparison CSV (default: <hpc-dir>/comparison.csv)")
    args = parser.parse_args(argv)

    gaab, bbob = load_gaab(args.hpc_dir), load_bbob(args.hpc_dir)
    if gaab is None and bbob is None:
        raise SystemExit(f"No results found under {args.hpc_dir}. Run the array job first.")

    if gaab is not None and bbob is not None:
        table = gaab.merge(bbob, on="optimizer", how="outer")
    else:
        table = gaab if gaab is not None else bbob

    sort_col = "gaab_aocc_objective" if "gaab_aocc_objective" in table else "bbob_aocc_objective"
    table = table.sort_values(sort_col, ascending=False).reset_index(drop=True)

    out = args.out or os.path.join(args.hpc_dir, "comparison.csv")
    table.to_csv(out, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.round(4).to_string(index=False))

    # The cross-benchmark question: do the two benchmarks agree on the ranking?
    if gaab is not None and bbob is not None:
        common = table.dropna(subset=["gaab_aocc_objective", "bbob_aocc_objective"])
        # Compare only the methods that ran on both (BBOB skips the gradient reference).
        common = common[common["optimizer"] != "adam"]
        if len(common) >= 3:
            tau = kendall_tau(common["gaab_aocc_objective"], common["bbob_aocc_objective"])
            rho = spearman(common["gaab_aocc_objective"], common["bbob_aocc_objective"])
            print(f"\nRanking agreement over {len(common)} black-box methods "
                  f"(gaab AOCC vs BBOB AOCC):")
            print(f"  Kendall tau  = {tau:+.3f}")
            print(f"  Spearman rho = {rho:+.3f}")
            print("  (low/negative => the grokking benchmark ranks optimizers "
                  "differently than BBOB -- evidence it measures something new)")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
