#!/usr/bin/env python
"""Landscape diagnostics across the WHOLE suite -- population statistics.

`landscape_analysis.py` draws the detailed figures for one problem; this runs
the same rigorous plateau / saddle / local-optima metrics on every problem in
the suite (in parallel) and reports the distributions, so the paper can say
"across the N benchmark problems, X% ..." rather than showing one example.

    python scripts/landscape_sweep.py --jobs 10
    python scripts/landscape_sweep.py --restarts 24 --instances 1 2 3
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import numpy as np  # noqa: E402

from gaab import landscape as L  # noqa: E402
from gaab.parallel import map_unordered  # noqa: E402
from gaab.problem import GrokkingProblem, ProblemSpec  # noqa: E402
from gaab.suite import load_suite  # noqa: E402


def _worker(payload):
    spec_dict, instance, cfg = payload
    spec = ProblemSpec.from_dict(spec_dict)
    obj = GrokkingProblem(spec, instance=instance).objective
    row = {
        "problem": spec.name,
        "instance": instance,
        "p": spec.p,
        "rank": spec.rank,
        "reg_strength": spec.reg_strength,
        "train_frac": spec.train_frac,
    }
    row.update(L.characterize(obj, **cfg))
    return row


def plot(df, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gaab.plots import PALETTE, SURFACE, TEXT_MUTED, TEXT_PRIMARY, _style

    _style()
    blue, orange, aqua, violet = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[6]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    def hist(ax, values, color, xlabel, title, vline=None):
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        ax.hist(
            values,
            bins=min(20, max(5, len(values) // 3)),
            color=color,
            edgecolor=SURFACE,
            linewidth=0.6,
        )
        if vline is not None:
            ax.axvline(vline, color=TEXT_MUTED, lw=1.0, ls=(0, (4, 3)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("problems")
        ax.set_title(title, loc="left", fontsize=10.5, color=TEXT_PRIMARY)

    hist(
        axes[0, 0],
        df["frac_reached_global"],
        blue,
        "fraction of descents reaching the global basin",
        f"Local optima: median {df['frac_reached_global'].median():.0%} of restarts "
        f"reach the global\n({(df['frac_reached_global'] < 0.99).mean():.0%} of problems trap "
        "at least one descent)",
    )
    hist(
        axes[0, 1],
        df["n_distinct_minima"],
        violet,
        "distinct converged objective levels",
        f"Multimodality: median {int(df['n_distinct_minima'].median())} distinct basins "
        f"per problem\n(out of {int(df['n_descents'].median())} descents)",
    )
    hist(
        axes[1, 0],
        df["median_saddle_index"],
        orange,
        "median Hessian index at critical points",
        f"Saddles: every problem's critical points are saddles\n"
        f"(median index {df['median_saddle_index'].median():.2f} of directions curve down)",
    )
    hist(
        axes[1, 1],
        df["min_flat_directions"],
        aqua,
        "flat Hessian directions at the global minimum",
        f"Plateaus at the optimum: median {int(df['min_flat_directions'].median())} "
        "flat directions\n(a flat valley, not a sharp well)",
    )

    fig.suptitle(
        f"Landscape structure across the suite ({len(df)} problem instances)",
        x=0.01,
        ha="left",
        fontsize=13,
        color=TEXT_PRIMARY,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(argv=None):
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default=None)
    parser.add_argument(
        "--instances",
        type=int,
        nargs="+",
        default=[1],
        help="which instance(s) per spec to analyze (default: 1)",
    )
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--n-critical", type=int, default=8)
    parser.add_argument("--descent-steps", type=int, default=8000)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/landscape")
    args = parser.parse_args(argv)

    suite = load_suite(args.suite)
    cfg = dict(
        restarts=args.restarts,
        n_critical=args.n_critical,
        descent_steps=args.descent_steps,
        seed=args.seed,
    )
    payloads = [
        (spec.to_dict(), inst, cfg)
        for spec in suite.specs
        for inst in args.instances
        if inst in suite.instances_for(spec)
    ]
    print(
        f"Characterizing {len(payloads)} problem instances "
        f"({args.restarts} restarts, {args.n_critical} critical-point searches each)..."
    )

    import time

    rows, started = [], time.perf_counter()
    for done, row in enumerate(map_unordered(_worker, payloads, args.jobs), start=1):
        rows.append(row)
        if done % 5 == 0 or done == len(payloads):
            print(
                f"  [{done}/{len(payloads)}] ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["p", "n_variables", "reg_strength"])
    df.to_csv(os.path.join(args.out_dir, "landscape_sweep.csv"), index=False)
    plot(df, os.path.join(args.out_dir, "landscape_population.png"))

    print("\n=== landscape across the suite ===")
    print(f"  problems analyzed:                 {len(df)}")
    print(
        f"  have >1 distinct basin:            {(df['n_distinct_minima'] > 1).mean():.0%}"
    )
    print(
        f"  trap >=1 descent below the global: {(df['frac_reached_global'] < 0.99).mean():.0%}"
    )
    print(
        f"  median distinct basins / problem:  {int(df['n_distinct_minima'].median())}"
    )
    print(
        f"  every problem has saddle critical points: {(df['frac_saddles'] > 0).all()}"
    )
    print(
        f"  median saddle index:               {df['median_saddle_index'].median():.2f}"
    )
    print(
        f"  median flat directions at the min: {int(df['min_flat_directions'].median())}"
    )
    print(f"\nWrote {args.out_dir}/landscape_sweep.csv and landscape_population.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
