#!/usr/bin/env python
"""Intrinsic dimension of the grokking descent trajectory, across the suite.

Runs Adam on every problem (over several seeds) and does PCA on the sequence of
parameter iterates.  The question: how many dimensions does the descent path
actually occupy, relative to the 72-199 parameters it moves through?  Writes a
per-run CSV and a figure, and prints the population summary the paper quotes
("the descent collapses onto a ~2D subspace").

    python scripts/trajectory_dim.py --jobs 10
    python scripts/trajectory_dim.py --seeds 0 1 2 3 4 --steps 8000

This complements `projection.py` (compress the *search space* to low dim and it
still groks) with the *dynamics* side (the uncompressed gradient path is already
low-dimensional).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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
    spec_dict, instance, seed, cfg = payload
    spec = ProblemSpec.from_dict(spec_dict)
    pr = GrokkingProblem(spec, instance=instance)
    obj = pr.objective
    theta0 = obj.initial_theta(seed=seed, scale=1.0)
    theta_star, loss_curve, iterates = L.descent_path(
        obj,
        theta0,
        steps=cfg["steps"],
        lr=cfg["lr"],
        sample_every=cfg["sample_every"],
    )
    ratios = L.trajectory_variance(iterates, center=theta_star)
    cum = np.cumsum(ratios)
    metrics = obj.metrics(theta_star)
    return {
        "problem": spec.name,
        "instance": instance,
        "seed": seed,
        "p": spec.p,
        "dim": pr.meta_data.n_variables,
        "reg_strength": spec.reg_strength,
        "train_frac": spec.train_frac,
        "n_iterates": len(iterates),
        "pc1": float(ratios[0]),
        "pc1_2": float(cum[1]) if len(cum) > 1 else float(cum[-1]),
        "pc1_5": float(cum[4]) if len(cum) > 4 else float(cum[-1]),
        "n90": int(np.searchsorted(cum, 0.90) + 1),
        "n95": int(np.searchsorted(cum, 0.95) + 1),
        "final_loss": float(loss_curve[-1, 1]),
        "test_accuracy": float(metrics.test_accuracy),
        "groked": bool(metrics.test_accuracy >= 0.80),
    }


def plot(df, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gaab.plots import PALETTE, SURFACE, TEXT_MUTED, TEXT_PRIMARY, _style

    _style()
    blue, violet = PALETTE[0], PALETTE[6]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    vals = df["pc1_2"].to_numpy() * 100
    ax.hist(vals, bins=20, color=blue, edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(np.median(vals), color=TEXT_MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.set_xlabel("variance captured by the top 2 PCs (%)")
    ax.set_ylabel("runs")
    ax.set_title(
        f"Descent path is ~2D: median {np.median(vals):.0f}% in 2 PCs",
        loc="left",
        fontsize=10.5,
        color=TEXT_PRIMARY,
    )

    ax = axes[1]
    ax.scatter(
        df["dim"],
        df["n95"],
        s=26,
        c=violet,
        edgecolor=SURFACE,
        linewidth=0.5,
        alpha=0.8,
    )
    corr = np.corrcoef(df["dim"], df["n95"])[0, 1]
    ax.set_xlabel("ambient dimension (parameter count)")
    ax.set_ylabel("# PCs for 95% of path variance")
    ax.set_title(
        f"...and it barely grows with size (corr {corr:.2f})",
        loc="left",
        fontsize=10.5,
        color=TEXT_PRIMARY,
    )

    fig.suptitle(
        f"Intrinsic dimension of the grokking descent ({len(df)} runs)",
        x=0.01,
        ha="left",
        fontsize=13,
        color=TEXT_PRIMARY,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
        default=None,
        help="instance(s) per spec (default: first available per spec)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--out-dir", default="results/trajectory_dim")
    args = parser.parse_args(argv)

    suite = load_suite(args.suite)
    cfg = dict(steps=args.steps, lr=args.lr, sample_every=args.sample_every)

    def spec_instances(spec):
        avail = list(suite.instances_for(spec))
        return (
            avail[:1]
            if args.instances is None
            else [i for i in args.instances if i in avail]
        )

    payloads = [
        (spec.to_dict(), inst, seed, cfg)
        for spec in suite.specs
        for inst in spec_instances(spec)
        for seed in args.seeds
    ]
    print(
        f"Tracing {len(payloads)} descents "
        f"({len(suite.specs)} specs x {len(args.seeds)} seeds, {args.steps} steps each)..."
    )

    rows, started = [], time.perf_counter()
    for done, row in enumerate(map_unordered(_worker, payloads, args.jobs), start=1):
        rows.append(row)
        if done % 25 == 0 or done == len(payloads):
            print(
                f"  [{done}/{len(payloads)}] ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["p", "dim", "problem", "seed"])
    df.to_csv(os.path.join(args.out_dir, "trajectory_dim.csv"), index=False)
    plot(df, os.path.join(args.out_dir, "trajectory_dim.png"))

    gk = df[df["groked"]]
    print("\n=== intrinsic dimension of the descent trajectory ===")
    print(
        f"  runs:                              {len(df)}  "
        f"(dims {df['dim'].min()}-{df['dim'].max()}, {df['groked'].mean():.0%} groked)"
    )
    print(f"  median variance in top 2 PCs:      {df['pc1_2'].median():.1%}")
    print(f"  median variance in top 5 PCs:      {df['pc1_5'].median():.1%}")
    print(
        f"  median #PCs for 90% / 95% var:     {int(df['n90'].median())} / {int(df['n95'].median())}"
    )
    print(
        f"  corr(ambient dim, #PCs for 95%):   {np.corrcoef(df['dim'], df['n95'])[0, 1]:.2f}"
    )
    if len(gk):
        print(
            f"  [grokked only] median top-2 var:   {gk['pc1_2'].median():.1%} "
            f"(#PCs 95%: {int(gk['n95'].median())})"
        )
    print(f"\nWrote {args.out_dir}/trajectory_dim.csv and trajectory_dim.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
