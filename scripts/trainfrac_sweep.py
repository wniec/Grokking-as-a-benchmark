#!/usr/bin/env python
"""Does the best train/test split for grokking depend on p and network size?

The training set is a ``train_frac`` fraction of the full ``p x p`` table, so
the *absolute* number of training pairs is ``train_frac * p^2`` -- and a small
prime has far fewer pairs to begin with. The hypothesis: lower ``p`` (and, to a
lesser degree, smaller networks) need a larger ``train_frac`` to have enough
data for a delayed transition to exist at all.

This sweeps ``train_frac`` for a grid of (p, network-size) cells, holding
regularization fixed at a reliably-grokking value, and reports the split that
maximizes the grok rate in each cell -- so the shift (if any) is visible.

    python scripts/trainfrac_sweep.py --steps 60000 --instances 4 --jobs 10
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

from gaab.parallel import map_unordered  # noqa: E402
from gaab.problem import ProblemSpec  # noqa: E402
from gaab.screening import AdamSettings, screen_problem  # noqa: E402

#: (label, shape) cells spanning small -> larger networks.  Each is screened at
#: every prime, so the cell's parameter count grows with p but the *shape* is
#: held fixed, isolating the size axis from p.
SHAPES = {
    "small": dict(embed_dim=2, hidden_dim=6, rank=3, rank_targets=("W1", "W2")),
    "mid": dict(embed_dim=2, hidden_dim=10, rank=4, rank_targets=("W1", "W2")),
    "large": dict(embed_dim=3, hidden_dim=8, rank=6, rank_targets=("W1",)),
}
PRIMES = (5, 7, 11, 13, 17)
TRAIN_FRACS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
REG = 0.01  # a reliably-grokking strength (see scripts/reg_sweep.py)
MAX_VARIABLES = 200


def _worker(payload):
    p, shape_label, shape, tf, instance, settings = payload
    spec = ProblemSpec(p=p, reg_strength=REG, train_frac=tf, activation="quad", **shape)
    report = screen_problem(spec, instance=instance, settings=settings)
    return {
        "p": p,
        "shape": shape_label,
        "n_variables": spec.n_variables,
        "train_pairs": int(round(tf * p * p)),
        "train_frac": tf,
        "instance": instance,
        "verdict": report.verdict,
        "is_grok": report.verdict == "grokking",
        "delay_ratio": report.delay_ratio,
        "best_test_accuracy": report.best_test_accuracy,
    }


def run(steps, instances, jobs):
    import pandas as pd

    settings = AdamSettings(steps=steps, eval_every=max(250, steps // 120))
    payloads = []
    for p in PRIMES:
        for label, shape in SHAPES.items():
            spec = ProblemSpec(
                p=p, reg_strength=REG, train_frac=0.6, activation="quad", **shape
            )
            if spec.n_variables > MAX_VARIABLES:
                continue
            for tf in TRAIN_FRACS:
                for inst in range(1, instances + 1):
                    payloads.append((p, label, shape, tf, inst, settings))
    rows = []
    for done, row in enumerate(map_unordered(_worker, payloads, jobs), start=1):
        rows.append(row)
        if done % 30 == 0 or done == len(payloads):
            print(f"  [{done:>4}/{len(payloads)}]", flush=True)
    return pd.DataFrame(rows)


def optimum_table(frame):
    """Best train_frac per (p, shape) cell, by grok rate then delay."""
    import pandas as pd

    cell = (
        frame.groupby(["p", "shape", "train_frac"])
        .agg(
            grok_rate=("is_grok", "mean"),
            median_delay=("delay_ratio", lambda s: np.nanmedian(s[np.isfinite(s)])),
        )
        .reset_index()
    )
    rows = []
    for (p, shape), g in cell.groupby(["p", "shape"]):
        g = g.sort_values(["grok_rate", "median_delay"], ascending=False)
        best = g.iloc[0]
        rows.append(
            {
                "p": p,
                "shape": shape,
                "best_train_frac": best["train_frac"],
                "best_grok_rate": round(float(best["grok_rate"]), 3),
                "grokked_at_any_frac": bool((g["grok_rate"] > 0).any()),
            }
        )
    return pd.DataFrame(rows), cell


def plot(cell, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gaab.plots import PALETTE, SURFACE, TEXT_PRIMARY, _style

    _style()
    primes = sorted(cell["p"].unique())
    colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(primes)}
    shapes = ["small", "mid", "large"]
    fig, axes = plt.subplots(
        1, len(shapes), figsize=(4.6 * len(shapes), 4.4), sharey=True
    )

    for ax, shape in zip(axes, shapes):
        sub = cell[cell["shape"] == shape]
        for p in primes:
            g = sub[sub["p"] == p].sort_values("train_frac")
            if not len(g):
                continue
            ax.plot(
                g["train_frac"],
                g["grok_rate"],
                color=colors[p],
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=0.7,
                label=f"p={p}",
            )
        ax.set_title(f"{shape} network", loc="left")
        ax.set_xlabel("train_frac")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("grok rate")
    axes[-1].legend(title="", loc="upper left", fontsize=8)
    fig.suptitle(
        "Best train/test split for grokking shifts with p and network size",
        x=0.01,
        ha="left",
        fontsize=12,
        color=TEXT_PRIMARY,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--out-dir", default="results/trainfrac_sweep")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"train_frac sweep: {len(PRIMES)} primes x {len(SHAPES)} shapes x "
        f"{len(TRAIN_FRACS)} fracs x {args.instances} instances at reg={REG}"
    )
    frame = run(args.steps, args.instances, args.jobs)
    frame.to_csv(os.path.join(args.out_dir, "runs.csv"), index=False)
    optima, cell = optimum_table(frame)
    optima.to_csv(os.path.join(args.out_dir, "optima.csv"), index=False)
    cell.to_csv(os.path.join(args.out_dir, "cells.csv"), index=False)
    path = plot(cell, os.path.join(args.out_dir, "trainfrac_sweep.png"))

    import pandas as pd

    with pd.option_context("display.width", 200):
        print("\nBest train_frac per (p, shape) cell:")
        print(optima.to_string(index=False))
        print("\nMean best train_frac by p (over shapes that grok):")
        gk = optima[optima["grokked_at_any_frac"]]
        print(gk.groupby("p")["best_train_frac"].mean().round(2).to_string())
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
