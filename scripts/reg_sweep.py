#!/usr/bin/env python
"""Characterize how grokking strength depends on the regularization factor.

Regularization here is a *direct factor of the loss* (`f = CE_train + reg * mean(theta^2)`),
so it is a landscape knob. This script fixes a handful of architectures that are
known to grok, then sweeps ``reg_strength`` finely -- including values below the
suite grid's floor and ``reg = 0`` as the no-pressure control -- and measures,
with the gradient reference, two facets of grokking strength:

* **grok rate** -- the fraction of (architecture, instance) pairs whose verdict
  is ``grokking``; how *reliably* the landscape groks.
* **delay ratio** ``t_gen / t_fit`` -- among the runs that grok, how much later
  generalization arrives than memorization; how *pronounced* the gap is.

The point is to locate both the peak and the low-end collapse: strong
regularization damps the transition, and ``reg -> 0`` removes the pressure
entirely, so the strongest grokking sits at some weak-but-nonzero value.

    python scripts/reg_sweep.py --steps 60000 --instances 5 --jobs 10
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np  # noqa: E402

from gaab.parallel import map_unordered  # noqa: E402
from gaab.problem import ProblemSpec  # noqa: E402
from gaab.screening import AdamSettings, screen_problem  # noqa: E402

#: Grokking-capable architectures (from the shipped suite), spanning p and rank.
ARCHITECTURES = [
    dict(p=11, embed_dim=2, hidden_dim=8, rank=4, rank_targets=("W1", "W2")),
    dict(p=11, embed_dim=2, hidden_dim=12, rank=4, rank_targets=("W1", "W2")),
    dict(p=13, embed_dim=2, hidden_dim=12, rank=4, rank_targets=("W1", "W2")),
    dict(p=13, embed_dim=3, hidden_dim=6, rank=6, rank_targets=("W1",)),
    dict(p=17, embed_dim=2, hidden_dim=8, rank=4, rank_targets=("W1", "W2")),
    dict(p=7, embed_dim=2, hidden_dim=12, rank=4, rank_targets=("W1", "W2")),
]

#: reg sweep: 0 (control) then a geometric ladder from 1e-5 up to 0.05.
REG_VALUES = [
    0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2,
]


def _worker(payload):
    arch, reg, instance, settings = payload
    spec = ProblemSpec(
        activation="quad", reg_strength=float(reg), train_frac=0.6, **arch
    )
    report = screen_problem(spec, instance=instance, settings=settings)
    return {
        "arch": spec.name.rsplit("_reg", 1)[0],
        "p": spec.p,
        "n_variables": spec.n_variables,
        "reg_strength": float(reg),
        "instance": instance,
        "verdict": report.verdict,
        "delay_ratio": report.delay_ratio,
        "t_fit": report.t_fit,
        "t_gen": report.t_gen,
        "best_test_accuracy": report.best_test_accuracy,
    }


def run(steps: int, instances: int, jobs: int):
    import pandas as pd

    settings = AdamSettings(steps=steps, eval_every=max(250, steps // 120))
    payloads = [
        (arch, reg, inst, settings)
        for arch in ARCHITECTURES
        for reg in REG_VALUES
        for inst in range(1, instances + 1)
    ]
    rows = []
    for done, row in enumerate(map_unordered(_worker, payloads, jobs), start=1):
        rows.append(row)
        if done % 20 == 0 or done == len(payloads):
            print(f"  [{done:>4}/{len(payloads)}]", flush=True)
    return pd.DataFrame(rows)


def summarize(frame):
    frame = frame.copy()
    frame["is_grok"] = frame["verdict"] == "grokking"
    grok = frame[frame["is_grok"]]
    summary = frame.groupby("reg_strength").agg(
        samples=("verdict", "size"),
        grok_rate=("is_grok", "mean"),
        mean_best_test=("best_test_accuracy", "mean"),
    )
    delay = grok.groupby("reg_strength")["delay_ratio"].median().rename("median_delay_ratio")
    tgen = grok.groupby("reg_strength")["t_gen"].median().rename("median_t_gen")
    summary = summary.join(delay).join(tgen).reset_index()
    return summary


def plot(summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gaab.plots import SURFACE, TEXT_MUTED, TEXT_PRIMARY, _style

    _style()
    reg = summary["reg_strength"].to_numpy()
    # reg = 0 cannot sit on a log axis; place it a decade below the smallest
    # positive value and mark the break.
    positive = reg[reg > 0]
    zero_x = positive.min() / 3.0
    x = np.where(reg > 0, reg, zero_x)

    fig, ax1 = plt.subplots(figsize=(8.6, 5.2))
    rate_color, delay_color = "#2a78d6", "#eb6834"

    ax1.plot(x, summary["grok_rate"], color=rate_color, marker="o", markersize=6,
             markeredgecolor=SURFACE, markeredgewidth=0.8, label="grok rate")
    ax1.set_xscale("log")
    ax1.set_xlabel("regularization strength (direct loss factor)")
    ax1.set_ylabel("grok rate  (fraction of runs that grok)", color=rate_color)
    ax1.tick_params(axis="y", labelcolor=rate_color)
    ax1.set_ylim(0, max(summary["grok_rate"].max() * 1.25, 0.05))
    ax1.axvline(zero_x, color=TEXT_MUTED, lw=0.9, ls=(0, (2, 3)))
    ax1.annotate("reg = 0\n(no pressure)", xy=(zero_x, ax1.get_ylim()[1]),
                 xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                 fontsize=7.5, color=TEXT_MUTED)

    ax2 = ax1.twinx()
    ax2.plot(x, summary["median_delay_ratio"], color=delay_color, marker="s",
             markersize=6, markeredgecolor=SURFACE, markeredgewidth=0.8,
             label="median delay ratio")
    ax2.set_ylabel("median delay ratio  t_gen / t_fit  (grokking runs)", color=delay_color)
    ax2.tick_params(axis="y", labelcolor=delay_color)
    ax2.grid(False)

    lines = ax1.get_lines()[:1] + ax2.get_lines()[:1]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper center")
    fig.suptitle("How grokking strength depends on regularization",
                 x=0.01, ha="left", fontsize=12, color=TEXT_PRIMARY, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--out-dir", default="results/reg_sweep")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Reg sweep: {len(ARCHITECTURES)} architectures x {len(REG_VALUES)} reg values "
          f"x {args.instances} instances x {args.steps} epochs")
    frame = run(args.steps, args.instances, args.jobs)
    frame.to_csv(os.path.join(args.out_dir, "runs.csv"), index=False)
    summary = summarize(frame)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    path = plot(summary, os.path.join(args.out_dir, "reg_sweep.png"))

    with __import__("pandas").option_context("display.width", 200):
        print("\n" + summary.round(3).to_string(index=False))
    peak_rate = summary.loc[summary["grok_rate"].idxmax(), "reg_strength"]
    peak_delay = summary.loc[summary["median_delay_ratio"].idxmax(), "reg_strength"]
    print(f"\nStrongest grok rate at reg = {peak_rate:g}; "
          f"largest delay ratio at reg = {peak_delay:g}")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
