#!/usr/bin/env python
"""Show that the grokking problems have plateaus, saddle points, and local optima.

Because the objective ships exact gradients, this is a *rigorous* landscape
characterization (real Hessian eigen-spectra), not surrogate ELA features. For a
chosen problem it produces four figures and a metrics row:

    landscape_optima.png   converged objectives from many descents -> multiple basins
    landscape_saddles.png  Hessian index vs loss at critical points -> saddle structure
    landscape_plateau.png  a descent curve + 1D profiles -> flat stretches and barriers
    landscape_slice.png    a 2D slice of the loss around a solution
    landscape_summary.csv  one quantified row per analyzed problem

    python scripts/landscape_analysis.py                 # a default representative problem
    python scripts/landscape_analysis.py --problem p13 --restarts 32
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
from gaab.problem import GrokkingProblem  # noqa: E402
from gaab.suite import load_suite  # noqa: E402


def pick_problem(suite, needle):
    specs = suite.specs
    if needle:
        specs = [s for s in specs if needle in s.name]
        if not specs:
            raise SystemExit(f"No suite problem matches {needle!r}.")
    # Prefer a mid-size one -- big enough to be interesting, small enough to be quick.
    spec = sorted(specs, key=lambda s: abs(s.n_variables - 150))[0]
    return spec, suite.instances_for(spec)[0]


def analyze(obj, restarts, seed):
    out = {}

    # --- local optima: descend from many inits, collect basins -------------
    descents = L.restart_minima(obj, n_restarts=restarts, seed=seed, steps=12_000)
    finals = np.array([d.objective for d in descents])
    global_f = float(finals.min())
    reached_global = finals <= global_f + 1e-2 * max(abs(global_f), 1.0)
    out["descents"] = descents
    out["finals"] = finals
    out["n_distinct_minima"] = L.count_distinct(finals, rel_tol=1e-2)
    out["frac_reached_global"] = float(reached_global.mean())

    # --- saddles: critical points of any index, with Hessian spectra -------
    rng = np.random.default_rng(seed + 1)
    crit = []
    for _ in range(max(12, restarts // 2)):
        th0 = obj.initial_theta(seed=int(rng.integers(1 << 30)), scale=1.0)
        cp = L.find_critical_point(obj, th0, steps=150)
        if cp.grad_norm < 0.2:  # only keep genuine (near-)critical points
            sp = L.spectrum(L.hessian(obj, cp.theta))
            crit.append((cp.objective, sp))
    out["critical"] = crit
    if crit:
        out["frac_saddles"] = float(np.mean([sp.kind == "saddle" for _, sp in crit]))
        out["median_saddle_index"] = float(np.median([sp.index for _, sp in crit]))
    else:
        out["frac_saddles"] = out["median_saddle_index"] = float("nan")

    # curvature at the *minima* too, to contrast with the saddles
    best = min(descents, key=lambda d: d.objective)
    out["best"] = best
    out["min_spectrum"] = L.spectrum(L.hessian(obj, best.theta))

    # --- plateaus: a descent curve, and 1D profiles ------------------------
    out["long_descent"] = L.descend_to_minimum(
        obj, obj.initial_theta(seed=seed, scale=1.0), steps=30_000, sample_every=100
    )
    two = sorted(descents, key=lambda d: d.objective)
    # init -> best minimum (plateau along the path); worst -> best (a barrier)
    out["profile_to_min"] = L.line_profile(
        obj, obj.initial_theta(seed=seed, scale=1.0), best.theta, n=121
    )
    if len(two) >= 2 and two[-1].objective - two[0].objective > 1e-2:
        # worst basin -> best basin: a bump in between means they are separated.
        out["profile_barrier"] = L.line_profile(
            obj, two[-1].theta, two[0].theta, n=121, extend=0.1
        )
    else:
        out["profile_barrier"] = None

    # --- a 2D slice around the best solution -------------------------------
    out["slice"] = L.plane_slice(obj, best.theta, span=0.6, n=41, seed=seed)
    return out


def make_figures(obj, spec, res, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gaab.plots import (
        PALETTE,
        SURFACE,
        TEXT_MUTED,
        TEXT_PRIMARY,
        TEXT_SECONDARY,
        _style,
    )

    _style()
    blue, orange, aqua, red = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[7]
    title = spec.name.replace("GaaB_add_", "")

    # ---- local optima ----
    finals = np.sort(res["finals"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        np.arange(1, len(finals) + 1),
        finals,
        "o",
        color=blue,
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=0.8,
    )
    ax.axhline(finals.min(), color=aqua, lw=1.2, ls=(0, (4, 3)))
    ax.annotate(
        f"global basin (f={finals.min():.3f})",
        xy=(1, finals.min()),
        xytext=(4, 6),
        textcoords="offset points",
        color=TEXT_SECONDARY,
        fontsize=8,
    )
    ax.set_xlabel("descent (sorted by final objective)")
    ax.set_ylabel("converged objective  f(theta*)")
    ax.set_title(
        f"Local optima: {res['n_distinct_minima']} distinct basins over "
        f"{len(finals)} descents ({res['frac_reached_global']:.0%} reach the global one)",
        loc="left",
        fontsize=11,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "landscape_optima.png"), dpi=160)
    plt.close(fig)

    # ---- saddles (Dauphin plot) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    if res["critical"]:
        fs = np.array([f for f, _ in res["critical"]])
        idx = np.array([sp.index for _, sp in res["critical"]])
        ax.scatter(
            fs,
            idx,
            s=46,
            color=orange,
            edgecolor=SURFACE,
            linewidth=0.8,
            label="critical points",
            zorder=3,
        )
    m = res["min_spectrum"]
    ax.scatter(
        [res["best"].objective],
        [m.index],
        s=70,
        marker="*",
        color=aqua,
        edgecolor=SURFACE,
        linewidth=0.8,
        label="global minimum",
        zorder=4,
    )
    ax.set_xlabel("objective  f  at the critical point")
    ax.set_ylabel("Hessian index  (fraction of negative eigenvalues)")
    ax.set_ylim(-0.02, max(0.5, ax.get_ylim()[1]))
    ax.set_title(
        "Saddle structure: high-loss critical points have many\n"
        "negative-curvature directions; the minimum has ~none",
        loc="left",
        fontsize=10.5,
        color=TEXT_PRIMARY,
    )
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]  # eigen-spectrum: a saddle vs the minimum
    ax.axhline(0, color=TEXT_MUTED, lw=0.8)
    if res["critical"]:
        worst = max(res["critical"], key=lambda c: c[1].index)[1]
        ax.plot(
            np.sort(worst.eigenvalues),
            color=orange,
            lw=1.8,
            label=f"a saddle (index {worst.index:.2f})",
        )
    ax.plot(
        np.sort(m.eigenvalues),
        color=aqua,
        lw=1.8,
        label=f"global min (index {m.index:.2f})",
    )
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("Hessian eigenvalue")
    ax.set_title(
        "Eigen-spectra: negative eigenvalues = escape directions",
        loc="left",
        fontsize=10.5,
        color=TEXT_PRIMARY,
    )
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"{title}   (n={obj.n_variables})",
        x=0.01,
        ha="left",
        fontsize=12,
        color=TEXT_PRIMARY,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, "landscape_saddles.png"), dpi=160)
    plt.close(fig)

    # ---- plateaus: descent curve + line profiles ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    traj = res["long_descent"].trajectory
    ax.plot(traj[:, 0], traj[:, 1], color=blue, lw=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("objective  f")
    ax.set_title(
        "A descent plateaus before it drops\n(the flat stretch is a landscape plateau)",
        loc="left",
        fontsize=10.5,
        color=TEXT_PRIMARY,
    )

    ax = axes[1]
    al, val = res["profile_to_min"]
    ax.plot(al, val, color=blue, lw=1.8, label="init -> global min")
    if res["profile_barrier"] is not None:
        alb, valb = res["profile_barrier"]
        ax.plot(alb, valb, color=red, lw=1.8, label="min -> other min (barrier)")
    ax.axvspan(0, 1, color=TEXT_MUTED, alpha=0.06)
    ax.set_xlabel("interpolation  alpha  (endpoints at 0 and 1)")
    ax.set_ylabel("objective  f")
    ax.set_title(
        "1D profiles: flat segments are plateaus,\na bump between minima is a basin barrier",
        loc="left",
        fontsize=10.5,
        color=TEXT_PRIMARY,
    )
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"{title}   (n={obj.n_variables})",
        x=0.01,
        ha="left",
        fontsize=12,
        color=TEXT_PRIMARY,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, "landscape_plateau.png"), dpi=160)
    plt.close(fig)

    # ---- 2D slice ----
    ts, Z = res["slice"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    extent = [ts[0], ts[-1], ts[0], ts[-1]]
    im = ax.imshow(Z, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    ax.contour(ts, ts, Z, levels=12, colors="white", linewidths=0.4, alpha=0.5)
    ax.plot(0, 0, "*", color="white", markersize=13, markeredgecolor="black")
    fig.colorbar(im, ax=ax, label="objective  f")
    ax.set_xlabel("random direction 1")
    ax.set_ylabel("random direction 2")
    ax.set_title(
        f"2D slice of the loss around a solution\n{title} (n={obj.n_variables})",
        loc="left",
        fontsize=10.5,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "landscape_slice.png"), dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--problem", default=None, help="substring of a suite problem name"
    )
    parser.add_argument("--suite", default=None)
    parser.add_argument("--restarts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/landscape")
    args = parser.parse_args(argv)

    suite = load_suite(args.suite)
    spec, instance = pick_problem(suite, args.problem)
    obj = GrokkingProblem(spec, instance=instance).objective
    print(f"Analyzing {spec.name}  (n={obj.n_variables}, instance {instance})")

    res = analyze(obj, args.restarts, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    make_figures(obj, spec, res, args.out_dir)

    import pandas as pd

    row = {
        "problem": spec.name,
        "n_variables": obj.n_variables,
        "n_descents": len(res["finals"]),
        "n_distinct_minima": res["n_distinct_minima"],
        "frac_reached_global": round(res["frac_reached_global"], 3),
        "global_objective": round(float(res["finals"].min()), 4),
        "worst_basin_objective": round(float(res["finals"].max()), 4),
        "n_critical_points": len(res["critical"]),
        "frac_saddles": round(res["frac_saddles"], 3),
        "median_saddle_index": round(res["median_saddle_index"], 3),
        "min_index": round(res["min_spectrum"].index, 3),
        "min_flat_directions": res["min_spectrum"].n_flat,
    }
    pd.DataFrame([row]).to_csv(
        os.path.join(args.out_dir, "landscape_summary.csv"), index=False
    )

    print("\n=== landscape summary ===")
    for k, v in row.items():
        print(f"  {k:24s} {v}")
    print(f"\nFigures + summary in {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
