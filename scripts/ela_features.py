#!/usr/bin/env python
"""Exploratory Landscape Analysis (ELA) features across the suite -- and BBOB.

Computes the standard sample-based ELA feature vector (via pflacco) for every
problem in the GaaB suite, and, with ``--bbob``, for a panel of BBOB functions
at matching dimensions.  Writes one tidy CSV of features and a 2D projection
figure placing GaaB problems relative to BBOB in feature space -- the evidence
for "GaaB covers a region classical benchmarks do not".

    python scripts/ela_features.py --jobs 10
    python scripts/ela_features.py --bbob --n-per-dim 50 --jobs 12

ELA features need the optional dependency ``pflacco`` (``uv pip install pflacco``).
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

from gaab import ela as E  # noqa: E402
from gaab.parallel import map_unordered  # noqa: E402
from gaab.problem import GrokkingProblem, ProblemSpec  # noqa: E402
from gaab.suite import load_suite  # noqa: E402

# BBOB panel spanning the five function groups (separable, low/high-conditioning
# unimodal, multimodal with global structure, multimodal weak structure).
BBOB_PANEL = (1, 2, 6, 8, 10, 12, 15, 16, 20, 21, 23, 24)


def _gaab_worker(payload):
    spec_dict, instance, cfg = payload
    spec = ProblemSpec.from_dict(spec_dict)
    problem = GrokkingProblem(spec, instance=instance)
    row = {
        "family": "gaab",
        "problem": spec.name,
        "instance": instance,
        "dim": problem.meta_data.n_variables,
        "p": spec.p,
        "reg_strength": spec.reg_strength,
        "train_frac": spec.train_frac,
    }
    row.update(E.ela_features(problem, **cfg))
    return row


def _bbob_worker(payload):
    fid, dim, iid, cfg = payload
    problem = E.bbob_problem(fid, dim, iid)
    row = {
        "family": "bbob",
        "problem": f"bbob_f{fid}_d{dim}_i{iid}",
        "instance": iid,
        "dim": dim,
        "fid": fid,
    }
    row.update(E.ela_features(problem, **cfg))
    return row


def projection_plot(df, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from gaab.plots import PALETTE, SURFACE, TEXT_PRIMARY, _style

    feat_cols = [c for c in df.columns if "." in c and not c.endswith("error_msg")]
    M = df[feat_cols].apply(lambda s: s.astype(float))
    M = M.loc[:, M.notna().all() & (M.std() > 0)]  # drop NaN / constant cols
    Z = StandardScaler().fit_transform(M.to_numpy())
    pcs = PCA(n_components=2, random_state=0).fit(Z)
    XY = pcs.transform(Z)

    _style()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for family, color in (("bbob", PALETTE[4]), ("gaab", PALETTE[0])):
        mask = (df["family"] == family).to_numpy()
        ax.scatter(
            XY[mask, 0],
            XY[mask, 1],
            s=46,
            c=color,
            edgecolor=SURFACE,
            linewidth=0.6,
            alpha=0.85,
            label=f"{family} ({mask.sum()})",
        )
    ax.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title(
        "GaaB vs BBOB in ELA feature space",
        loc="left",
        fontsize=13,
        color=TEXT_PRIMARY,
        weight="bold",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return len(M.columns)


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
    parser.add_argument("--n-per-dim", type=int, default=E.DEFAULT_N_PER_DIM)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bbob",
        action="store_true",
        help="also compute ELA for a BBOB panel at matching dims",
    )
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--out-dir", default="results/ela")
    args = parser.parse_args(argv)

    cfg = dict(n_per_dim=args.n_per_dim, seed=args.seed, sets=E.DEFAULT_SETS)
    suite = load_suite(args.suite)

    def spec_instances(spec):
        avail = list(suite.instances_for(spec))
        if args.instances is None:  # one ELA point per spec: first available
            return avail[:1]
        return [i for i in args.instances if i in avail]

    payloads = [
        (_gaab_worker, (spec.to_dict(), inst, cfg))
        for spec in suite.specs
        for inst in spec_instances(spec)
    ]
    if args.bbob:
        # Compare at matched dimensions: run the BBOB panel at the low/median/high
        # dimensions present in the suite, so family differences in feature space
        # are not just a dimensionality artifact.
        gaab_dims = sorted(
            {
                GrokkingProblem(s, instance=spec_instances(s)[0]).meta_data.n_variables
                for s in suite.specs
                if spec_instances(s)
            }
        )
        bbob_dims = sorted(
            {gaab_dims[0], gaab_dims[len(gaab_dims) // 2], gaab_dims[-1]}
        )
        payloads += [
            (_bbob_worker, (fid, dim, 1, cfg))
            for fid in BBOB_PANEL
            for dim in bbob_dims
        ]

    print(
        f"Computing ELA features for {len(payloads)} problems "
        f"({args.n_per_dim}*d samples each)..."
    )
    rows, started = [], time.perf_counter()
    for done, row in enumerate(map_unordered(_dispatch, payloads, args.jobs), start=1):
        rows.append(row)
        if done % 5 == 0 or done == len(payloads):
            print(
                f"  [{done}/{len(payloads)}] ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["family", "dim", "problem"])
    csv_path = os.path.join(args.out_dir, "ela_features.csv")
    df.to_csv(csv_path, index=False)
    print(
        f"\nWrote {csv_path} ({len(df)} rows, "
        f"{sum('.' in c for c in df.columns)} feature columns)"
    )

    if args.bbob and (df["family"] == "bbob").any():
        fig_path = os.path.join(args.out_dir, "ela_projection.png")
        n_used = projection_plot(df, fig_path)
        print(f"Wrote {fig_path} (PCA over {n_used} usable features)")
    return 0


def _dispatch(job):
    fn, payload = job
    return fn(payload)


if __name__ == "__main__":
    sys.exit(main())
