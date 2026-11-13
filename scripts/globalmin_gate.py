#!/usr/bin/env python
"""Gate a curated suite on the global-minimum criterion.

Screening keeps problems where grokking is *reachable*.  This stronger gate
keeps only problems where the global minimum of ``f = CE_train + reg*mean(w^2)``
is the *generalizing* solution -- the only case in which driving ``f`` down is
the same thing as generalizing, and therefore the only case a grokking
benchmark should score optimizers on.

For every ``(spec, instance)`` in the input suite it approximates the global
minimum by the lowest ``f`` found over a pool of candidate minima (a geometric
ladder of random restarts plus explicitly constructed min-norm memorizers),
then keeps the pair iff that lowest minimum generalizes.

    python scripts/globalmin_gate.py \
        --suite suites/grokking_v1.json \
        --out suites/grokking_v1_globalmin.json \
        --restarts 64 --steps 60000 --jobs 10

Writes ``<out-dir>/globalmin.csv`` (one row per instance) and
``<out-dir>/globalmin_summary.csv`` (one row per spec) alongside the suite.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from gaab.globalmin import GateSettings, filter_by_globalmin, gate_specs  # noqa: E402
from gaab.problem import GrokkingSuite  # noqa: E402
from gaab.screening import AdamSettings  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", required=True, help="curated suite JSON to gate")
    parser.add_argument("--out", required=True, help="path for the gated suite JSON")
    parser.add_argument(
        "--out-dir",
        default="results",
        help="where to write the gate CSVs (default: results/)",
    )
    parser.add_argument(
        "--restarts", type=int, default=64, help="random restarts per (spec, instance)"
    )
    parser.add_argument(
        "--memorizers",
        type=int,
        default=4,
        help="explicitly constructed memorizers per (spec, instance)",
    )
    parser.add_argument(
        "--steps", type=int, default=60_000, help="Adam steps per descent"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="how much lower in f the generalizer must sit than any memorizer",
    )
    parser.add_argument("--test-threshold", type=float, default=0.95)
    parser.add_argument("--train-threshold", type=float, default=0.99)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)

    suite = GrokkingSuite.load(args.suite)
    settings = AdamSettings(steps=args.steps)
    gate = GateSettings(
        n_restarts=args.restarts,
        n_memorizers=args.memorizers,
        margin=args.margin,
        train_threshold=args.train_threshold,
        test_threshold=args.test_threshold,
    )

    n_pairs = len(suite)
    per_pair = 1 + args.restarts + 2 * args.memorizers
    print(
        f"Gating {len(suite.specs)} specs / {n_pairs} instances "
        f"(1 reference + {args.restarts} restarts + {args.memorizers} memorizers = "
        f"~{per_pair} descents each, {n_pairs * per_pair} descents total).",
        flush=True,
    )

    reports = gate_specs(
        suite.specs,
        spec_instances=suite.spec_instances,
        instances=suite.instances,
        settings=settings,
        gate=gate,
        jobs=args.jobs,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    reports.to_csv(os.path.join(args.out_dir, "globalmin.csv"), index=False)
    kept, spec_instances, summary = filter_by_globalmin(suite, reports)
    summary.to_csv(os.path.join(args.out_dir, "globalmin_summary.csv"), index=False)

    print("\nverdicts (per instance):")
    print(reports["verdict"].value_counts().to_string())

    if not kept:
        print(
            "\nNo problem's global minimum was the generalizing one; "
            "nothing written. Inspect globalmin.csv before relaxing --margin."
        )
        return 1

    gated = GrokkingSuite(
        [s.with_id(i + 1) for i, s in enumerate(kept)],
        instances=list(suite.instances),
        spec_instances={s.name: spec_instances[s.name] for s in kept},
    )
    gated.save(args.out)

    n_inst = sum(len(v) for v in spec_instances.values())
    dropped = len(suite.specs) - len(kept)
    print(
        f"\nKept {len(kept)}/{len(suite.specs)} specs ({n_inst} instances); "
        f"dropped {dropped} whose global minimum was not the generalizing one "
        f"-> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
