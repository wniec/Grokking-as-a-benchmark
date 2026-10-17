#!/usr/bin/env python
"""Subsample a screened suite to a tractable, stratified benchmark set.

Raising ``train_frac`` for low ``p`` unlocked so many grokking configurations
(hundreds) that the raw screened suite is too large to benchmark. A benchmark
suite wants *diversity*, not exhaustiveness, so this keeps the single strongest
grokker in each ``(p, reg_strength)`` cell -- ranked by how reliably it groks
(``grokking_fraction``) and then by how pronounced the gap is
(``median_delay_ratio``) -- and caps the instances per problem.

    python scripts/curate_suite.py \
        --screened suites/grokking_full.json \
        --summary results/screening_summary.csv \
        --out suites/grokking_v1.json --per-cell 1 --max-instances 2
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gaab.problem import GrokkingSuite  # noqa: E402


def main(argv=None) -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--screened", required=True, help="the full screened suite JSON")
    parser.add_argument("--summary", required=True, help="screening_summary.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-cell", type=int, default=1,
                        help="problems to keep per (p, reg_strength) cell")
    parser.add_argument("--max-instances", type=int, default=2,
                        help="cap on kept instances per problem")
    args = parser.parse_args(argv)

    suite = GrokkingSuite.load(args.screened)
    by_name = {s.name: s for s in suite.specs}
    summary = pd.read_csv(args.summary)
    summary = summary[summary["kept"] & summary["problem"].isin(by_name)].copy()

    # Rank within each (p, reg) cell: reliability first, then gap size.
    summary = summary.sort_values(
        ["grokking_fraction", "median_delay_ratio"], ascending=False
    )
    chosen = (
        summary.groupby(["p", "reg_strength"], sort=True)
        .head(args.per_cell)["problem"]
        .tolist()
    )

    specs, spec_instances = [], {}
    for i, name in enumerate(chosen):
        spec = by_name[name].with_id(i + 1)
        specs.append(spec)
        instances = suite.instances_for(by_name[name])[: args.max_instances]
        spec_instances[spec.name] = instances

    curated = GrokkingSuite(specs, instances=list(suite.instances),
                            spec_instances=spec_instances)
    curated.save(args.out)

    n_inst = sum(len(v) for v in spec_instances.values())
    print(f"Curated {len(specs)} problems / {n_inst} instances "
          f"({args.per_cell} per (p, reg) cell) -> {args.out}")
    print("by p:  ", curated.table()["p"].value_counts().sort_index().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
