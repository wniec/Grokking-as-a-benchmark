"""Command-line entry points: ``python -m gaab.cli <command>``.

screen      run the gradient reference over a candidate grid and write a
            curated suite containing only the problems that grok
run         benchmark one or more optimizers on a suite, writing CSVs
analyze     turn a results directory into tables and figures
suite       describe a curated suite
optimizers  list the registered optimizers
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence


def _add_suite_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--suite",
        default=None,
        help="Path to a curated suite JSON (default: shipped suite).",
    )
    parser.add_argument(
        "--instances",
        type=int,
        nargs="+",
        default=None,
        help="Override the instance ids to use.",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=None,
        help="Only run problems whose name contains one of these substrings.",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=None,
        help="Skip problems with more than this many parameters.",
    )


def _load_suite(args):
    from gaab.suite import load_suite

    suite = load_suite(args.suite, instances=args.instances)
    specs = suite.specs
    if args.problems:
        specs = [s for s in specs if any(tok in s.name for tok in args.problems)]
    if args.max_dimension:
        specs = [s for s in specs if s.n_variables <= args.max_dimension]
    if not specs:
        raise SystemExit("No problems left after filtering.")
    suite.specs = specs
    return suite


# ---------------------------------------------------------------------------
def cmd_screen(args) -> int:
    from gaab.problem import GrokkingSuite
    from gaab.screening import AdamSettings, filter_specs, screen_specs
    from gaab.suite import candidate_specs

    specs = candidate_specs(args.preset, max_variables=args.max_dimension)
    dims = sorted(s.n_variables for s in specs)
    print(
        f"Screening {len(specs)} candidate problems x {len(args.instances)} instances "
        f"with {args.steps} Adam steps each.\n"
        f"Dimensions {dims[0]}-{dims[-1]} (median {dims[len(dims) // 2]}).\n"
    )

    settings = AdamSettings(steps=args.steps, lr=args.lr, eval_every=args.eval_every)
    reports = screen_specs(
        specs,
        instances=args.instances,
        settings=settings,
        train_threshold=args.train_threshold,
        test_threshold=args.test_threshold,
        early_factor=args.early_factor,
        jobs=args.jobs,
    )
    kept, spec_instances, summary = filter_specs(
        specs,
        reports,
        min_fraction=args.min_fraction,
        instance_level=not args.keep_all_instances,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    reports.to_csv(os.path.join(args.output_dir, "screening.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "screening_summary.csv"), index=False)

    print("\nVerdicts (modal, per problem):")
    print(summary["modal_verdict"].value_counts().to_string())

    if not kept:
        print(
            "\nNo candidate problem grokked, so no suite was written. "
            "Try more --steps, a lower --early-factor, or a wider --preset; "
            f"the per-run verdicts are in {args.output_dir}/screening.csv."
        )
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    suite = GrokkingSuite(
        [s.with_id(i + 1) for i, s in enumerate(kept)],
        instances=list(args.instances),
        spec_instances=spec_instances,
    )
    suite.save(args.out)

    print(
        f"\nKept {len(kept)}/{len(specs)} problems ({len(suite)} problem instances) -> {args.out}"
    )

    if args.plots:
        from gaab.plots import plot_screening

        path = plot_screening(summary, os.path.join(args.output_dir, "screening.png"))
        print(f"Wrote {path}")
    return 0


def cmd_run(args) -> int:
    from gaab.runner import RunConfig, run_suite

    suite = _load_suite(args)
    config = RunConfig(
        optimizers=args.optimizer,
        budget=args.budget,
        repetitions=args.reps,
        instances=args.instances,
        seed=args.seed,
        x0_mode=args.x0,
        ioh_logs=args.ioh_logs,
        jobs=args.jobs,
    )
    per_spec = (
        [len(args.instances)] * len(suite.specs)
        if args.instances
        else [len(suite.instances_for(s)) for s in suite.specs]
    )
    n_runs = sum(per_spec) * len(args.optimizer) * args.reps
    print(
        f"Suite: {len(suite.specs)} problems, dimensions "
        f"{min(s.n_variables for s in suite.specs)}-{max(s.n_variables for s in suite.specs)}\n"
        f"Optimizers: {', '.join(args.optimizer)}\n"
        f"{n_runs} runs x {args.budget} evaluations\n"
    )
    runs, curves = run_suite(suite, config, output_dir=args.output_dir)

    from gaab.analysis import summary_table

    print("\n" + summary_table(runs).to_string(index=False))
    if args.plots:
        cmd_analyze(argparse.Namespace(results=args.output_dir, out=None, target=None))
    return 0


def _landscape_worker(payload):
    spec_dict, instance, out_path = payload
    from gaab.landscape import problem_figure
    from gaab.problem import GrokkingProblem, ProblemSpec

    spec = ProblemSpec.from_dict(spec_dict)
    obj = GrokkingProblem(spec, instance=instance).objective
    title = spec.name.replace("GaaB_add_", "") + f"  (n={spec.n_variables})"
    return problem_figure(obj, title, out_path)


def _render_landscape_figures(runs, out_dir, suite_path, jobs):
    """One per-problem landscape picture for every benchmarked problem."""
    from gaab.parallel import map_unordered
    from gaab.suite import load_suite

    try:
        suite = load_suite(suite_path)
    except FileNotFoundError:
        print("\n(skipping landscape figures: no suite found; pass --suite)")
        return []
    by_name = {s.name: s for s in suite.specs}
    names = [n for n in runs["problem"].unique() if n in by_name]
    if not names:
        return []

    land_dir = os.path.join(out_dir, "landscape")
    os.makedirs(land_dir, exist_ok=True)
    payloads = []
    for name in names:
        spec = by_name[name]
        instance = suite.instances_for(spec)[0]
        payloads.append(
            (spec.to_dict(), instance, os.path.join(land_dir, f"{name}.png"))
        )

    print(
        f"\nRendering {len(payloads)} per-problem landscape figures -> {land_dir}/ ..."
    )
    paths = list(map_unordered(_landscape_worker, payloads, jobs))
    print(f"  wrote {len(paths)} landscape figures")
    return paths


def cmd_analyze(args) -> int:
    import pandas as pd

    from gaab.analysis import ert_table, summary_table
    from gaab.plots import make_all

    results = args.results
    runs = pd.read_csv(os.path.join(results, "runs.csv"))
    curves = pd.read_csv(os.path.join(results, "curves.csv"))

    ert_path = os.path.join(results, "ert.csv")
    ert_df = pd.read_csv(ert_path) if os.path.exists(ert_path) else ert_table(runs)
    ert_df.to_csv(ert_path, index=False)
    summary = summary_table(runs)
    summary.to_csv(os.path.join(results, "summary.csv"), index=False)

    screening_summary = None
    for candidate in (
        os.path.join(results, "screening_summary.csv"),
        os.path.join(os.path.dirname(results), "screening_summary.csv"),
        "results/screening_summary.csv",
    ):
        if os.path.exists(candidate):
            screening_summary = pd.read_csv(candidate)
            break

    out_dir = args.out or os.path.join(results, "figures")
    paths = make_all(
        runs,
        curves,
        ert_df,
        out_dir,
        screening_summary=screening_summary,
        ert_target=getattr(args, "target", None),
    )

    if getattr(args, "landscape", True):
        paths += _render_landscape_figures(
            runs, out_dir, args.suite, getattr(args, "landscape_jobs", 4)
        )

    print(summary.to_string(index=False))
    print("\nERT to the hardest reached generalization target:")
    finite = ert_df[ert_df["ert"] < float("inf")]
    if len(finite):
        target = finite["target_test_accuracy"].max()
        view = ert_df[ert_df["target_test_accuracy"] == target]
        print(view.sort_values("ert").head(20).to_string(index=False))
    else:
        print("  (no run reached any generalization target)")
    print("\nFigures:")
    for path in paths:
        print(f"  {path}")
    return 0


def cmd_suite(args) -> int:
    import pandas as pd

    suite = _load_suite(args)
    table = suite.table()
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(
            table[
                [
                    "name",
                    "n_variables",
                    "p",
                    "rank",
                    "reg_strength",
                    "train_frac",
                    "instances",
                ]
            ].to_string(index=False)
        )
    print(f"\n{len(suite.specs)} problems, {len(suite)} problem instances.")
    return 0


def cmd_optimizers(args) -> int:
    from gaab.optimizers import GRADIENT_BASED, LARGE_SCALE, OPTIMIZERS

    for name, fn in sorted(OPTIMIZERS.items()):
        tags = []
        if name in GRADIENT_BASED:
            tags.append("gradient-based")
        if name in LARGE_SCALE:
            tags.append("large-scale")
        suffix = f"  [{', '.join(tags)}]" if tags else ""
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"{name:<22}{doc}{suffix}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from gaab.optimizers import DEFAULT_BENCHMARK_OPTIMIZERS

    parser = argparse.ArgumentParser(prog="gaab", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("screen", help="curate a suite of problems that actually grok")
    s.add_argument(
        "--preset", default="default", help="candidate grid: quick | default | full"
    )
    s.add_argument("--instances", type=int, nargs="+", default=[1, 2, 3])
    s.add_argument(
        "--steps", type=int, default=60_000, help="Adam steps per screening run"
    )
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--eval-every", type=int, default=250)
    s.add_argument("--train-threshold", type=float, default=0.99)
    s.add_argument("--test-threshold", type=float, default=0.95)
    s.add_argument(
        "--early-factor",
        type=float,
        default=2.0,
        help="minimum t_gen / t_fit ratio to count as grokking",
    )
    s.add_argument(
        "--min-fraction",
        type=float,
        default=1 / 3,
        help="fraction of instances that must grok to keep a problem",
    )
    s.add_argument(
        "--keep-all-instances",
        action="store_true",
        help="keep every instance of a surviving problem, not just the grokking ones",
    )
    s.add_argument(
        "--max-dimension",
        type=int,
        default=None,
        help="dimension ceiling (default: the preset's own)",
    )
    s.add_argument("--jobs", type=int, default=1)
    s.add_argument("--out", default="suites/grokking_v1.json")
    s.add_argument("--output-dir", default="results")
    s.add_argument("--plots", action="store_true")
    s.set_defaults(func=cmd_screen)

    r = sub.add_parser("run", help="benchmark optimizers on a curated suite")
    r.add_argument(
        "--optimizer",
        nargs="+",
        default=list(DEFAULT_BENCHMARK_OPTIMIZERS),
        help="optimizers to run (default: the experiment roster, without de/random_search)",
    )
    r.add_argument("--budget", type=int, default=60_000)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument(
        "--x0",
        default="init",
        choices=["init", "uniform", "zeros"],
        help="starting point: the network's own init, uniform in box, or zeros",
    )
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument(
        "--ioh-logs",
        action="store_true",
        help="also write IOHprofiler-format logs for IOHanalyzer",
    )
    r.add_argument("--output-dir", default="results/benchmark")
    r.add_argument("--plots", action="store_true")
    _add_suite_args(r)
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("analyze", help="tables and figures from a results directory")
    a.add_argument("--results", default="results/benchmark")
    a.add_argument(
        "--out", default=None, help="figure directory (default: <results>/figures)"
    )
    a.add_argument(
        "--target",
        type=float,
        default=None,
        help="test-accuracy target for the ERT figure (default: hardest one reached)",
    )
    a.add_argument(
        "--suite",
        default=None,
        help="suite JSON for the per-problem landscape figures (default: shipped)",
    )
    a.add_argument(
        "--landscape",
        dest="landscape",
        action="store_true",
        default=True,
        help="render a landscape figure per problem (default: on)",
    )
    a.add_argument(
        "--no-landscape",
        dest="landscape",
        action="store_false",
        help="skip the per-problem landscape figures",
    )
    a.add_argument("--landscape-jobs", type=int, default=4)
    a.set_defaults(func=cmd_analyze)

    d = sub.add_parser("suite", help="describe a curated suite")
    _add_suite_args(d)
    d.set_defaults(func=cmd_suite)

    o = sub.add_parser("optimizers", help="list registered optimizers")
    o.set_defaults(func=cmd_optimizers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
