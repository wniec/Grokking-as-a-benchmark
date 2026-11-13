import json

import pandas as pd
import pytest

from gaab import cli
from gaab.problem import GrokkingSuite, ProblemSpec
from gaab.suite import PRESETS, candidate_specs, candidate_suite, load_suite


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_candidate_grids_are_unique_and_respect_their_ceiling(preset):
    ceiling = PRESETS[preset]["max_variables"]
    specs = candidate_specs(preset)
    names = [s.name for s in specs]
    assert len(names) == len(set(names))
    assert all(s.n_variables <= ceiling for s in specs)
    assert all(s.problem_id is not None for s in specs)


@pytest.mark.parametrize("preset", ["quick", "default", "full"])
def test_suite_presets_stay_under_200_parameters(preset):
    from gaab.suite import DEFAULT_MAX_VARIABLES

    assert DEFAULT_MAX_VARIABLES == 200
    assert all(s.n_variables <= 200 for s in candidate_specs(preset))


def test_default_grid_reaches_down_to_p_three():
    primes = {s.p for s in candidate_specs("default")}
    assert 3 in primes  # the floor the user asked for
    assert 2 not in primes  # Z_2 addition is a single bit


def test_candidate_grid_spans_the_documented_knobs():
    specs = candidate_specs("default")
    assert len({s.p for s in specs}) > 1
    assert len({s.rank for s in specs}) > 1
    assert len({s.reg_strength for s in specs}) > 1
    assert None in {s.rank for s in specs}


def test_max_variables_filters():
    assert all(
        s.n_variables <= 120 for s in candidate_specs("default", max_variables=120)
    )


def test_overrides_narrow_the_grid():
    specs = candidate_specs("quick", p=(11,), reg_strength=(1.0,))
    assert {s.p for s in specs} == {11}
    assert {s.reg_strength for s in specs} == {1.0}


def test_candidate_suite_is_iterable():
    suite = candidate_suite("quick", instances=(1,))
    first = next(iter(suite))
    assert first.n_variables == first.spec.n_variables


def test_shipped_suite_loads_and_only_contains_screened_problems():
    suite = load_suite()
    assert len(suite.specs) > 0
    assert len(suite) > 0
    for spec in suite.specs:
        assert suite.instances_for(spec), f"{spec.name} has no curated instances"
    problem = next(iter(suite))
    assert problem(list(problem.initial_theta())) > 0


def test_load_suite_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_suite(str(tmp_path / "nope.json"))


def test_cli_optimizers_lists_the_registry(capsys):
    assert cli.main(["optimizers"]) == 0
    out = capsys.readouterr().out
    assert "cma_es" in out and "gradient-based" in out
    assert "lm_cma_es" in out and "large-scale" in out


def test_cli_suite_describes_the_shipped_suite(capsys):
    assert cli.main(["suite"]) == 0
    out = capsys.readouterr().out
    assert "problem instances" in out and "n_variables" in out


def test_cli_run_and_analyze_end_to_end(tmp_path, capsys):
    suite_path = tmp_path / "suite.json"
    spec = ProblemSpec(
        p=7, embed_dim=4, hidden_dim=8, reg_strength=0.5, activation="quad"
    )
    GrokkingSuite([spec], instances=[1]).save(str(suite_path))

    out_dir = tmp_path / "bench"
    assert (
        cli.main(
            [
                "run",
                "--optimizer",
                "random_search",
                "local_random_search",
                "--budget",
                "150",
                "--reps",
                "1",
                "--suite",
                str(suite_path),
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    runs = pd.read_csv(out_dir / "runs.csv")
    assert len(runs) == 2

    assert cli.main(["analyze", "--results", str(out_dir)]) == 0
    figures = out_dir / "figures"
    for name in (
        "convergence.png",
        "generalization.png",
        "grokking_gap.png",
        "aocc.png",
        "ecdf.png",
    ):
        assert (figures / name).exists(), name
    assert (out_dir / "ert.csv").exists()


def test_cli_run_filters_problems(tmp_path):
    specs = [
        ProblemSpec(p=7, embed_dim=4, hidden_dim=8),
        ProblemSpec(p=11, embed_dim=4, hidden_dim=8),
    ]
    suite_path = tmp_path / "suite.json"
    GrokkingSuite(specs, instances=[1]).save(str(suite_path))
    out_dir = tmp_path / "bench"
    cli.main(
        [
            "run",
            "--optimizer",
            "random_search",
            "--budget",
            "100",
            "--reps",
            "1",
            "--suite",
            str(suite_path),
            "--output-dir",
            str(out_dir),
            "--problems",
            "p11",
        ]
    )
    runs = pd.read_csv(out_dir / "runs.csv")
    assert set(runs["p"]) == {11}


def test_cli_screen_reports_when_nothing_groks(tmp_path, capsys):
    # 300 Adam steps is far too short for any candidate to grok, so the command
    # must say so rather than write an empty suite.
    out = tmp_path / "suite.json"
    assert (
        cli.main(
            [
                "screen",
                "--preset",
                "quick",
                "--instances",
                "1",
                "--steps",
                "300",
                "--eval-every",
                "100",
                "--out",
                str(out),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert "No candidate problem grokked" in capsys.readouterr().out
    assert not out.exists()
    assert (tmp_path / "screening.csv").exists()
    assert (tmp_path / "screening_summary.csv").exists()
    reports = pd.read_csv(tmp_path / "screening.csv")
    assert {"verdict", "delay_ratio", "t_fit", "t_gen"} <= set(reports.columns)


@pytest.mark.slow
def test_cli_screen_writes_a_curated_suite(tmp_path):
    """End-to-end curation: the quick grid does contain problems that grok."""
    out = tmp_path / "suite.json"
    assert (
        cli.main(
            [
                "screen",
                "--preset",
                "quick",
                "--instances",
                "1",
                "2",
                "3",
                "--steps",
                "20000",
                "--eval-every",
                "250",
                "--jobs",
                "6",
                "--out",
                str(out),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text())
    assert payload["specs"] and "spec_instances" in payload
    suite = load_suite(str(out))
    assert all(suite.instances_for(s) for s in suite.specs)
    reports = pd.read_csv(tmp_path / "screening.csv")
    kept = {s.name for s in suite.specs}
    grokked = set(reports.loc[reports["verdict"] == "grokking", "problem"])
    assert kept <= grokked


def test_grids_sweep_the_weak_regularization_regime():
    # The reg axis is a fine weak-regularization sweep: >= 10 values, reaching
    # below 1e-4 and capped at 0.05 (strong regularization over-damps and
    # screens out; see scripts/reg_sweep.py).
    for preset in PRESETS:
        reg = sorted(PRESETS[preset]["reg_strength"])
        assert len(reg) >= 10
        assert min(reg) <= 1e-4
        assert max(reg) <= 0.05
        spec_regs = {s.reg_strength for s in candidate_specs(preset)}
        assert min(spec_regs) <= 1e-4 and max(spec_regs) <= 0.05


def test_default_grid_uses_a_higher_train_frac_for_lower_p():
    # trainfrac_sweep.py: lower p needs more training data to grok, so the
    # default schedule gives small primes a larger split than large primes.
    by_p = {}
    for s in candidate_specs("default"):
        by_p.setdefault(s.p, set()).add(s.train_frac)
    assert by_p[3] == {0.8} and by_p[5] == {0.8} and by_p[7] == {0.8}
    assert by_p[11] == {0.7}
    assert 0.8 in by_p[13]  # small nets at p=13 still want the high split
    # The lowest primes never use the plain 0.6 that the old fixed grid did.
    assert 0.6 not in by_p[3] and 0.6 not in by_p[7]


def test_candidate_specs_accepts_a_flat_train_frac_override():
    # A flat sequence still overrides the per-p schedule uniformly.
    specs = candidate_specs("default", train_frac=(0.8,))
    assert {s.train_frac for s in specs} == {0.8}
