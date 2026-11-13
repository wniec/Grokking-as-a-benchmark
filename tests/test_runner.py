import numpy as np
import pandas as pd
import pytest

from gaab.problem import GrokkingProblem, GrokkingSuite, ProblemSpec
from gaab.runner import RunConfig, _run_seed, run_single, run_suite

SPEC = ProblemSpec(p=7, embed_dim=4, hidden_dim=8, reg_strength=0.5, activation="quad")
CONFIG = RunConfig(
    optimizers=["local_random_search"], budget=200, repetitions=2, curve_points=20
)


def test_run_single_returns_a_row_and_a_curve():
    row, curve = run_single(SPEC, 1, "local_random_search", CONFIG, repetition=0)
    assert row["problem"] == SPEC.name and row["evaluations"] == CONFIG.budget
    assert 0.0 <= row["aocc_objective"] <= 1.0
    assert 0.0 <= row["aocc_test_accuracy"] <= 1.0
    assert row["final_objective"] <= row["random_guess_loss"] + 1e-9
    assert len(curve) > 0
    evaluations = [c["evaluations"] for c in curve]
    assert evaluations == sorted(evaluations) and max(evaluations) <= CONFIG.budget


def test_curve_objective_is_non_increasing():
    _, curve = run_single(SPEC, 1, "local_random_search", CONFIG, repetition=0)
    values = [c["objective"] for c in curve]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def test_row_carries_a_hitting_time_column_per_target():
    row, _ = run_single(SPEC, 1, "local_random_search", CONFIG, repetition=0)
    for target in CONFIG.accuracy_targets:
        assert f"t_acc_{target:g}" in row


def test_run_seed_is_deterministic_and_run_specific():
    assert _run_seed(1, "a", 1, 0) == _run_seed(1, "a", 1, 0)
    assert _run_seed(1, "a", 1, 0) != _run_seed(1, "a", 1, 1)
    assert _run_seed(1, "a", 1, 0) != _run_seed(1, "b", 1, 0)
    assert _run_seed(1, "a", 1, 0) != _run_seed(2, "a", 1, 0)


def test_repetitions_differ_but_reruns_reproduce():
    a0, _ = run_single(SPEC, 1, "local_random_search", CONFIG, repetition=0)
    a1, _ = run_single(SPEC, 1, "local_random_search", CONFIG, repetition=1)
    again, _ = run_single(SPEC, 1, "local_random_search", CONFIG, repetition=0)
    assert a0["final_objective"] != a1["final_objective"]
    assert a0["final_objective"] == pytest.approx(again["final_objective"])


def test_run_suite_writes_every_csv(tmp_path):
    suite = GrokkingSuite([SPEC], instances=[1, 2])
    config = RunConfig(
        optimizers=["local_random_search", "random_search"],
        budget=150,
        repetitions=2,
        curve_points=15,
    )
    runs, curves = run_suite(suite, config, output_dir=str(tmp_path), verbose=False)

    assert len(runs) == 2 * 2 * 2  # optimizers x instances x repetitions
    for name in ("runs.csv", "curves.csv", "summary.csv", "ert.csv"):
        assert (tmp_path / name).exists()

    on_disk = pd.read_csv(tmp_path / "runs.csv")
    assert len(on_disk) == len(runs)
    assert set(on_disk["optimizer"]) == {"local_random_search", "random_search"}
    assert set(curves["instance"]) == {1, 2}

    summary = pd.read_csv(tmp_path / "summary.csv")
    assert set(summary["optimizer"]) == {"local_random_search", "random_search"}


def test_run_suite_honours_per_spec_instances(tmp_path):
    suite = GrokkingSuite([SPEC], instances=[1, 2], spec_instances={SPEC.name: [2]})
    config = RunConfig(
        optimizers=["random_search"], budget=100, repetitions=1, curve_points=10
    )
    runs, _ = run_suite(suite, config, output_dir=str(tmp_path), verbose=False)
    assert list(runs["instance"]) == [2]


def test_run_suite_writes_ioh_logs_with_the_generalization_properties(tmp_path):
    other = ProblemSpec(p=11, embed_dim=4, hidden_dim=8, activation="quad")
    suite = GrokkingSuite([SPEC, other], instances=[1])
    config = RunConfig(
        optimizers=["random_search"],
        budget=100,
        repetitions=2,
        curve_points=10,
        ioh_logs=True,
    )
    run_suite(suite, config, output_dir=str(tmp_path), verbose=False)
    ioh_dir = tmp_path / "ioh"
    assert any(p.suffix == ".json" for p in ioh_dir.rglob("*"))

    data_files = sorted(ioh_dir.rglob("*.dat"))
    assert len(data_files) == 2  # one per problem, both under the same logger
    for path in data_files:
        lines = path.read_text().splitlines()
        header = lines[0].split()
        assert header[:2] == ["evaluations", "raw_y"]
        assert set(GrokkingProblem.WATCHED) <= set(header)

        # Every problem after the first shares the logger; its generalization
        # columns must carry that problem's own values, not NaN.
        values = dict(zip(header, (float(v) for v in lines[1].split())))
        assert all(np.isfinite(values[name]) for name in GrokkingProblem.WATCHED)
        assert 0.0 <= values["test_accuracy"] <= 1.0
        assert values["train_loss"] > 0.0


def test_ioh_logging_falls_back_to_serial(tmp_path, capsys):
    suite = GrokkingSuite([SPEC], instances=[1])
    config = RunConfig(
        optimizers=["random_search"],
        budget=80,
        repetitions=1,
        curve_points=8,
        ioh_logs=True,
        jobs=4,
    )
    run_suite(suite, config, output_dir=str(tmp_path), verbose=False)
    assert "running serially" in capsys.readouterr().out
    assert any(p.suffix == ".dat" for p in (tmp_path / "ioh").rglob("*"))


def test_parallel_and_serial_agree(tmp_path):
    suite = GrokkingSuite([SPEC], instances=[1, 2])
    base = dict(
        optimizers=["random_search"], budget=120, repetitions=1, curve_points=10
    )
    serial, _ = run_suite(
        suite, RunConfig(jobs=1, **base), output_dir=str(tmp_path / "s"), verbose=False
    )
    parallel, _ = run_suite(
        suite, RunConfig(jobs=2, **base), output_dir=str(tmp_path / "p"), verbose=False
    )
    key = ["problem", "instance", "run"]
    a = serial.set_index(key)["final_objective"].sort_index()
    b = parallel.set_index(key)["final_objective"].sort_index()
    assert np.allclose(a.to_numpy(), b.to_numpy())
