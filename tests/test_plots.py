import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from gaab import plots  # noqa: E402


def fake_curves(
    optimizers=("cma_es", "pso"), problems=("GaaB_add_p7_a", "GaaB_add_p11_b")
):
    rng = np.random.default_rng(0)
    rows = []
    for opt in optimizers:
        for i, problem in enumerate(problems):
            for run in range(2):
                for t in np.unique(np.geomspace(1, 1000, 20).astype(int)):
                    rows.append(
                        {
                            "optimizer": opt,
                            "problem": problem,
                            "instance": 1,
                            "run": run,
                            "n_variables": 100 * (i + 1),
                            "evaluations": int(t),
                            "objective": float(
                                2.0 / (1 + t / 100) + 0.01 * rng.random()
                            ),
                            "train_accuracy": float(min(1.0, t / 500)),
                            "test_accuracy": float(min(1.0, t / 900)),
                        }
                    )
    return pd.DataFrame(rows)


def fake_runs(optimizers=("cma_es", "pso")):
    return pd.DataFrame(
        [
            {
                "optimizer": opt,
                "problem": f"GaaB_add_p7_{j}",
                "instance": 1,
                "run": r,
                "n_variables": 100 * (j + 1),
                "p": 7,
                "budget": 1000,
                "aocc_objective": 0.4 + 0.1 * i,
                "aocc_test_accuracy": 0.3 + 0.1 * i,
                "final_objective": 0.5,
                "final_test_accuracy": 0.6,
                "best_test_accuracy": 0.7,
                "grokked": bool(i),
                "t_acc_0.5": 400 if i else np.nan,
            }
            for i, opt in enumerate(optimizers)
            for j in range(2)
            for r in range(2)
        ]
    )


def test_color_map_is_stable_under_subsetting():
    full = plots.color_map(["pso", "cma_es", "de"])
    subset = plots.color_map(["pso", "cma_es"])
    assert full["cma_es"] == subset["cma_es"]
    assert full["pso"] != full["cma_es"]


def test_color_map_folds_a_ninth_series_into_other():
    names = [f"opt{i}" for i in range(9)]
    mapping = plots.color_map(names)
    assert sum(v == plots.OTHER for v in mapping.values()) == 1
    assert len(set(mapping.values())) == len(plots.PALETTE) + 1


def test_marker_map_gives_every_series_a_second_channel():
    mapping = plots.marker_map(["a", "b", "c"])
    assert len(set(mapping.values())) == 3


@pytest.mark.parametrize(
    "fn,arg",
    [
        (plots.plot_convergence, "curves"),
        (plots.plot_generalization, "curves"),
        (plots.plot_grokking_gap, "curves"),
        (plots.plot_ecdf, "curves"),
        (plots.plot_aocc, "runs"),
    ],
)
def test_each_figure_writes_a_file(tmp_path, fn, arg):
    data = fake_curves() if arg == "curves" else fake_runs()
    out = tmp_path / f"{fn.__name__}.png"
    assert fn(data, str(out)) == str(out)
    assert out.stat().st_size > 5_000


def test_ert_figure_marks_unreached_targets(tmp_path):
    from gaab.analysis import ert_table

    ert_df = ert_table(fake_runs(), accuracy_targets=[0.5])
    assert np.isinf(ert_df["ert"]).any(), "the fixture needs an unreached target"
    out = tmp_path / "ert.png"
    assert plots.plot_ert(ert_df, str(out)) == str(out)
    assert out.exists()


def test_screening_figure(tmp_path):
    summary = pd.DataFrame(
        {
            "problem": ["a", "b", "c"],
            "modal_verdict": ["grokking", "no_generalization", "no_memorization"],
            "n_variables": [100, 200, 300],
            "reg_strength": [0.5, 0.0, 10.0],
        }
    )
    out = tmp_path / "screening.png"
    assert plots.plot_screening(summary, str(out)) == str(out)
    assert out.exists()


def test_make_all_renders_the_whole_set(tmp_path):
    from gaab.analysis import ert_table

    runs = fake_runs()
    paths = plots.make_all(runs, fake_curves(), ert_table(runs, [0.5]), str(tmp_path))
    assert {p.split("/")[-1] for p in paths} == {
        "convergence.png",
        "generalization.png",
        "grokking_gap.png",
        "aocc.png",
        "ecdf.png",
        "ert.png",
    }


def test_series_order_gives_hues_to_the_compared_methods():
    names = [
        "random_search",
        "adam",
        "de",
        "pso",
        "cma_es",
        "sep_cma_es",
        "lm_cma_es",
        "lm_ma_es",
        "local_random_search",
    ]
    mapping = plots.color_map(names)
    # Nine series, eight hues: the trivial floor is the one that goes grey.
    assert mapping["random_search"] == plots.OTHER
    for name in ("adam", "lm_cma_es", "lm_ma_es", "sep_cma_es", "cma_es"):
        assert mapping[name] in plots.PALETTE
    assert len({mapping[n] for n in names}) == len(names)


def test_unknown_solvers_still_get_a_stable_colour():
    a = plots.color_map(["adam", "my_solver", "cma_es"])
    b = plots.color_map(["my_solver", "cma_es", "adam"])
    assert a == b
    assert a["my_solver"] in plots.PALETTE
