"""Figures for comparing optimizers on the grokking suite.

Five figures, each answering one question:

``convergence.png``     how fast does the objective fall?
``generalization.png``  does the incumbent ever generalize, and when?
``grokking_gap.png``    the classic grokking picture -- train fit early, test
                        accuracy much later -- drawn per optimizer.
``aocc.png``            anytime performance, aggregated.
``ert.png``             expected running time to a generalization target,
                        against problem dimension.
``ecdf.png``            fraction of (run, target) pairs solved by each budget.

Colors come from a pre-validated categorical palette, assigned to optimizers in
a fixed order so a series keeps its hue across every figure (and across a rerun
with a different subset of optimizers).  Every figure has a CSV beside it, which
is also the accessible table view for the three light-mode hues that sit below
3:1 contrast.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

# Categorical slots, in the fixed order of the reference palette.  Assigned by
# sorted optimizer name so identity never depends on which subset is plotted.
PALETTE: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
OTHER = "#6f6e69"

#: Display priority for hue assignment.  Hues are scarce (eight slots), so they
#: go to the series a reader is actually comparing: the gradient reference, then
#: the CMA family, then the population methods.  Anything past the eighth slot
#: -- by default the trivial random-search floor, which plots as a flat line --
#: takes the neutral "other" grey.  Unknown solvers sort alphabetically after
#: these, so a user's own method still gets a stable colour.
SERIES_ORDER: tuple[str, ...] = (
    "adam",
    "lm_cma_es",
    "lm_ma_es",
    "sep_cma_es",
    "cma_es",
    "pso",
    "de",
    "local_random_search",
    "random_search",
)

#: Distinct markers double as secondary encoding wherever the all-pairs colour
#: gate applies (scatter) or where print/CVD readers need a second channel.
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.8,
            "font.size": 9,
        }
    )


def _ordered(names: Sequence[str]) -> list[str]:
    """Known solvers in display priority, then anything else alphabetically."""
    rank = {name: i for i, name in enumerate(SERIES_ORDER)}
    return sorted(dict.fromkeys(names), key=lambda n: (rank.get(n, len(rank)), n))


def color_map(names: Sequence[str]) -> dict[str, str]:
    """Fixed hue per optimizer; series past the eighth take the neutral grey."""
    return {
        name: (PALETTE[i] if i < len(PALETTE) else OTHER)
        for i, name in enumerate(_ordered(names))
    }


def marker_map(names: Sequence[str]) -> dict[str, str]:
    return {name: MARKERS[i % len(MARKERS)] for i, name in enumerate(_ordered(names))}


def _grid_shape(n: int, max_cols: int = 3) -> tuple[int, int]:
    cols = min(max_cols, max(n, 1))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _label_last_point(ax, x, y, label, dy: float = 0.0) -> None:
    """Direct label at the end of a line, so identity is not colour-alone."""
    if len(x) == 0:
        return
    ax.annotate(
        label,
        xy=(x[-1], y[-1]),
        xytext=(3, dy),
        textcoords="offset points",
        color=TEXT_SECONDARY,
        fontsize=7.5,
        va="center",
        clip_on=False,
    )


def _median_curves(curves, problem: str, value: str):
    """Median across instances and repetitions, per optimizer."""
    sub = curves[curves["problem"] == problem]
    grouped = sub.groupby(["optimizer", "evaluations"])[value]
    med = grouped.median().reset_index()
    lo = grouped.quantile(0.25).reset_index().rename(columns={value: "lo"})
    hi = grouped.quantile(0.75).reset_index().rename(columns={value: "hi"})
    return med.merge(lo, on=["optimizer", "evaluations"]).merge(
        hi, on=["optimizer", "evaluations"]
    )


def _problem_order(curves, max_problems: int) -> list[str]:
    """Show the hardest-to-reach problems first; they separate optimizers most."""
    order = (
        curves.groupby("problem")["n_variables"].first().sort_values().index.tolist()
    )
    return order[:max_problems]


def _panel_title(name: str) -> str:
    return name.replace("GaaB_add_", "").replace("_", " ")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def plot_convergence(curves, out_path: str, max_problems: int = 9, log_y: bool = True):
    """Median best-so-far objective per optimizer, one panel per problem."""
    import matplotlib.pyplot as plt

    _style()
    problems = _problem_order(curves, max_problems)
    rows, cols = _grid_shape(len(problems))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.0 * rows), squeeze=False)
    colors = color_map(curves["optimizer"].unique())

    for ax, problem in zip(axes.ravel(), problems):
        med = _median_curves(curves, problem, "objective")
        for opt, g in med.groupby("optimizer"):
            ax.plot(g["evaluations"], g["objective"], color=colors[opt], label=opt)
            ax.fill_between(
                g["evaluations"], g["lo"], g["hi"], color=colors[opt], alpha=0.12, lw=0
            )
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        n = int(curves.loc[curves["problem"] == problem, "n_variables"].iloc[0])
        ax.set_title(f"{_panel_title(problem)}  (n={n})", loc="left")
        ax.set_xlabel("function evaluations")
        ax.set_ylabel("best-so-far objective")
    for ax in axes.ravel()[len(problems) :]:
        ax.set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6))
    fig.suptitle(
        "Convergence on the objective the optimizer actually minimizes",
        x=0.01, ha="left", fontsize=12, color=TEXT_PRIMARY, weight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_generalization(curves, out_path: str, max_problems: int = 9):
    """Test accuracy of the incumbent -- the quantity no optimizer optimizes."""
    import matplotlib.pyplot as plt

    _style()
    problems = _problem_order(curves, max_problems)
    rows, cols = _grid_shape(len(problems))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.0 * rows), squeeze=False)
    colors = color_map(curves["optimizer"].unique())

    for ax, problem in zip(axes.ravel(), problems):
        med = _median_curves(curves, problem, "test_accuracy")
        for opt, g in med.groupby("optimizer"):
            ax.plot(g["evaluations"], g["test_accuracy"], color=colors[opt], label=opt)
        ax.axhline(0.95, color=TEXT_MUTED, lw=0.9, ls=(0, (4, 3)))
        ax.annotate(
            "grokked (95%)", xy=(1, 0.95), xytext=(2, 3), textcoords="offset points",
            color=TEXT_MUTED, fontsize=7.5,
        )
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        n = int(curves.loc[curves["problem"] == problem, "n_variables"].iloc[0])
        ax.set_title(f"{_panel_title(problem)}  (n={n})", loc="left")
        ax.set_xlabel("function evaluations")
        ax.set_ylabel("test accuracy of incumbent")
    for ax in axes.ravel()[len(problems) :]:
        ax.set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6))
    fig.suptitle(
        "Generalization: held-out accuracy never enters the objective",
        x=0.01, ha="left", fontsize=12, color=TEXT_PRIMARY, weight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_grokking_gap(curves, out_path: str, max_optimizers: int | None = None):
    """Train vs test accuracy per optimizer, pooled over problems.

    The gap between the two lines *is* grokking: the left line is when the
    optimizer fits, the right one is when it understands.
    """
    import matplotlib.pyplot as plt

    _style()
    optimizers = _ordered(curves["optimizer"].unique())
    if max_optimizers is not None:
        optimizers = optimizers[:max_optimizers]
    rows, cols = _grid_shape(len(optimizers))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 2.9 * rows), squeeze=False)
    colors = color_map(curves["optimizer"].unique())

    for ax, opt in zip(axes.ravel(), optimizers):
        sub = curves[curves["optimizer"] == opt]
        med = sub.groupby("evaluations")[["train_accuracy", "test_accuracy"]].median()
        ax.plot(med.index, med["train_accuracy"], color=colors[opt], ls=(0, (4, 2)))
        ax.plot(med.index, med["test_accuracy"], color=colors[opt])
        # Nudge the two labels apart: on an optimizer that never learns
        # anything the curves coincide exactly.
        gap = abs(med["train_accuracy"].iloc[-1] - med["test_accuracy"].iloc[-1])
        offset = 0.0 if gap > 0.05 else 5.0
        _label_last_point(ax, med.index.values, med["train_accuracy"].values, "train", offset)
        _label_last_point(ax, med.index.values, med["test_accuracy"].values, "test", -offset)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_title(opt, loc="left")
        ax.set_xlabel("function evaluations")
        ax.set_ylabel("median accuracy")
    for ax in axes.ravel()[len(optimizers) :]:
        ax.set_visible(False)

    fig.suptitle(
        "The grokking gap: dashed = train, solid = held-out (median over the suite)",
        x=0.01, ha="left", fontsize=12, color=TEXT_PRIMARY, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_aocc(runs, out_path: str):
    """Anytime performance: AOCC on the objective, and on test accuracy."""
    import matplotlib.pyplot as plt

    _style()
    colors = color_map(runs["optimizer"].unique())
    order = (
        runs.groupby("optimizer")["aocc_objective"].mean().sort_values(ascending=False).index.tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for ax, column, title in zip(
        axes,
        ["aocc_objective", "aocc_test_accuracy"],
        ["AOCC (objective, log-scaled)", "AOCC (test accuracy)"],
    ):
        means = [runs.loc[runs["optimizer"] == o, column].mean() for o in order]
        y = np.arange(len(order))
        # Rounded data-ends anchored to the baseline; a 2px surface gap between
        # neighbouring bars comes from the bar height vs. spacing.
        ax.barh(y, means, height=0.62, color=[colors[o] for o in order])
        for yi, o in zip(y, order):
            vals = runs.loc[runs["optimizer"] == o, column].to_numpy()
            ax.scatter(
                vals, np.full_like(vals, yi, dtype=float), s=9,
                color=SURFACE, edgecolor=colors[o], linewidth=0.8, zorder=3, alpha=0.75,
            )
        # Park the value at the right edge: printed at the bar end it collides
        # with the per-run scatter that overhangs the mean.
        limit = max(1.0, float(np.max([np.max(runs[column]), *means])) * 1.12)
        for yi, v in zip(y, means):
            ax.annotate(
                f"{v:.3f}", xy=(limit, yi), xytext=(-2, 0), textcoords="offset points",
                va="center", ha="right", fontsize=8, color=TEXT_SECONDARY,
            )
        ax.set_yticks(y, order, color=TEXT_PRIMARY)
        ax.invert_yaxis()
        ax.set_xlim(0, limit)
        ax.set_xlabel("higher is better")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)

    fig.suptitle(
        "Anytime performance over the curated grokking suite",
        x=0.01, ha="left", fontsize=12, color=TEXT_PRIMARY, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_ert(ert_df, out_path: str, target: float | None = None):
    """ERT to a generalization target, against problem dimension.

    Runs that never reach the target have infinite ERT; they are drawn as open
    markers pinned above the axis rather than dropped, because "never solved"
    is the most important thing this figure has to say.
    """
    import matplotlib.pyplot as plt

    _style()
    if target is None:
        # The hardest target anybody reached, else the easiest one available.
        reached = ert_df[np.isfinite(ert_df["ert"])]
        pool = reached if len(reached) else ert_df
        target = float(pool["target_test_accuracy"].max())
    sub = ert_df[np.isclose(ert_df["target_test_accuracy"], target)].copy()
    colors = color_map(ert_df["optimizer"].unique())
    markers = marker_map(ert_df["optimizer"].unique())

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    finite = sub[np.isfinite(sub["ert"])]
    floor = float(finite["ert"].min()) if len(finite) else 1.0
    ceiling = float(finite["ert"].max()) * 2.5 if len(finite) else float(sub["budget"].max())

    optimizers = _ordered(sub["optimizer"].unique())
    for opt in optimizers:
        g = sub[sub["optimizer"] == opt]
        ok = np.isfinite(g["ert"])
        # Points only: several problems share a dimension, so a connecting line
        # would draw a trend that the x-axis does not support.
        ax.scatter(
            g.loc[ok, "n_variables"], g.loc[ok, "ert"],
            color=colors[opt], marker=markers[opt], s=46,
            edgecolor=SURFACE, linewidth=0.8, label=opt, zorder=3,
        )

    # Everything that never reached the target goes in a banded strip above the
    # data, one row per optimizer, so a total failure stays readable instead of
    # collapsing into one overplotted line.
    ax.axhline(ceiling, color=TEXT_MUTED, lw=0.9, ls=(0, (2, 3)))
    for k, opt in enumerate(optimizers):
        g = sub[(sub["optimizer"] == opt) & ~np.isfinite(sub["ert"])]
        if not len(g):
            continue
        row = ceiling * (1.55 ** (k + 1))
        ax.scatter(
            g["n_variables"], np.full(len(g), row),
            facecolor="none", edgecolor=colors[opt], marker=markers[opt],
            s=44, linewidth=1.2,
        )
        ax.annotate(
            f"{opt} — never reached ({len(g)})",
            xy=(1.0, row), xycoords=("axes fraction", "data"),
            xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=7.5, color=TEXT_SECONDARY, annotation_clip=False,
        )
    ax.set_ylim(floor / 1.6, ceiling * 1.55 ** (len(optimizers) + 0.7))

    ax.set_yscale("log")
    ax.set_xlabel("search-space dimension (number of network parameters)")
    ax.set_ylabel(f"ERT to test accuracy >= {target:g}  (evaluations, lower is better)")
    ax.set_title(
        f"Expected running time to generalize (target {target:g})",
        loc="left", fontsize=12, color=TEXT_PRIMARY,
    )
    # The never-reached labels take the right quarter of the figure, so the
    # legend has to fit inside the narrowed axes.
    ax.legend(ncol=2 if len(optimizers) > 4 else 1, loc="lower left")
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_ecdf(curves, out_path: str, n_targets: int = 12):
    """Fraction of (run, objective-target) pairs solved by a given budget."""
    import matplotlib.pyplot as plt

    _style()
    colors = color_map(curves["optimizer"].unique())
    hi = float(curves["objective"].quantile(0.95))
    lo = max(float(curves["objective"].min()), 1e-3)
    targets = np.geomspace(hi, lo, n_targets)

    optimizers = _ordered(curves["optimizer"].unique())
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for opt, g in curves.groupby("optimizer"):
        pivot = g.pivot_table(
            index="evaluations", columns=["problem", "instance", "run"], values="objective"
        )
        # Cumulative-best is already encoded in the incumbent curve.
        frac = np.array(
            [(pivot.to_numpy() <= t).mean(axis=1) for t in targets]
        ).mean(axis=0)
        ax.plot(pivot.index.to_numpy(), frac, color=colors[opt], label=opt)
        if len(optimizers) <= 4:
            # Past four series the end-of-line labels collide with each other;
            # the legend carries identity on its own from there.
            _label_last_point(ax, pivot.index.to_numpy(), frac, opt)

    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("function evaluations")
    ax.set_ylabel("fraction of (run, target) pairs solved")
    ax.set_title(
        "Empirical cumulative distribution over objective targets",
        loc="left", fontsize=12, color=TEXT_PRIMARY,
    )
    ax.legend(ncol=min(len(optimizers), 4), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_screening(summary, out_path: str):
    """Why each candidate problem was kept or dropped, by verdict."""
    import matplotlib.pyplot as plt

    _style()
    verdict_color = {
        "grokking": "#1baf7a",
        "early_generalization": "#eda100",
        "no_generalization": "#2a78d6",
        "no_memorization": "#eb6834",
        "divergence": "#e34948",
    }
    counts = summary["modal_verdict"].value_counts()
    order = [v for v in verdict_color if v in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    y = np.arange(len(order))
    ax.barh(y, [counts[v] for v in order], height=0.6, color=[verdict_color[v] for v in order])
    for yi, v in zip(y, order):
        ax.annotate(
            str(counts[v]), xy=(counts[v], yi), xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=8, color=TEXT_SECONDARY,
        )
    ax.set_yticks(y, [v.replace("_", " ") for v in order], color=TEXT_PRIMARY)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("candidate problems")
    ax.set_title("Screening verdicts", loc="left")

    ax = axes[1]
    for verdict in order:
        sub = summary[summary["modal_verdict"] == verdict]
        ax.scatter(
            sub["reg_strength"], sub["n_variables"], s=34,
            color=verdict_color[verdict], edgecolor=SURFACE, linewidth=0.8,
            label=verdict.replace("_", " "),
        )
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("regularization strength (direct loss factor)")
    ax.set_ylabel("search-space dimension")
    ax.set_title("Where grokking lives in the family", loc="left")
    ax.legend(ncol=2, fontsize=7.5)

    fig.suptitle(
        "Filtering the candidate family down to problems that actually grok",
        x=0.01, ha="left", fontsize=12, color=TEXT_PRIMARY, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def make_all(
    runs, curves, ert_df, out_dir: str, screening_summary=None, ert_target: float | None = None
) -> list[str]:
    """Render every figure into ``out_dir``; returns the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    paths = [
        plot_convergence(curves, os.path.join(out_dir, "convergence.png")),
        plot_generalization(curves, os.path.join(out_dir, "generalization.png")),
        plot_grokking_gap(curves, os.path.join(out_dir, "grokking_gap.png")),
        plot_aocc(runs, os.path.join(out_dir, "aocc.png")),
        plot_ecdf(curves, os.path.join(out_dir, "ecdf.png")),
    ]
    if ert_df is not None and len(ert_df):
        paths.append(plot_ert(ert_df, os.path.join(out_dir, "ert.png"), target=ert_target))
    if screening_summary is not None and len(screening_summary):
        paths.append(plot_screening(screening_summary, os.path.join(out_dir, "screening.png")))
    return paths
