"""IOH-compatible problem objects.

``GrokkingProblem`` subclasses ``ioh.problem.RealSingleObjective``, so it is the
same kind of object as a BBOB function: it can be handed to any IOHexperimenter
algorithm, attached to ``ioh.logger.Analyzer`` (and thus read by IOHanalyzer),
and it keeps IOH's own evaluation counter and best-so-far state.

Two things are added on top of a plain IOH problem, both needed because
grokking is about *generalization*, which no ordinary objective exposes:

* watchable properties (train/test loss and accuracy of the incumbent), which
  ``logger.watch`` writes into the IOH data files;
* an *improvement history* -- one record per strict improvement of the
  objective -- from which the best-so-far curve, AOCC and ERT can be
  reconstructed at any budget without logging every evaluation.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterator, Sequence

import numpy as np

try:
    import ioh
except ImportError as exc:  # pragma: no cover - dependency is declared
    raise ImportError(
        "gaab requires IOHexperimenter; install it with `pip install ioh`."
    ) from exc

from gaab.data import is_prime, make_dataset
from gaab.model import ModelConfig, n_parameters
from gaab.objective import GrokkingObjective, Metrics

@dataclass(frozen=True)
class ProblemSpec:
    """A point in the benchmark family: one landscape, before instancing.

    The IOH *instance* id is deliberately not part of the spec -- it selects
    which pairs are held out and how the reference init is drawn, which is
    exactly the role instances play for BBOB.
    """

    p: int = 11
    embed_dim: int = 8
    hidden_dim: int = 32
    rank: int | None = None
    reg_strength: float = 0.0
    train_frac: float = 0.5
    op: str = "add"
    activation: str = "relu"
    rank_targets: tuple[str, ...] = ("W1",)
    lb: float = -2.0
    ub: float = 2.0
    problem_id: int | None = None

    def __post_init__(self) -> None:
        if not is_prime(self.p):
            raise ValueError(f"p must be prime, got {self.p}.")
        if self.ub <= self.lb:
            raise ValueError("ub must exceed lb.")

    @property
    def model_config(self) -> ModelConfig:
        return ModelConfig(
            p=self.p,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            rank=self.rank,
            activation=self.activation,
            rank_targets=tuple(self.rank_targets),
        )

    @property
    def n_variables(self) -> int:
        return n_parameters(self.model_config)

    @property
    def name(self) -> str:
        # The *targets* are part of the identity: a rank-4 W1 and a rank-4
        # W1+W2 are different landscapes with different dimensions.
        rank = "full" if self.rank is None else f"{self.rank}{''.join(self.rank_targets)}"
        return (
            f"GaaB_{self.op}_p{self.p}_e{self.embed_dim}_h{self.hidden_dim}"
            f"_r{rank}_reg{self.reg_strength:g}_tf{self.train_frac:g}"
        )

    @property
    def stable_id(self) -> int:
        """Deterministic small integer id, stable across processes.

        ``hash()`` is salted per interpreter, so CRC32 of the name is used
        instead -- results CSVs from separate runs have to line up.
        """
        if self.problem_id is not None:
            return int(self.problem_id)
        return 1 + int(zlib.crc32(self.name.encode())) % 900_000

    def with_id(self, problem_id: int) -> "ProblemSpec":
        return replace(self, problem_id=problem_id)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["name"] = self.name
        d["n_variables"] = self.n_variables
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProblemSpec":
        fields = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in fields}
        if "rank_targets" in kwargs:  # JSON round-trips tuples as lists
            kwargs["rank_targets"] = tuple(kwargs["rank_targets"])
        return cls(**kwargs)


@dataclass
class ImprovementRecord:
    """One strict improvement of the objective, with what it generalized to."""

    evaluations: int
    objective: float
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    reg_penalty: float
    weight_norm: float


class GrokkingProblem(ioh.problem.RealSingleObjective):
    """A modular-addition network's weight space, as an IOH real problem."""

    def __init__(
        self,
        spec: ProblemSpec | None = None,
        instance: int = 1,
        track_history: bool = True,
    ) -> None:
        spec = spec or ProblemSpec()
        self.spec = spec
        self.instance_id = int(instance)
        self.track_history = bool(track_history)

        data = make_dataset(
            p=spec.p, op=spec.op, train_frac=spec.train_frac, seed=self.instance_id
        )
        self.objective = GrokkingObjective(
            spec.model_config, data, reg_strength=spec.reg_strength
        )
        n = self.objective.n_variables

        super().__init__(
            spec.name,
            n,
            self.instance_id,
            True,  # minimization
            ioh.RealBounds(n, spec.lb, spec.ub, ioh.ConstraintEnforcement.NOT),
        )
        # ``meta_data.problem_id`` is read-only and stays 0 for problems
        # defined outside IOH's own registry; IOH keys its log files by name as
        # well, so distinct specs never collide.  ``spec.stable_id`` is what we
        # key our own CSVs by.
        self.problem_id = spec.stable_id

        self._best_y = np.inf
        self._incumbent: Metrics | None = None
        self.history: list[ImprovementRecord] = []

    # -- IOH interface ------------------------------------------------------
    def evaluate(self, x: Sequence[float]) -> float:
        theta = np.asarray(x, dtype=np.float64)
        y = self.objective(theta)
        if y < self._best_y:
            # Test metrics cost a second forward pass, so they are only paid
            # for when the incumbent actually moves.
            self._best_y = y
            self._incumbent = self.objective.metrics(theta)
            if self.track_history:
                self.history.append(
                    ImprovementRecord(
                        # IOH increments its counter after `evaluate` returns.
                        evaluations=self.state.evaluations + 1,
                        **self._incumbent.as_dict(),
                    )
                )
        return y

    def reset(self) -> None:
        super().reset()
        self._best_y = np.inf
        self._incumbent = None
        self.history = []

    # -- watchable properties (picked up by ``logger.watch``) ---------------
    def _incumbent_value(self, key: str, default: float = float("nan")) -> float:
        if self._incumbent is None:
            return default
        return float(getattr(self._incumbent, key))

    @property
    def train_loss(self) -> float:
        return self._incumbent_value("train_loss")

    @property
    def train_accuracy(self) -> float:
        return self._incumbent_value("train_accuracy")

    @property
    def test_loss(self) -> float:
        return self._incumbent_value("test_loss")

    @property
    def test_accuracy(self) -> float:
        return self._incumbent_value("test_accuracy")

    @property
    def reg_penalty(self) -> float:
        return self._incumbent_value("reg_penalty")

    @property
    def weight_norm(self) -> float:
        return self._incumbent_value("weight_norm")

    #: Property names to pass to ``ioh.logger.Analyzer.watch``.
    WATCHED = (
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "reg_penalty",
        "weight_norm",
    )

    # -- convenience --------------------------------------------------------
    @property
    def n_variables(self) -> int:
        return self.meta_data.n_variables

    @property
    def random_guess_loss(self) -> float:
        return self.objective.random_guess_loss

    def initial_theta(self, seed: int | None = None, scale: float = 1.0) -> np.ndarray:
        """A sensible starting point (fan-in scaled init), for warm starts."""
        return self.objective.initial_theta(
            seed=self.instance_id if seed is None else seed, scale=scale
        )

    def history_frame(self):
        """Improvement history as a ``pandas.DataFrame`` (lazy import)."""
        import pandas as pd

        return pd.DataFrame([asdict(r) for r in self.history])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<GrokkingProblem {self.spec.name} instance={self.instance_id} "
            f"n={self.n_variables}>"
        )


class IncumbentWatcher:
    """Stable stand-in that forwards the watched properties to the live problem.

    ``logger.watch`` binds to whichever Python object it was handed and keeps
    reading that one object for the logger's lifetime -- a second ``watch``
    call with a different problem does not rebind it.  A campaign reuses one
    logger across many problems, so without this indirection every problem
    after the first logs the *first* problem's (reset, hence NaN) properties.
    Watching a forwarding object once and repointing it is what makes the
    generalization columns correct for the whole campaign.
    """

    def __init__(self) -> None:
        self.problem: "GrokkingProblem | None" = None

    def _value(self, name: str) -> float:
        if self.problem is None:
            return float("nan")
        return float(getattr(self.problem, name))


for _name in GrokkingProblem.WATCHED:
    setattr(
        IncumbentWatcher,
        _name,
        property(lambda self, _n=_name: self._value(_n)),
    )


class GrokkingSuite:
    """An iterable collection of problems, in the spirit of ``ioh.suite``.

    Instances may be curated *per spec*: screening finds that a landscape often
    groks under some train/test splits and not others, and a suite is more
    useful if it keeps the splits that grok rather than dropping the whole
    configuration.  ``spec_instances`` holds those exceptions; anything not
    listed falls back to the suite-wide ``instances``.
    """

    def __init__(
        self,
        specs: Sequence[ProblemSpec],
        instances: Sequence[int] = (1,),
        track_history: bool = True,
        spec_instances: dict[str, Sequence[int]] | None = None,
    ) -> None:
        if not specs:
            raise ValueError("A suite needs at least one spec.")
        # Assign compact ids in declaration order unless a spec pinned its own.
        self.specs = [
            s if s.problem_id is not None else s.with_id(i + 1)
            for i, s in enumerate(specs)
        ]
        self.instances = list(instances)
        self.track_history = track_history
        self.spec_instances = {k: list(v) for k, v in (spec_instances or {}).items()}

    @property
    def name(self) -> str:
        return "GaaB"

    def instances_for(self, spec: ProblemSpec) -> list[int]:
        return self.spec_instances.get(spec.name, self.instances)

    def __len__(self) -> int:
        return sum(len(self.instances_for(s)) for s in self.specs)

    def __iter__(self) -> Iterator[GrokkingProblem]:
        for spec in self.specs:
            for instance in self.instances_for(spec):
                yield GrokkingProblem(spec, instance, self.track_history)

    def get(self, index: int, instance: int = 1) -> GrokkingProblem:
        return GrokkingProblem(self.specs[index], instance, self.track_history)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(
                {
                    "instances": self.instances,
                    "spec_instances": self.spec_instances,
                    "specs": [s.to_dict() for s in self.specs],
                },
                fh,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "GrokkingSuite":
        with open(path) as fh:
            payload = json.load(fh)
        return cls(
            [ProblemSpec.from_dict(d) for d in payload["specs"]],
            instances=payload.get("instances", [1]),
            spec_instances=payload.get("spec_instances"),
        )

    def table(self):
        """One row per spec: the suite as a readable ``pandas.DataFrame``."""
        import pandas as pd

        rows = []
        for spec in self.specs:
            row = spec.to_dict()
            row["instances"] = ",".join(str(i) for i in self.instances_for(spec))
            rows.append(row)
        return pd.DataFrame(rows)
