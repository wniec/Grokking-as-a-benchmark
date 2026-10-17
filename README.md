# GaaB — Grokking as a Benchmark

A benchmark suite for **continuous black-box optimization** whose problems are
small neural networks learning ***p*-modular addition** — `a + b = c (mod p)`,
the canonical grokking task.

Each problem hands an optimizer a flat vector of network weights and asks it to
minimize a training loss. What makes the suite unusual is a second axis that
the optimizer is never allowed to see: **accuracy on held-out pairs**. On these
landscapes that accuracy arrives *long after* the training loss has been driven
down — the delayed transition known as **grokking** — and only problems where
that transition is demonstrably achievable are kept in the suite.

That gives a benchmark with two questions instead of one:

| axis | metric | what it asks |
|---|---|---|
| optimization | `aocc_objective`, ERT to loss targets | how fast does the loss fall? |
| generalization | `aocc_test_accuracy`, ERT to accuracy targets | does the optimizer ever *grok*, and when? |

---

## Contents

```
src/gaab/
  data.py        p-modular arithmetic table + train/test split
  model.py       NumPy embedding-MLP over a flat weight vector, with low-rank
                 factorization and analytic gradients
  objective.py   f(theta) = CE_train(theta) + lambda * mean(theta^2)
  problem.py     GrokkingProblem (an ioh.problem.RealSingleObjective), GrokkingSuite
  suite.py       candidate grids + the curated, screened suite
  screening.py   gradient reference run + the grokking filter
  optimizers.py  CMA-ES, sep-CMA-ES, LM-CMA-ES, LM-MA-ES, sep-NES, PSO, DE,
                 random search, Adam reference, + optional Nevergrad/pycma
  parallel.py    process-pool helpers (spawned workers, single-threaded BLAS)
  runner.py      run a family x optimizers, write CSVs
  analysis.py    AOCC, ERT, ECDF, aggregation
  landscape.py   Hessian/eigen-spectra, saddle & local-optima diagnostics
  plots.py       the figures
  cli.py         `python -m gaab.cli screen | run | analyze | suite | optimizers`
scripts/
  screen_suite.py     curate the suite
  run_benchmark.py    run the whole family on given optimizers  -> CSVs
  analyze_results.py  tables + figures
  validate_optimizers.py  sanity-check the solvers on BBOB functions
  reg_sweep.py        regularization -> grokking-strength study
  trainfrac_sweep.py  train/test split -> grokking study (per-p optima)
  curate_suite.py     stratified-subsample a screened suite
  landscape_analysis.py  plateaus / saddles / local optima on one problem (figures)
  landscape_sweep.py     the same diagnostics across the whole suite (population stats)
  run_one_algorithm.py  run ONE optimizer on gaab and/or BBOB (HPC array unit)
  aggregate_hpc.py    combine per-algorithm HPC results + gaab-vs-BBOB ranking
hpc/                  Slurm array job (all optimizers x both benchmarks), see hpc/README.md
suites/grokking_v1.json   the shipped curated suite
```

## Install

```bash
uv pip install -e .          # or: pip install -e .
uv pip install -e ".[cma]"   # optional: the reference `cma` package
```

## Run the benchmark

One script runs the whole family on a given optimizer and stores results as CSV:

```bash
python scripts/run_benchmark.py --optimizer lm_cma_es
python scripts/run_benchmark.py --plots --jobs 10   # the full experiment roster
```

With no `--optimizer`, the run uses the **experiment roster** — `adam`,
`lm_cma_es`, `lm_ma_es`, `sep_cma_es`, `cma_es`, `pso`, `local_random_search`
(`de` and the uniform `random_search` floor are excluded; both sat at the bottom
on every metric). The budget defaults to **60,000 function evaluations**
(`gaab.optimizers.DEFAULT_BENCHMARK_OPTIMIZERS`, `--budget`).

Output in `results/benchmark/`:

| file | contents |
|---|---|
| `runs.csv` | one row per (optimizer, problem, instance, repetition): AOCC, final loss, final/best test accuracy, `t_fit`, `t_gen`, `grokked`, hitting times for every target |
| `curves.csv` | incumbent loss / train accuracy / test accuracy on a log-spaced budget grid |
| `summary.csv` | per-optimizer aggregate |
| `ert.csv` | ERT and success rate per (optimizer, problem, accuracy target) |
| `figures/` | with `--plots` |
| `ioh/` | with `--ioh-logs`: IOHprofiler-format data, openable in IOHanalyzer |

## IOH compatibility

`GrokkingProblem` **is** an `ioh.problem.RealSingleObjective` — the same kind of
object as a BBOB function — so it drops into any IOHexperimenter workflow:

```python
import ioh
from gaab import GrokkingProblem, IncumbentWatcher, load_suite

suite = load_suite()
logger = ioh.logger.Analyzer(root="results", folder_name="cma", algorithm_name="cma_es")

# `logger.watch` binds to one object for the logger's lifetime -- calling it
# again per problem does NOT rebind it -- so watch a forwarding object once and
# repoint it. (`gaab.runner` does this for you.)
watcher = IncumbentWatcher()
logger.watch(watcher, list(GrokkingProblem.WATCHED))

for problem in suite:                      # iterates (spec, instance) pairs
    watcher.problem = problem
    problem.attach_logger(logger)
    my_optimizer(problem, budget=20_000)   # problem(x) -> float
    problem.detach_logger()
    problem.reset()
logger.close()
```

`problem.bounds`, `problem.meta_data`, `problem.state.current_best` and
`problem.reset()` all behave as IOH defines them. The watched properties
(`train_loss`, `train_accuracy`, `test_loss`, `test_accuracy`, `reg_penalty`,
`weight_norm`) record the incumbent's generalization alongside the objective,
so the grokking transition survives into the IOHanalyzer data files.

## Large-scale methods

Full CMA-ES stores and eigendecomposes an `n x n` covariance matrix, which is
the wrong shape of cost for a neural-network weight space. Several cheaper
methods are bundled, all `O(n)` or `O(m n)` per sample with `m ~ 4 + 3 ln n`:

| solver | model / family | learns rotation? |
|---|---|---|
| `sep_cma_es` | diagonal covariance (Ros & Hansen, 2008) | no |
| `lm_cma_es` | limited-memory Cholesky factor over `m` stored evolution paths, step size by the population success rule (Loshchilov, 2015) | yes |
| `lm_ma_es` | `m` rank-one corrections at geometrically spaced learning rates (Loshchilov, Glasmachers & Beyer, 2019) | yes |
| `sep_nes` | separable Natural Evolution Strategy — natural gradient on a per-coordinate Gaussian (Wierstra et al., 2014) | no |

A further set is available *optionally* through [Nevergrad](https://facebookresearch.github.io/nevergrad/)
(`pip install nevergrad`), skipped cleanly when it is absent: `ng_ngopt`
(Nevergrad's self-configuring meta-optimizer), `ng_diagonal_cma`,
`ng_rescaled_cma`, `ng_cma`. Likewise `pycma_es` wraps the reference `cma`
package. The HPC sweep (below) picks up every registered method automatically.

The self-contained methods are implementations, not wrappers, so they are
checked against the functions whose answers are known:

```bash
python scripts/validate_optimizers.py --dimensions 128 512
```

It runs the solvers on IOHexperimenter's BBOB functions and asserts the
qualitative facts the literature reports — measured here at `n = 512`,
budget `100n`, median gap to the optimum over 2 runs:

| function | `cma_es` | `sep_cma_es` | `lm_cma_es` | `lm_ma_es` |
|---|---|---|---|---|
| sphere | 1.7e-07 (16s) | 5.1e-08 (1s) | **1.1e-11** (3s) | 2.3e-10 (2s) |
| ellipsoid (separable) | 3.9e+06 | **2.4e+05** | 2.5e+05 | 3.9e+05 |
| ellipsoid (rotated) | 3.4e+06 (30s) | 4.2e+06 (16s) | **2.9e+05** (30s) | 3.8e+05 (17s) |
| discus | 8.6e+03 | 8.9e+03 | **3.3e+03** | 8.7e+03 |
| different powers | 3.3e-02 | 2.8e-02 | **5.9e-03** | 1.0e-02 |

A diagonal covariance wins the *separable* ellipsoid and loses the *rotated*
one by an order of magnitude; the limited-memory methods follow the rotation,
and `lm_ma_es` does it in a fifteenth of full CMA-ES's runtime.

## The problem definition

For a prime `p`, every pair `(a, b)` in `{0..p-1}^2` is an example with label
`(a + b) mod p`; a fraction `train_frac` is used for training and the rest is
held out. The network is

```
x      = concat(E[a], E[b])          E: (p, embed_dim)
h      = act(x @ W1 + b1)            W1: (2*embed_dim, hidden_dim)
logits = h @ W2 + b2                 W2: (hidden_dim, p)
```

and the objective is

```
f(theta) = cross_entropy_train(theta) + reg_strength * mean(theta^2)
```

Four knobs define the family:

* **`p`** — the prime modulus (primality is enforced; `Z_p` must be a field for
  the Fourier-feature solution to exist). `p = 3` is the supported floor, but
  see below: whether a given `p` *groks* is a separate question the screener
  answers.
* **`reg_strength`** — regularization as a **direct factor of the loss**, not as
  weight decay in an update rule. That matters: a black-box optimizer has no
  update rule to decay, so the pressure has to live in the landscape. Sweeping
  it moves a problem between "memorizes and never generalizes", "groks", and
  "cannot even fit".
* **`rank` / `rank_targets`** — any weight matrix named in `rank_targets`
  (`E`, `W1`, `W2`) is replaced by a rank-*r* product `U @ V`. This reshapes
  the landscape and *may* shrink the parameter count — a rank above a matrix's
  natural bottleneck makes it grow instead (`(m + n) * r` vs `m * n`), so the
  knob is not simply "fewer parameters". A rank at or above `min(m, n)` is no
  restriction at all, so it silently degenerates back to the dense matrix
  rather than wasting parameters.
* **`embed_dim` / `hidden_dim` / `train_frac`** — width and data budget.

### Which matrix the rank restricts matters more than the rank does

`W1` tolerates a hard bottleneck: the Fourier solution to modular addition
needs only a handful of frequencies, so a few directions out of `2*embed_dim`
carry it. `W2` does not — it is the readout onto `p` classes, and squeezing it
through `r < p` directions caps how many classes can be told apart at all.

That asymmetry is what makes the low-rank knob usable as a *dimension* knob.
Restricting `W1` alone, or `W1` together with a `W2` rank close to its own
bottleneck `min(hidden_dim, p)`, keeps grokking; driving `W2` far below `p`
turns every candidate into `no_generalization` under screening. Hence the
default `rank_targets = ("W1",)`, and a curated suite held under a hard
**200-parameter ceiling** (`gaab.suite.DEFAULT_MAX_VARIABLES`): **32 of 36
problems are low-rank**, and every problem is 121–199 parameters — where a
dense parameterization of the same task runs to four figures.

### How low can `p` go?

`p = 3` is *allowed* — the data layer, the grid, and the screener all accept it
— but a prime that small does not **grok**: the `p = 3` table has only 9 pairs
and the `p = 5` table 25, too little held-out data for a delayed transition to
exist, and both come back `no_generalization` (best held-out accuracy 0.75 and
0.60). Where the floor sits depends on how long you screen. At 20k reference
epochs `p = 7` only ever generalized *immediately* (no grokking gap); tripling
the horizon to **60k epochs** exposes genuinely delayed `p = 7` transitions, and
the shipped suite now reaches down to **`p = 7`** (3 problems), with the bulk at
`p ∈ {11, 13, 17}`. The smaller primes stay available for studying the
pre-grokking regime.

The default activation is **quadratic**. Gromov (2023) showed a two-layer MLP
with a quadratic activation has closed-form Fourier solutions for modular
addition, and it is by far the most reliable way to make a network *small
enough for CMA-ES* grok; `relu`, `tanh` and `gelu` are also available.

The IOH **instance** id selects which pairs are held out — the same role
instances play for BBOB.

## Filtering: which problems actually grok

A landscape belongs in this benchmark only if delayed generalization is
achievable on it. That is a property of the problem, not of the optimizer under
test, so it is established once with a gradient-based reference (full-batch Adam
on the exact same objective, using the analytic gradients in `model.py`):

```bash
python scripts/screen_suite.py --preset default --jobs 10 --plots
```

Four candidate grids ship. Three sit under the 200-parameter suite ceiling;
`large` deliberately lifts it to stress-test the large-scale optimizers on the
grokking task itself, and is never screened into the shipped suite:

| preset | dimensions | `p` | what it is for |
|---|---|---|---|
| `quick` | 149–187 | 11 | smoke tests |
| `default` | 57–200 | 3–17 | the shipped suite (≤ 200 params) |
| `full` | 53–200 | 3–23 | a wider study, same ceiling |
| `large` | 573–3371 | 17–23 | **above the ceiling**, for `lm_cma_es` / `lm_ma_es` |

The `reg_strength` grid targets the **weak-regularization regime** — `0.0001,
0.001, 0.01, 0.5` (plus `0.25` in `full`), with everything `≥ 1.0` dropped,
since strong regularization over-damps a sub-200-parameter network and screens
out. The shipped screen runs **60k reference epochs** (`--steps`, three times
the original default) because weak regularization groks *late* — most survivors
only cross the threshold in the last third of that horizon, so a shorter screen
would miss them.

Each candidate gets a verdict, and four of the five are filtered out:

| verdict | meaning | kept |
|---|---|---|
| `grokking` | `t_gen >= 2 * t_fit` — fits early, generalizes much later | ✅ |
| `early_generalization` | test accuracy arrives with train accuracy; nothing was delayed | ❌ |
| `no_generalization` | train accuracy saturates, test accuracy never follows | ❌ |
| `no_memorization` | the reference never even fits the training set | ❌ |
| `divergence` | the reference blew up | ❌ |

Filtering happens per **(problem, instance)**: a landscape often groks under
some train/test splits and not others, so the curated suite keeps the splits
that grok rather than dropping the whole configuration.
`results/screening.csv` records every verdict, and
`results/screening_summary.csv` records why each candidate was kept or dropped.

## Metrics

**AOCC** (area over the convergence curve), in `[0, 1]`, higher is better:

```
AOCC = 1 - (1/B) * sum_{t=1..B} normalized_log( f_best(t) )
```

Computed exactly, not by quadrature: the best-so-far curve is a step function,
so the integral is a sum over its segments. `aocc_test_accuracy` is the
generalization counterpart — the mean incumbent test accuracy over the budget,
so grokking at the last evaluation scores near zero.

**ERT** (expected running time), the standard restart estimator:

```
ERT(target) = (sum of evaluations of successful runs + B * #failures) / #successes
```

reported both against loss targets and — the metric this benchmark exists for —
against **test-accuracy** targets: the expected running time *to grok*.

## What the suite shows

The shipped `suites/grokking_v1.json` holds **62 problems / 73 problem
instances**, every one **121–199 parameters** (median 183, hard ceiling 200),
screened from a 964-problem candidate grid (`--preset default`, `p ∈ {3..17}`,
3 instances, **60k** reference epochs each). 902 candidates were filtered out —
864 never generalized, 75 never fit the training set, 7 generalized immediately.
**54 of the 62 survivors are low-rank**, and the suite is spread evenly across
the weak-regularization grid — 16 problems at `reg = 0.0001`, 16 at `0.001`, 17
at `0.01`, 13 at `0.5`. That even spread is the point of targeting weak
regularization: even `reg = 0.0001` groks, just with a longer delay (`t_gen`
around 15–20k epochs against 5–10k at `reg = 0.5`), which is why the 60k horizon
is needed to see it. `p` reaches down to 7 (see [above](#how-low-can-p-go)).

A reference campaign — the 7-optimizer roster x 73 problem instances x 5
repetitions x **60,000 evaluations**, `results/benchmark/` — gives:

| optimizer | AOCC (objective) | AOCC (test acc) | best test acc | runs reaching 95% |
|---|---|---|---|---|
| `adam` *(gradient reference)* | **0.666** | **0.497** | 0.582 | **141 / 365** |
| `cma_es` | 0.447 | 0.439 | **0.605** | 110 / 365 |
| `sep_cma_es` | 0.426 | 0.397 | 0.551 | 102 / 365 |
| `lm_ma_es` | 0.424 | 0.373 | 0.535 | 93 / 365 |
| `local_random_search` | 0.294 | 0.240 | 0.396 | 45 / 365 |
| `lm_cma_es` | 0.277 | 0.248 | 0.377 | 24 / 365 |
| `pso` | 0.092 | 0.066 | 0.175 | 0 / 365 |

Expected running time to a generalization target, median over the problems
where a method ever got there (`ert.csv`):

| optimizer | reach 75% | median ERT → 75% | reach 95% | median ERT → 95% |
|---|---|---|---|---|
| `cma_es` | **57 / 62** | **66,590** | 39 | 136,311 |
| `lm_ma_es` | 47 | 68,395 | 36 | 147,576 |
| `sep_cma_es` | 45 | 81,984 | 36 | 116,698 |
| `adam` *(gradient)* | 56 | 92,358 | **48** | **93,356** |
| `local_random_search` | 39 | 135,277 | 18 | 179,865 |
| `lm_cma_es` | 34 | 214,792 | 14 | 258,516 |

Three things make this a benchmark rather than a leaderboard:

* **Given enough budget, black-box search catches the gradient reference — and
  the crossover depends on how hard a target you set.** At the 75% generalization
  target the CMA methods are *ahead* of `adam`: `cma_es` reaches it on the most
  problems (57/62) and fastest (median ERT 67k vs Adam's 92k). Push the target to
  95% and Adam retakes the lead — more problems solved (48) and the shortest
  running time (93k). `cma_es` also reaches the highest *best* held-out accuracy
  of any method (0.605 vs Adam's 0.582). So the ranking flips with the target: a
  leaderboard would have to pick one and hide the rest.
* **Black-box grokking spans the whole weak-regularization grid.** 74 black-box
  runs show the full delayed transition, and they are spread evenly across the
  regularization axis — 19 at `reg = 0.0001`, 17 at `0.001`, 21 at `0.01`, 17 at
  `0.5`. Even the weakest regularization the grid allows produces genuine
  black-box grokking; it just needs the 60k-evaluation horizon to show up, since
  weaker regularization delays the transition further. 8 of the 74 are on the
  `p = 7` problems.
* **BBOB skill does not transfer intact.** `lm_cma_es` is the strongest method
  on the rotated BBOB functions (see [above](#large-scale-methods)) and the
  weakest CMA variant here. Its population success rule reads step-size signal
  from rank changes between consecutive populations, which a rugged
  neural-network landscape scrambles. Showing that is the point of a benchmark
  built from a real learning problem.

![the grokking gap](results/benchmark/figures/grokking_gap.png)

## Adding your own optimizer

```python
from gaab.optimizers import OPTIMIZERS

def my_solver(problem, budget, rng, x0=None, **kw):
    ...                       # call problem(x) at most `budget` times
    return best_x

OPTIMIZERS["my_solver"] = my_solver
```

then `python scripts/run_benchmark.py --optimizer my_solver`. Solvers see only
`problem(x)`, `problem.bounds` and `problem.meta_data`; the one exception is the
`adam` reference, which uses the analytic gradients on purpose and is reported
separately as the ceiling black-box methods are measured against.

Every candidate is clipped into the problem's box before evaluation, so the
search domain is the same for all methods. `python -m gaab.cli optimizers`
lists what is registered.

## References

* Power et al. (2022), *Grokking: Generalization Beyond Overfitting on Small
  Algorithmic Datasets*.
* Liu et al. (2022), *Omnigrok: Grokking Beyond Algorithmic Data*.
* Gromov (2023), *Grokking Modular Arithmetic*.
* Hansen (2016), *The CMA Evolution Strategy: A Tutorial*.
* Ros & Hansen (2008), *A Simple Modification in CMA-ES Achieving Linear Time
  and Space Complexity* (sep-CMA-ES).
* Loshchilov (2015), *LM-CMA: An Alternative to L-BFGS for Large Scale Black-box
  Optimization*.
* Loshchilov, Glasmachers & Beyer (2019), *Large Scale Black-box Optimization by
  Limited-Memory Matrix Adaptation* (LM-MA-ES).
* Wierstra, Schaul, Glasmachers, Sun, Peters & Schmidhuber (2014), *Natural
  Evolution Strategies* (separable NES).
* de Nobel et al. (2024), *IOHexperimenter: Benchmarking Platform for Iterative
  Optimization Heuristics*.
