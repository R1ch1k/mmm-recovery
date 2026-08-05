# CLAUDE.md — mmm-recovery

Working context for Claude Code. Read `PREREGISTRATION.md` first; it is the specification
and it is binding.

---

## What this is

A pre-registered measurement-validity audit of Marketing Mix Modelling. We simulate a
world where the true causal effect of each marketing channel is known, fit MMM to what a
real analyst would see, and measure whether the resulting **budget decision** is any good.

Headline framing: **attributable ≠ incremental**. Third in a series with `priced-in`
(detectable ≠ tradeable) and `marketplace-mispricing` (detectable ≠ actionable).

---

## Non-negotiable rules

1. **No grid run before `PREREGISTRATION.md` is committed.** The timestamp is the evidence.
   If you find yourself running experiments to decide what the thresholds should be, stop.
2. **Ground truth comes from intervention, never from parameters.** True contribution is
   `sales(spend) − sales(spend_c := 0)` evaluated on the DGP. Do not compare `β̂` to `β`
   and call it recovery.
3. **The estimator never sees the latent demand series.** It gets spend, sales, trend and
   Fourier terms. Nothing else. Leaking `d_t` into the design matrix invalidates C3 and C7.
4. **Determinism.** Every stochastic step takes an explicit seed. `make reproduce` must
   produce byte-identical output on a clean checkout. No wall-clock, no unseeded RNG, no
   dict-ordering dependence in output.
5. **Do not tune the estimator to make a condition fail.** If a condition fails, run the
   K3 anti-strawman remedies before recording it as a failure. A rigged failure is worse
   than no result.
6. **No results in `README.md` until the grid has actually run.** Placeholder numbers have
   a way of surviving into published work.
7. **No network calls at run time.** No API keys, no data downloads. The whole point of
   this design is that it reproduces anywhere.

---

## Stack

Python 3.12. `uv` for env and lockfile.

Runtime: `numpy`, `scipy`, `pandas`.
Dev: `pytest`, `ruff`, `mypy` (strict), `pytest-cov`.
Reporting: `plotly` (vendored into a self-contained HTML file — zero external references,
byte-deterministic output, same approach as `pitchvalue`).
Optional extra `[meridian]`: `google-meridian` + `tensorflow-probability`. Must be an
optional dependency group — the core grid runs without it.

Simplest thing that works. Stdlib and numpy before new dependencies. No speculative
abstractions, no plugin architectures, no config framework beyond dataclasses.

---

## Layout

```
mmm-recovery/
  PREREGISTRATION.md          binding spec, already written
  CLAUDE.md                   this file
  README.md                   written LAST, after results exist
  pyproject.toml
  Makefile                    reproduce, test, lint, report
  src/mmm_recovery/
    __init__.py
    transforms.py             adstock + saturation, correct and misspecified variants
    dgp.py                    simulator; pure function of (params, seed)
    truth.py                  do()-style interventions, mROAS, optimal allocation
    estimator.py              RidgeMMM
    conditions.py             C0-C7 as dataclasses
    metrics.py                three metric rungs
    experiment.py             grid runner, writes results/grid.csv
    report.py                 self-contained HTML dashboard
    meridian_anchor.py        optional, C0/C3/C6/C7 only
  tests/
  results/                    grid.csv, dashboard.html — committed
  docs/
    WHEN-TO-TRUST-YOUR-MMM.md one page, no equations, for a marketing reader
```

Repo and distribution name are `mmm-recovery`; the import package is `mmm_recovery`.
Hyphens are illegal in Python identifiers, so the underscore form is the importable one.
Do not collapse it to `mmmrecovery`.

---

## Build order

Each step ends with tests passing, `ruff` clean, `mypy --strict` clean. Do not start the
next step until the current one is green.

**Step 1 — `transforms.py`.** Geometric adstock, Weibull adstock, Hill saturation,
logistic saturation. Pure functions on numpy arrays.
Tests: unit impulse through geometric adstock decays at exactly λ; kernels sum to 1;
Weibull kernel peaks at lag 2; Hill is monotone increasing and concave for α ≤ 1;
saturation maps 0 → 0.

**Step 2 — `dgp.py`.** `simulate(params, seed) -> SimResult` holding spend, sales, and the
noiseless component series. Pure: same inputs, same outputs, always.
Tests: determinism under repeated calls; total media contribution lands near the 25%
target under C0 defaults; induced spend correlation matches the requested ρ within
tolerance; setting φ > 0 produces positive correlation between spend and `d_t`.

**Step 3 — `truth.py`.** Intervention API, mROAS, SLSQP optimal allocation.
Tests: **zeroing the placebo channel leaves noiseless sales bit-identical** (this is the
single most important test in the repo — it proves β=0 really means zero);
contributions are non-negative and sum below total sales; the optimiser's solution beats
status quo; ~~multi-start solutions agree within 0.1%~~ — **superseded by D17.** That
requirement was wrong: the surface is non-concave, so requiring the starts to agree is
requiring convexity. The standard is that the returned optimum matches a 64-start reference.
Do not re-add the agreement test.

**Step 4 — `estimator.py`.** RidgeMMM: transform, expanding-window CV, random search over
hyperparameters, non-negative ridge, moving-block bootstrap intervals.
Tests: on a noiseless C0 draw with T=520, recovers contributions within 5%; bootstrap
intervals have positive width; the design matrix provably excludes `d_t`.

**Step 5 — `conditions.py` + `metrics.py`.** Conditions exactly as tabulated in the
pre-registration. All six gates implemented as pure functions returning pass/fail plus the
underlying number.
Tests: each gate fires correctly on hand-built pass and fail cases.

**Step 6 — `experiment.py`.** Grid runner. Parallel over seeds, deterministic regardless of
worker count or completion order. Writes tidy long-format `results/grid.csv`, one row per
(condition, level, seed, channel, metric).
**Gate: run C0 first and confirm K1 before running anything else.**

**Step 7 — K3 remedies** for any failing condition, then the Meridian anchor.

**Step 8 — `report.py`, `docs/WHEN-TO-TRUST-YOUR-MMM.md`, `README.md`.** In that order.

---

## Performance budget

The full grid must complete in under 30 minutes on a laptop CPU. If it doesn't, reduce the
random-search draws from 200 before reducing seed counts — seeds buy statistical
precision on the false-positive rate, random-search draws buy very little after the first
hundred. Never cut seeds below 200 for the placebo conditions.

---

## Style

- Type hints everywhere; `mypy --strict` passes.
- Dataclasses for parameter objects. No dicts-as-config.
- Docstrings state units (£k per week) and shapes.
- Numbers in prose carry their uncertainty. "23.4%" alone is not a result.
- Commits are small and describe the gate they close.

---

## Recording a spec-vs-reality gap

When `PREREGISTRATION.md` asserts something the code measurably cannot satisfy, encode it as
`pytest.mark.xfail(strict=True)` with the measurement in the `reason` string. Not a comment,
not a loosened assertion, not a deleted test.

The suite stays green, the gap prints on every `pytest -rx` run with its numbers, and it
becomes an **error** the moment it silently closes. A loosened assertion is how a spec
violation stops existing.

`pytest -q -rx` is therefore the list of open decisions: each reason carries the diagnosis, the
numbers and the recommended fix. Delete an xfail when the decision lands and is logged as a
Deviation — never because it has become inconvenient.

---

## Failure modes to watch for

- **Silent leakage** of `d_t` or of any true parameter into the estimator. Assert against it.
- **Optimiser convergence failures** in `truth.py` reported as low regret. Check the
  SLSQP exit status on every call; a failed solve must raise, not return a number. Re-verify
  feasibility and the bounds independently — a solver reporting success is a claim, not
  evidence.
- **An unscaled optimiser objective.** Scale the objective and every constraint to O(1) before
  handing them to SLSQP. Media contribution is of order 1.3e5, which puts a default
  finite-difference increment at 6e-9 relative — inside the summation noise of a T×C reduction.
  It presents as `status 8, "Positive directional derivative for linesearch"`, which reads like
  a modelling error and is not one. Supply an analytic gradient where one exists.
- **Tuning the DGP so a threshold passes.** The generator's undocumented knobs (seasonal
  amplitudes, spend volatility, quarterly amplitude, demand SD) are not free parameters once
  results exist. If a gate fails because of one, report it; do not turn it.
- **Bootstrap intervals that don't move** across conditions — usually means the block
  bootstrap is resampling the wrong axis.
- **Regret above 100%** silently clipped. Do not clip it. Worse-than-nothing is the most
  interesting outcome in the study.
- **`except Exception` swallowing real bugs** as condition failures. Let it crash.
