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
  PRODUCT.md                  the dashboard's surface brief (Impeccable schema); surface mode READ,
                              the three hard constraints, and detector findings already declined
  README.md                   written LAST, after results exist
  pyproject.toml
  Makefile                    reproduce, test, lint, report
  src/mmm_recovery/
    __init__.py
    transforms.py             adstock + saturation, correct and misspecified variants
    dgp.py                    simulator; pure function of (params, seed)
    truth.py                  do()-style interventions, mROAS, optimal allocation
    estimator.py              RidgeMMM
    meridian_anchor.py        optional, C0/C3/C6/C7 only
    sweep.py                  D26 spend-variation sweep; owns the gate arithmetic
    robustness.py             D33/D35 optimiser-bound and guardrail checks
    flighting.py              D34 flighted-spend validity check on C0
    plateau.py                D39 identification-plateau grid; regenerated, does not reproduce
    report.py                 the dashboard; plotly vendored, byte-deterministic. Emits EIGHT
                              figure divs: 2 figures × 2 colour modes × 2 layouts (a plotly
                              subplot grid cannot reflow with CSS, so the phone layout is a
                              separately rendered figure, switched at 900px)
    conditions.py             NOT BUILT — K1 fired before Step 5
    metrics.py                NOT BUILT — the gates live in sweep.py instead
    experiment.py             NOT BUILT — no degradation grid was ever run
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
Tests: ~~on a noiseless C0 draw with T=520, recovers contributions within 5%~~ — this is the
strict xfail described below; bootstrap intervals have positive width; the design matrix
provably excludes `d_t`.
The control block is **10 columns, not 6** (D22): intercept, trend, 2 Fourier pairs, and
trend × each Fourier term. §2's baseline is a *product*, so without the interactions the
controls cannot span it. Every column is still a function of the week index alone.

> **STOP — K1 has fired. Steps 5 and 6 are blocked (D21, D23).**
> C0 fails all five gates at 200 seeds: G1 0.540, G2 0.417, G3 0.650, G4 2.355, G5 0.200.
> Under K1, C1–C7 must not be reported and the grid must not run. This is a decision for the
> pre-registration's author, not something to code around. **Do not amend K1, G1 or G2, and do
> not "improve the search" as a fix** — better optimisation makes recovery measurably *worse*,
> which is the K3 anti-strawman result.

**Step 5 — `conditions.py` + `metrics.py`.** Conditions exactly as tabulated in the
pre-registration. All six gates implemented as pure functions returning pass/fail plus the
underlying number.
Tests: each gate fires correctly on hand-built pass and fail cases.

**Step 6 — `experiment.py`.** Grid runner. Parallel over seeds, deterministic regardless of
worker count or completion order. Writes tidy long-format `results/grid.csv`, one row per
(condition, level, seed, channel, metric).
**Gate: run C0 first and confirm K1 before running anything else.**

**Step 7 — K3 remedies** for any failing condition, then the Meridian anchor.
The anchor was pulled forward to Step 4 because D21 required it before any structural claim.
`meridian_anchor.py` exists and C0 is done (D24); C3, C6 and C7 are not. The `[meridian]`
extra is installed — `uv sync --extra meridian`. Budget **186 s per seed**, CPU only.

**Step 8 — `report.py`, `docs/WHEN-TO-TRUST-YOUR-MMM.md`, `README.md`.** All three exist. The
dashboard needs the `[report]` extra (`uv sync --extra report`) and `make report` passes it; plotly
is vendored into the HTML at build time and is never a run-time dependency of the grid.

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
- **A published number whose harness was never committed.** D39 regenerated the plateau sweep and
  it did not reproduce: 639 of 780 against a published 177, because the original predated D22 and
  described the superseded six-column control block. It had been quoted as the current mechanism
  for weeks and nothing failed. The Nelder–Mead diagnostic is still in that state. **Treat any
  figure with no artefact behind it as unverified**, and prefer regenerating it to citing it.
- **A chart that renders blank while every structural check passes.** The first dashboard build put
  the vendored `plotly.js` at the end of `<body>`, after the `Plotly.newPlot` calls it defines.
  Valid HTML, correct data, four silent `Plotly is not defined` errors, two empty panels. Nothing
  short of opening it in a browser would have caught it — so open it, every time.
- **An axis whose units contradict its own label.** The same build divided contributions by 1000
  and titled the axis £k, so it read 50 where the annotation beside it read £51,989k.
- **A panel that does not contain its own subject.** The noiseless plateau panel exists to show the
  truth beating the whole grid by 2,590×, and on autorange the log axis started at the *best
  competitor* (0.0518) so the truth (0.00002) was clipped off the bottom. Both panels then read as
  the same point cloud twice and the contrast — the mechanism — was invisible. **Nothing fails when
  a mark falls outside a range**, so the axis floor is asserted in `tests/test_report.py`.
- **A plotly annotation on a log axis takes its y in log10 units.** Passing the raw score put the
  "1% band" label four decades above its own line. Shapes (`add_hline`) take data units; annotations
  do not. The two look identical in the source.
- **Mixing label strings on one categorical axis.** The phone layout shortens the arm names; when
  only *some* traces were switched over, the axis silently gained eight categories instead of four,
  the explicit range clipped to the first four, and the bottom two arms **drew nothing at all** — no
  error, no warning, half the figure's data absent. Every trace on an axis must use one label set.
- **A scanner that cannot read the file type returns the same empty result as a clean pass.** The
  Impeccable detector reports `[]` for `.py`; its extensions are `.astro .eex .ex .heex .html .jsx
  .md .svelte .tsx .vue`. Since `report.py` carries the page's entire CSS and copy in an f-string,
  that "clean" scan covered none of it. Render to a scratch HTML file and scan that. Related: on a
  page with a 5.6 MB inlined bundle, a whole-file scanner attributes the **library's** source to
  your design — check the reported line against where the page's own `<style>` block starts.
- **Comparing a computed statistic to a literal threshold with a bare `>=`.** Spearman's ρ on five
  channels is `1 - 6·Σd²/120`, and scipy evaluates that to 0.7999999999999999889 — one unit in the
  last place *below* the double nearest to `0.80`. A median ρ of exactly 0.8 was therefore reported
  as **failing** a `>= 0.80` gate. Use `sweep.passes`, which carries 1e-9 of slack; that is nine
  orders of magnitude below any threshold's meaningful precision, so it cannot rescue a real
  failure. A study about careful measurement cannot afford to mislabel a gate on a representation
  artefact.
- **Quoting a decision metric without naming its action space.** §3 lets the optimiser zero a
  channel or take it to 3× spend. Under a two-sided ±30% guardrail the worse-than-nothing rate
  falls from 80% to 24% (D37). Both are true; neither travels alone.
- **Counting one bound and concluding about both.** D33 first claimed the recommendations were
  "interior solutions" on the strength of an upper-bound counter, while the binding constraint was
  the lower one. If a claim is about the boundary, count every boundary.
- **Quoting a minimum over noisy tries as though it were a capability.** "The best of N runs
  reached X" is selection on noise, not a configuration anyone can choose. This has caused two
  wrong diagnoses here, the second after the first was written down. Carry N and the spread
  with every extremum, and check the winner's rank on the criterion actually used — in the
  first instance the low-error draws ranked 57th–157th of 200 on it. Corollary: a lone number
  that contradicts your own headline is more likely an extremum than a refutation.
- **Inferring an objective's global shape from points that never went near its optimum.** The
  cheap decisive test is to *start the optimiser at the known truth* and see whether it stays.
  Here it walks away — CV 3.63160 → 3.06272 while bias goes 2.4% → 57.3% — which distinguishes
  non-identification from search failure in one run, where a budget sweep gave only noise.
