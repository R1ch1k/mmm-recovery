# Pre-registration — mmm-recovery

**Author:** Richik Mandal
**Written:** 2026-08-05
**Status:** COMMITTED BEFORE ANY GRID RUN. No results existed when this file was written.

This document fixes the hypotheses, conditions, metrics, thresholds and kill criteria
before any experiment is executed. Any change after the commit that contains this file
must be appended to the Deviations Log at the bottom, with a date and a reason. Silent
edits invalidate the study.

---

## 1. Question

Marketing Mix Modelling (MMM) estimates how much each marketing channel caused of a
company's sales, using only aggregate weekly spend and sales data. It is used to allocate
real budgets. Its estimates are almost never validated, because the counterfactual
("what would sales have been without TV?") is unobservable in the field.

In simulation the counterfactual is computable. This study asks:

**RQ1 (recovery).** Under conditions that resemble a real client's data, does MMM recover
the true incremental contribution and marginal ROAS of each channel?

**RQ2 (decision).** Does the budget allocation implied by the MMM estimate outperform the
status quo when evaluated against the true response surface — and by how much does it
fall short of the true optimum?

**RQ3 (fabrication).** When a channel with exactly zero true effect is present, does MMM
assign it credit, and does the resulting allocation route real budget into it?

RQ2 is the headline. RQ1 is the mechanism. RQ3 is the finding a marketing team can act on.

### Positioning against prior art

This is **not** a novelty claim. The failure modes tested here are documented:

- Chan, D. & Perry, M. (2017), *Challenges and Opportunities in Media Mix Modeling*
  (Google) — selection bias, funnel effects, and an explicit recommendation to evaluate
  MMM via simulators.
- Jin, Y., Wang, Y., Sun, Y., Chan, D. & Koehler, J. (2017), *Bayesian Methods for Media
  Mix Modeling with Carryover and Shape Effects* — reports bias at the typical sample size
  of a couple of years of weekly national data. Condition C2 is a deliberate replication.
- Zhang, S. & Vaver, J. (2017), *Introduction to the Aggregate Marketing System Simulator*;
  open-sourced as the R package `google/amss`, which generates simulated marketing data
  together with ground-truth ROAS and mROAS. This project is a Python-native descendant.

The contributions claimed here are narrow and specific:

1. Ground truth defined by **intervention on the generating process**, not by reading
   coefficients — so the target is the causal estimand, not a model parameter.
2. **Decision-level metrics** (allocation regret, beats-status-quo, placebo spend share)
   rather than parameter recovery alone.
3. A **pre-registered** condition grid with thresholds fixed in advance, run against a
   current-generation Bayesian implementation (Meridian, 2025–26) rather than the tooling
   available when the source literature was written.

Any claim beyond these three is overclaiming and must be cut in review.

---

## 2. Generating process (the truth)

Weekly national-level data. Sales are additive in media contribution on top of a
multiplicative baseline:

```
sales_t = baseline_t + Σ_c contrib_{c,t} + ε_t

baseline_t   = B0 · (1 + τ·t/T) · season_t · exp(γ · d_t)
contrib_{c,t} = β_c · sat_c( adstock_c( spend_{c,t} ) )
```

- `d_t` — latent demand: AR(1), φ_AR = 0.8, plus an annual sinusoid. **Never observed by
  the estimator.** Under C0 its coefficient γ = 0, so it is absent entirely.
- `season_t` — annual seasonality, 2 Fourier pairs at period 52. Observable in principle;
  the estimator is given Fourier terms as controls.
- `ε_t` ~ N(0, σ), σ = 3% of mean sales.
- `B0` = 1,000 (£k/week), τ = 0.15 over the full span.

Media is calibrated so total media contribution ≈ 25% of total sales — within the range
normally reported in practice.

### Channel truths

| channel | λ (decay) | α (Hill shape) | κ (half-saturation, £k) | β (max weekly contribution, £k) | mean spend (£k/wk) |
|---|---|---|---|---|---|
| tv      | 0.70 | 1.8 | 60 | 220 | 55 |
| video   | 0.45 | 1.2 | 30 | 120 | 28 |
| search  | 0.10 | 0.9 | 18 |  90 | 20 |
| social  | 0.30 | 1.0 | 22 |  70 | 18 |
| ooh     | 0.60 | 2.2 | 25 |  60 | 12 |
| **placebo** | 0.30 | 1.0 | 20 | **0.0** | 15 |

The placebo channel is present only in C5, C6, C7. Its true contribution is exactly zero
by construction, not approximately zero.

### Transforms

- **Adstock (correct form):** geometric, `x̃_t = x_t + λ·x̃_{t−1}`, normalised so the
  kernel sums to 1.
- **Adstock (misspecified truth, C4):** Weibull-PDF kernel with peak at lag 2, shape 2.0,
  truncated at lag 12 and normalised.
- **Saturation (correct form):** Hill, `x̃^α / (x̃^α + κ^α)`.
- **Saturation (misspecified truth, C4):** logistic `1/(1 + exp(−(x̃ − κ)/s))`.

Order is adstock → saturation, matching Robyn and Meridian convention.

### Spend generation

Base spend per channel is log-normal around its mean with a quarterly budget cycle.
Three knobs modify it:

- **ρ (collinearity):** a shared budget factor injected into every channel's spend to
  induce a target pairwise Pearson correlation.
- **φ (endogeneity):** spend scaled by `exp(φ · d_t)`, i.e. the marketer spends more when
  latent demand is high. This is the confounding channel.
- **placebo coupling:** placebo spend correlated at 0.8 with `search` spend and with the
  seasonal term.

---

## 3. Ground truth is computed by intervention

This is the load-bearing design decision. Truth is never read off a parameter.

- **True incremental contribution of channel c** = `sales(spend) − sales(spend with
  spend_c := 0)`, evaluated noiselessly (ε excluded) with all other inputs and the same
  seed held fixed. This is `do(spend_c = 0)`, implemented as a function call.
- **True mROAS of channel c** = `[sales(1.1·spend_c) − sales(spend_c)] / (0.1 · Σ_t spend_{c,t})`,
  noiseless.
- **True optimal allocation** = the per-channel spend multipliers `m_c` maximising
  noiseless total sales subject to `Σ_c m_c · Σ_t spend_{c,t} = total budget`, with
  `m_c ∈ [0, 3]`. Weekly spend patterns are preserved and only scaled — this mirrors how
  budgets are actually reallocated. Solved with SLSQP from multiple starts; the solution
  must be verified to beat the status quo.

Because the DGP is a pure function of its inputs, all three are exact, not estimated.

---

## 4. Estimator under test

### Primary: `RidgeMMM` (in-house)

A faithful reduced form of the frequentist MMM family. Geometric adstock → Hill
saturation → ridge regression with non-negative media coefficients, plus intercept,
linear trend, and 2 Fourier pairs at period 52 as controls.

- Hyperparameters (λ, α, κ per channel, plus ridge penalty) selected by random search,
  N = 200 fixed-seed draws from bounded ranges, scored by expanding-window time-series CV
  (3 folds) on RMSE.
- Uncertainty by moving-block bootstrap of residuals, block length 13 weeks, B = 200
  replicates, refitting coefficients only at the selected hyperparameters.
- **The estimator never sees `d_t`.** It sees spend, sales, trend and Fourier terms —
  exactly what a real analyst has.

The bootstrap holds hyperparameters fixed and therefore understates total uncertainty.
This is a known limitation and must be stated wherever coverage is reported.

### Anchor: Google Meridian

Run on C0, C3, C6, C7 only, 10 seeds each, default priors. Purpose is verify-not-trust:
to establish that any conclusion is a property of MMM as a method rather than an artefact
of `RidgeMMM`.

**Pre-committed handling of disagreement:** if Meridian and `RidgeMMM` reach different
verdicts on any condition, that disagreement is reported as a primary result, not
reconciled away or dropped. If Meridian's priors rescue a condition where `RidgeMMM`
fails, the correct interpretation — the prior is supplying information the data does not
contain — is stated explicitly.

### Excluded: Meta Robyn

Robyn's production path is R (CRAN 3.12.1). The Python port `robynpy` is a self-declared
LLM-translated beta. A failure observed there would be ambiguous between the method and
the translation, so it cannot serve as evidence. Named in limitations, not run.

---

## 5. Conditions

Every condition is C0 with exactly one knob moved, except C7.
**200 seeds** per condition; **500 seeds** for C5, C6 and C7 to tighten the false-positive
rate estimate.

| ID | Name | Change from C0 | Levels |
|---|---|---|---|
| **C0** | Clean | — (T=520, ρ≈0, φ=0, γ=0, correct forms, no placebo) | — |
| **C1** | Collinear spend | ρ | 0.5, 0.8, 0.95 |
| **C2** | Short series | T | 260, 156, 104 |
| **C3** | Demand confounding | γ=0.5, φ | 0.3, 0.6 |
| **C4** | Misspecification | true adstock Weibull(peak=2), true saturation logistic | — |
| **C5** | Placebo, orthogonal | + placebo channel, spend independent | — |
| **C6** | Placebo, correlated | + placebo channel, corr 0.8 with search and season | — |
| **C7** | Realistic composite | T=104, ρ=0.7, γ=0.5, φ=0.6, correlated placebo, misspecified forms | — |

C0 is a validity gate on the harness, not a finding. C7 is the condition an actual client
has, and the interaction between knobs is expected to be the most informative result.

---

## 6. Metrics

**Rung 1 — parameters.** Relative bias of λ̂, α̂, κ̂ per channel. Reported as a diagnostic
only. Expected to be uninformative about decision quality; that expectation is itself
part of the finding.

**Rung 2 — contributions.**
- Relative bias of estimated incremental contribution, per channel and for total media.
- Relative bias of estimated mROAS.
- Empirical coverage of the nominal 90% interval.
- Spearman rank correlation between true and estimated channel contribution.

**Rung 3 — decisions.**
- **Allocation regret** = `(S_opt − S_model) / (S_opt − S_status_quo)`, where each `S` is
  noiseless true sales under that allocation. Reported as a percentage of achievable lift.
  Values above 100% mean the model's advice is worse than doing nothing.
- **Beats-status-quo** = fraction of runs where `S_model > S_status_quo`.
- **Placebo spend share** = fraction of total budget the model's recommended allocation
  assigns to the zero-effect channel.
- **Placebo flag rate** = fraction of runs where the placebo channel's 90% interval on
  contribution excludes zero.

---

## 7. Pass thresholds (fixed in advance)

A condition PASSES only if all applicable gates hold.

| Gate | Threshold |
|---|---|
| G1 Contribution | median across channels of \|relative bias\| < 20% |
| G2 Coverage | empirical coverage of the 90% interval ≥ 80% |
| G3 Rank | median Spearman ρ(true, estimated contribution) ≥ 0.80 |
| G4 Regret | median allocation regret < 20% of achievable lift |
| G5 Robustness | beats-status-quo in ≥ 90% of runs |
| G6 Placebo | flag rate ≤ 10% **and** median placebo spend share ≤ 2% |

G6 applies only to C5, C6, C7.

---

## 8. Directional predictions

Recorded so the study can be wrong. Scored honestly in the results section.

| Condition | Prediction |
|---|---|
| C0 | Passes all gates |
| C1 (ρ=0.95) | G1 and G3 fail; total-media contribution bias stays under 10%; G4 may still pass |
| C2 (T=104) | G2 fails first (intervals too narrow); G1 marginal; G4 marginal |
| C3 (φ=0.6) | Every channel biased **upward**; total media contribution overstated by more than 30%; G1 fails |
| C4 | Rung 1 badly wrong while G1 and G4 pass — the diagnostic/decision divergence |
| C5 | Passes G6 comfortably (flag rate < 10%) |
| C6 | **Fails G6** — flag rate above 30%, placebo spend share above 5% |
| C7 | Fails G1–G4; regret above 50%; G5 in genuine doubt |

The single prediction I am least confident in is C4 passing G4. If misspecified functional
forms turn out to be decision-irrelevant, that is a genuinely useful reassurance for
practitioners and should be reported as such.

---

## 9. Kill criteria and protocol

**K1 — Harness validity.** If C0 fails any of G1–G5, the harness is wrong. C1–C7 must not
be reported until C0 passes. No exceptions, no partial publication.

**K2 — Null result.** If every condition including C7 passes, the study publishes
unchanged, with the title and framing adjusted to report that MMM is more robust than its
critics claim. A null is a result, not a failure. This project does not require MMM to
break in order to be worth publishing.

**K3 — Anti-strawman.** No condition may be reported as failing until three legitimate
remedies have been attempted and also failed:

1. Tighter regularisation plus enforced non-negativity.
2. Shrinkage toward an informative prior ROAS (the ridge analogue of Meridian's priors).
3. Giving the model a noisy proxy for latent demand — a corrupted observation of `d_t`
   at signal-to-noise ratios of 0.3, 0.6 and 0.9.

Both raw and remedied results are reported. Remedy 3 doubles as a practical finding:
*how good does a demand proxy have to be before MMM becomes trustworthy?*

**K4 — Independent verification.** Every headline number must be reproduced by an
independent adversarial pass (Codex) before publication, recomputing from the raw grid
output rather than reading the summary tables.

---

## 10. Explicitly out of scope for v1

Geo-hierarchical models; reach and frequency modelling; comparison with multi-touch
attribution; lift-test calibration; multiple DGP families; anything requiring an API key
or a network call at run time. Lift-test calibration is deliberately reserved as the
follow-up study.

---

## 11. Deviations log

| Date | Change | Reason |
|---|---|---|
| — | none yet | — |
