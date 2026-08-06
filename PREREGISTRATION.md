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

All entries dated. Each records a decision made after this document was first committed,
the reason, and whether it tightens, loosens, or merely specifies the study.

### D1 — 2026-08-05 — C6 placebo correlation reduced to 0.6 and given levels

**Specified.** §5 required the placebo to correlate 0.8 with both search spend and the
seasonal term. That is jointly impossible: the 3×3 correlation matrix for
(placebo, search, season) with two 0.8 off-diagonals is not positive semi-definite unless
search and season themselves correlate at least 0.28. Confirmed by eigenvalue check.

Two fixes were available. Requiring a search↔season correlation was rejected because it
moves two knobs at once and breaks the one-knob-at-a-time design that makes C6 comparable
to C0.

C6 now uses equal targets against search and season at levels **0.3, 0.45, 0.6**, with
search↔season left at its C0 value. The ceiling for equal targets under uncorrelated
search and season is 1/√2 ≈ 0.707, so 0.6 keeps a feasibility margin. Adding levels turns
a point estimate into a dose-response curve on the placebo flag rate, which is strictly
more informative than the original single condition.

**Effect on §8:** the C6 prediction (flag rate above 30%) was written against an infeasible
specification. It is retained unchanged and now applies to the 0.6 level. Confidence in it
is lower than when written, and that is recorded rather than corrected.

### D2 — 2026-08-05 — Logistic saturation scale fixed at s_c = κ_c / 4

**Specified.** §2 named the logistic form but never gave s, which alone determines whether
C4's misspecification is severe or cosmetic. s_c = κ_c/4 gives a full S-curve across
[0, 2κ] and reuses the existing Hill κ column, so no new parameters enter.

It also keeps the optimiser out of trouble. `expit` returns exactly 1.0 in float64 once its
argument exceeds roughly 36.7, so a logistic with too small an s contains a genuinely flat,
zero-gradient plateau. At s = κ/4 that plateau begins far above the largest spend any
channel reaches at the optimiser's upper bound m_c = 3. `truth.py` must assert this rather
than assume it.

### D3 — 2026-08-05 — Logistic saturation is zero-anchored

**Corrected.** As literally specified, g(0) = 1/(1 + exp(κ/s)) > 0, so every channel would
emit sales at zero spend — 1.8% of β at s = κ/4, about £3.96k/week for TV. The offset
cancels in the intervention difference of §3 but corrupts the 25% media-share calibration
and the baseline, and contradicts the 0 → 0 requirement in CLAUDE.md.

Saturation is now `(g(x) − g(0)) / (1 − g(0))`, which maps 0 → 0, preserves the range
[0, 1], and keeps the S-shape. This matters for what C4 tests: the intended
misspecification is curve *shape*, and an additive offset would have made C4 measure two
different things at once.

### D4 — 2026-08-05 — β is fixed at the §2 table for every condition

**Specified.** The 25% media-share target is a property of C0 only, asserted in C0's tests
and nowhere else. Recalibrating β per condition to hold the share would make ground truth
condition-dependent and confound every cross-condition comparison — C1 against C4 would
stop being comparable.

Media share will therefore drift: upward under C3, where spend rises with demand, and by an
unknown amount under C4, where the transforms differ. That drift is expected, reported per
condition as a descriptive statistic, and not corrected.

### D5 — 2026-08-05 — ρ targets spend levels, with a stated tolerance

**Specified.** §2 did not say whether ρ applies to levels or logs. Levels, because that is
what a practitioner computes and what "channels move together" means operationally; for
log-normal spend the two differ materially, so ρ = 0.95 would otherwise describe two
different datasets.

Equal pairwise ρ across all pairs is not attainable with unequal channel volatilities. The
shared-factor weight is solved numerically per condition against the target, and the
generator asserts **mean pairwise Pearson ρ on levels within ±0.02 of target, with no
individual pair further than ±0.10 from it**.

### D6 — 2026-08-05 — Placebo excluded from G1, G2 and G3; evaluated only by G6

**Specified, with a small loosening recorded honestly.** Relative bias is undefined for the
placebo, whose true contribution is exactly zero. G1–G3 are computed over the five real
channels only.

This has a side effect on G3 that is stated here rather than discovered later. Spearman on
five channels takes 21 distinct values and 0.80 is exactly one of them; on six it is not,
and the effective threshold would have been 0.8286. Excluding the placebo therefore makes
G3 marginally easier than the six-channel reading. The exclusion is justified independently
— the placebo has no defined bias and has its own dedicated gate — but the direction of the
effect is a loosening and is logged as one.

For coherence: for the placebo, the G6 flag rate is exactly one minus the G2-style coverage
of the true value zero. The two are the same measurement, which is a useful internal
consistency check.

### D7 — 2026-08-05 — Two descriptive metrics added; no gate changed

**Neither tightens nor loosens.** Both are reported alongside the gates; neither is one.

1. **Fraction of channel pairs correctly ordered** by estimated contribution. Spearman on
   five channels is coarse; this reads more finely and is easier to explain to a
   non-technical reader. G3's threshold is unchanged.
2. **Placebo spend reduction relative to status quo.** The status quo already routes
   15/148 = 10.14% of budget to the placebo, so G6's ≤ 2% gate demands roughly an 80% cut
   rather than mere restraint. The gate is intentionally that demanding — the correct answer
   is zero — and is unchanged. The reduction ratio is the version a marketing reader
   understands.

### D8 — 2026-08-05 — Geometric adstock normalisation confirmed

**No change; recording the reading.** The recursion in §2 has kernel λ^k summing to
1/(1−λ), so unit-sum normalisation is a scaling by (1−λ), implemented as
x̃_t = (1−λ)·x_t + λ·x̃_{t−1}. This is the Meridian convention rather than Robyn's
unnormalised form, and it is the right one here: without normalisation a larger λ inflates
the adstocked signal and confounds λ with β.

### D9 — 2026-08-05 — B0 corrected so C0 media share is 25.0%

**Corrected.** At B0 = 1000 the realised C0 media share is 18.57%, not the ≈25% §2 claims.
The channel table and the calibration claim were over-determined; β is locked by D4, so B0
is the only lever that moves the ratio without touching ground truth. Contributions are
unchanged in absolute terms.

The correction is applied rather than the document amended to 18.6%, because media share is
a difficulty knob, not a cosmetic one. A smaller share means a weaker media signal against
baseline, which makes recovery harder and failure more likely — the direction that flatters
the interesting result. Running a version accidentally harder than specified and reporting
the resulting failures is what K3 exists to prevent.

B0 is solved numerically and C0's realised share asserted within ±0.05pp. Note a
discrepancy to resolve: the analytic solve from 18.568% gives B0 = 684.0, but 680.8 was
reported. Baseline is linear in B0, so the two should agree; if the gap survives
re-derivation, something in the calibration is not linear in B0 and must be identified
before proceeding.

**Resolved 2026-08-05.** Not a bug. Baseline is exactly linear in B0 and contributions are
bit-identical across it. The two closed forms inverted different inputs — seed 0 alone gives
680.82, the 30-seed mean gives 684.05 — and since a mean of ratios is not a ratio of means,
neither is the root of the quantity that matters. B0 = **682.5179**, the numerical root of
mean-share = 0.25 over seeds 0–199, the set C0 actually runs. Realised share 25.0000% in
sample, 24.9951% out of sample on seeds 200–399. Committed as a constant with a test that
re-runs the solver to prove it is derived rather than asserted.

### D10 — 2026-08-05 — C7 placebo coupling reduced to 0.45

**Corrected.** A coupling of 0.6 is unreachable on roughly 1.8% of seeds, so C7 would fail
to construct nine times in a 500-seed run. The generator raises rather than clips, which is
the correct behaviour and is what surfaced this; a clipping implementation would have
produced a C7 silently claiming a coupling it did not have.

0.45 rather than 0.5 because it is C6's middle level, which makes C7 decomposable against
C6[0.45]: the interaction can be isolated only when each knob sits at a level also run
alone.

**Known limitation:** C7 remains only partially decomposable. Its ρ = 0.7 matches no C1
level (0.5 / 0.8 / 0.95). Moving it to 0.8 was rejected because that would make C7 harder,
again the unsafe direction. T = 104, φ = 0.6 and the misspecified forms all match their
single-knob levels.

### D11 — 2026-08-05 — D5's per-pair tolerance replaced with a sampling-noise rule

**Corrected.** The fixed ±0.10 per-pair bound is tighter than sampling noise at T = 104:
the SE of a single correlation there is (1−ρ²)/√(T−3) = 0.0507, making ±0.10 a two-sigma
bound that 2–5% of pairs must breach by chance. A matched control at the same ρ and T but
none of C7's other knobs breaches at 1.70% against C7's 2.08%, confirming the cause is the
short series rather than the composite.

The per-pair bound is now **4 · (1−ρ_target²) / √(T−3)**. One formula, no magic numbers,
scaling with both T and ρ: ±0.203 at T = 104 with ρ = 0.7, ±0.132 at T = 520 with ρ = 0.5.
A four-sigma breach is about 6e-5 per pair, so the assertion still fires on a real defect.
The mean bound of ±0.02 is unchanged and is hit exactly, since the solver targets it.

A distributional restatement ("≥95% of pairs within ±0.10") was rejected as near-vacuous —
it would merely restate the sampling distribution it was meant to check.

### D12 — 2026-08-05 — Latent demand standardised to SD = 0.25

**Specified.** §2 fixes φ_AR = 0.8 but never the scale of d_t, which leaves γ = 0.5 and
φ = 0.6 dimensionless. d_t is standardised in-sample to SD = 0.25, so a one-SD demand shock
multiplies baseline by 1.13.

Two measured reasons. At SD = 1.0 the baseline swings 0.37× to 2.72× at ±2 SD, which is not
a real client's data, and a devastating C3 built that way would be the rigged failure that
K3 and CLAUDE.md rule 5 forbid. At SD = 1.0, C7 is not constructible at all: φ = 0.6 alone
induces a cross-channel correlation floor of 0.738, above C7's own ρ = 0.7 target. Floors
are 0.178 / 0.456 / 0.636 / 0.738 at SD = 0.25 / 0.5 / 0.75 / 1.0.

This makes C3's confounding weaker than a larger-SD version would, i.e. it makes MMM more
likely to pass — the safe direction.

### D13 — 2026-08-05 — C3's media-share drift is downward; D4's expectation was wrong

**Recorded, not corrected.** D4 anticipated media share drifting upward under C3 where
spend rises with demand. It drifts down: −0.14pp at φ = 0.3 and −0.10pp at φ = 0.6. Mean
spend does rise, by exp(φ²σ_d²/2) ≈ 1.1%, but multiplying spend by exp(φ·d_t) also raises its
variance, and the response curves are concave over most of the observed range, so Jensen
takes back more than the mean adds.

This concerns the DGP's realised media share only. It does **not** revise §8's C3
prediction, which is about the estimator's bias under confounding — a different quantity,
and still expected upward, since spend remains correlated with unobserved demand.

### D14 — 2026-08-05 — C4's Weibull kernel is global; the confound is acknowledged

**Limitation recorded.** §2 states a single Weibull specification, so under C4 and C7 every
channel shares peak lag 2 and shape 2, and per-channel carryover heterogeneity disappears.
C4 therefore tests wrong-shape *and* homogeneous-carryover jointly, not shape alone.

Per-channel Weibull peaks matched to each channel's geometric mean lag were considered and
rejected: search's mean lag of 0.11 weeks is unreachable for a kernel with zero weight at
lag 0, so any mapping would introduce a further arbitrary choice. Instead, **if C4 fails G1
or G4, a C4b diagnostic variant with per-channel peak lags is run** to attribute the failure
between the two causes. If C4 passes, C4b is not run.

### D15 — 2026-08-05 — Endogeneity applied literally, not recentred

**Specified.** Spend is multiplied by exp(φ·d_t) with no recentring, raising mean spend by
1.1% at φ = 0.6. A recentred variant was built and removed on parsimony: it is an extra
assumption not in §2, for a 1.1% effect.

Recorded for transparency: the recentred version also flips the sign of C3's media-share
drift. The choice was made on parsimony rather than on that outcome, but the sensitivity is
logged so the decision can be re-examined rather than discovered later.

### D16 — 2026-08-05 — D11's bound gains a systematic allowance

**Loosened, and classified as such.** D11's pure sampling-noise bound fails at C1[0.95]:
0.0281 observed against a 0.0172 bound, passing at every other level. The cause is
systematic rather than sampling — pair means spread by 0.0327 against a seed-to-seed SD of
0.0030, an 11× ratio, and the ordering is perfectly monotone in the quarterly phase gap
between channels. The systematic term sits at 0.017–0.021 across all four levels and does
not scale with ρ, whereas D11's sampling term collapses as ρ → 1. That is why it bites only
at 0.95.

The bound is now **0.025 + 4·(1−ρ_target²)/√(T−3)**. A **separate** assertion requires the
measured systematic component to stay at or below 0.021, so the allowance cannot silently
absorb a growing defect — if the generator changes and the systematic spread grows, that
assertion fires rather than the bound quietly accommodating it.

This completes D5's own concession that equal pairwise ρ is unattainable with unequal
channel volatilities; D11's formula simply had no term for it. **This is a generator
self-consistency check, not one of the G-gates**, so loosening it does not change the
difficulty of the study for MMM and flatters no finding in either direction.

Setting `quarterly_amplitude` to 0.03 would also make it pass and was rejected: changing a
generator parameter to satisfy a threshold written after seeing the data is the wrong
direction regardless of who owns the parameter.

### D17 — 2026-08-05 — Multi-start agreement replaced with reference-optimum verification

**Corrected; the original test was wrong.** CLAUDE.md required multi-start solutions to
agree within 0.1%. Only 3–7 of 8 starts reach the best solution, with a spread of
0.105–0.156 on media contribution. This is correct behaviour, not a solver defect: TV
(α = 1.8) and OOH (α = 2.2) make the objective non-concave, genuine local optima exist, and
they are interpretable — they scale one S-shaped channel up and starve the rest. Requiring
starts to agree is requiring convexity, and if it held, one start would suffice.

The standard is now that **the returned optimum matches a 64-start reference**, with the
structured-plus-screened start set validated against a 256-start reference across at least
100 trials at zero misses.

**Limitation:** global optimality is therefore empirically supported, not proven. On a
non-concave surface no finite start set can prove it, and that is stated wherever the
optimum is reported.

### D18 — 2026-08-05 — Oracle-surface control added

**New control, no gate changed.** Because the objective is non-concave (D17), the
estimator's own allocation solve faces the same local optima as the truth solve. Without a
control, regret would conflate estimation error with optimisation error.

Two requirements. First, the model's allocation must use an **identical optimiser
configuration** to the truth solve, so residual optimisation error is common-mode. Second,
an **oracle-surface control** is run: the allocation procedure is applied to the *true*
response surface, which must return approximately zero regret. Any non-zero result is
optimisation error and is reported separately from, not folded into, the estimator's regret.

### D19 — 2026-08-05 — Regret's denominator varies tenfold; interpretation rule committed

**Interpretation rule plus a descriptive metric. G4 is unchanged.** True achievable lift
runs from 1.10% of total sales at C0 to 11.18% at C7. Regret normalises by that quantity, so
its denominator varies roughly tenfold across the grid. C0 has the tightest denominator in
the study: the surface is nearly flat near the status quo there, so small allocation errors
produce large regret percentages.

This was not anticipated when G4 was written and is recorded now, before the estimator
exists and before any regret number has been seen.

1. **G4's threshold of 20% is unchanged.** If C0 fails it, that is a genuine K1 question
   about the harness and is treated as one. No exemption is created here.
2. **Absolute lift lost, as a percentage of total sales, is reported alongside regret** for
   every condition. A 40% regret at C0 is 0.44% of sales; the same 40% at C7 is 4.5%.
3. **Cross-condition regret comparisons read as "share of what was achievable there," not
   "damage done."** The two orderings can differ and the absolute figure settles which is
   meant.

### D20 — 2026-08-05 — OOH is defunded at the true optimum in every condition

**Recorded, not corrected.** The true optimum drops OOH entirely (m = 0) in every condition,
and gives the placebo exactly zero budget wherever one exists. OOH has κ = 25 and α = 2.2
against a mean spend of 12, so it sits below the take-off point of its own S-curve and never
earns its marginal pound.

Two consequences. It sharpens G6: the correct placebo share is exactly 0%, not merely small,
so any share the model recommends is fabrication rather than rounding. And it makes OOH a
structural second quasi-placebo in decision terms, present in all conditions, which
marginally reduces how informative G3's rank metric is — OOH is always last.

β is not adjusted to fix this, because D4 locks the table and because a channel that should
be defunded is a realistic and useful thing for the study to contain. It is a finding for
the practitioner-facing write-up, not a defect.

### D21 — 2026-08-05 — K1 has fired on C0. The gate is not amended.

**Kill criterion triggered.** C0 fails G1 (median bias 64.0% against a 20% threshold) and G2
(coverage 32.0% against 80%) over 10 seeds. Under K1, C1–C7 are not reported and the grid is
blocked.

**K1 is not amended, and its threshold is not moved.** The question raised was whether C0's
failure means the harness is wrong, as K1 assumes, or whether "a clean-world MMM cannot
identify the media level" is itself the study's result. It is the result. But that is
reported by recording that K1 fired and explaining the mechanism, not by rewriting the gate
that fired. A pre-registered study which halts at its validity gate and explains why is a
complete result; one which reinterprets its kill criterion at the moment of firing is not.

**Mechanism, as measured.** The estimator trades media level against the free intercept:
media error −4.6 £k/week against baseline error +4.6, cancelling to −0.00 in predicted sales.
Starting Nelder-Mead at the true hyperparameters walks away from them — CV improves
3.63160 → 3.06272 while median bias goes 2.4% → 57.3%. Sweeping TV's (α, κ) with all else at
truth puts 177 of 780 points within 1% of truth's CV while TV's contribution ranges
43,938–240,522 £k. Differential evolution beats truth on fit and loses to random search on
recovery.

The decisive signature is the plateau, not the cancellation: the true parameters are not a
CV optimum, and a 5.5× range of channel contribution is indistinguishable on fit. A defect
produces a specific wrong answer; non-identification produces a flat direction. Structurally
this is expected — Hill saturation contains a near-constant function as a limiting case, and
a constant is collinear with a free intercept, so the media level is identified only through
curvature across the observed spend range.

A 13-agent adversarial audit raised 9 candidate implementation defects; all 9 were refuted.
The bounded-ridge solve was verified three independent ways and reproduces bit-identically
via an eigendecomposition path.

**K3 is answered in the estimator's favour.** Remedy 1 (tighter regularisation) is
monotonically worse. Remedy 3 is inapplicable at C0, where γ = φ = 0. Remedy 2 is a Step 7
item and rescuing the harness gate with a prior would mean C0 validates only when the
estimator is told the answer. The 200-draw random search is not a weakened MMM but an
accidentally flattering one.

**Required before any structural claim is made:**

1. **Reconcile the remedy-1 sweep.** It begins at 13.8% median bias, below G1's 20%
   threshold, which contradicts the 64.0% headline. Until it is established that the two
   figures measure different configurations, the C0 verdict is not settled.
2. **Fix the additive/multiplicative control mismatch first.** §2's baseline is a product
   while §4's controls are additive, leaving a structured 2.45 £k residual in every
   condition. It is assessed as not the cause, and probably is not. But the claim about to be
   made is that a *clean* world defeats MMM, and that claim is only as strong as C0's
   cleanliness. Implementation-level explanations are exhausted before a structural one is
   asserted. Add the trend×Fourier interaction columns and re-run C0.
3. **If C0 then passes**, K1 is discharged exactly as designed and the grid proceeds.
4. **If C0 still fails**, run the Meridian anchor on C0. §4 already names C0 as an anchor
   condition and already pre-commits the interpretation: if Meridian's priors rescue what
   RidgeMMM cannot, the prior is supplying information the data does not contain. No
   amendment is required; the machinery was written for this case.

**If the failure survives all four steps**, the study's reported outcome is that a
pre-registered stress test of MMM never reached its stress conditions, because under ten
years of weekly data with no confounding, no collinearity and correct functional forms, the
specified model could not identify how much of sales the media caused. The degradation grid
becomes moot rather than negative. That is a narrower study than planned and a stronger one.

### D22 — 2026-08-05 — The control block gains trend × Fourier interactions

**Corrected; §4's control list was incomplete for §2's own baseline.** §2 builds the baseline
as a *product*, `B0·(1 + τ·t/T)·season_t`, which expands to
`B0 + B0·τ·(t/T) + B0·season' + B0·τ·(t/T)·season'`. §4 names controls covering the first
three terms and not the fourth, so the estimator's control block could not represent the
baseline of the world it is tested in — a structured residual of 2.45 £k per week, 0.25% of
sales, present in *every* condition including the clean one.

Four `trend × Fourier` columns are added, taking the control block from 6 columns to 10.
Measured: the projection residual on C0's baseline falls from sd 2.45 £k to 1.4e-13, and at
the exactly true hyperparameters contribution recovery becomes exact (β̂ = β to printed
precision) where it was 3.7% biased.

**This is not a concession to the estimator, and it did not rescue C0.** D21 required
implementation-level explanations to be exhausted before a structural claim is made, and this
was the last one. Its measured effect on the failure is 0.641 → 0.640 median relative bias
under noise, and 0.580 → 0.503 noiseless. The C0 gates fail with it in place; see D23.

Two properties are asserted permanently rather than assumed. The control block now spans C0's
and C2's baselines to 1e-9. It still **cannot** absorb C3's `exp(γ·d_t)`, which leaves a
41.7 £k residual — that is the confounding C3 and C7 exist to test, and completing the
seasonal span must not have handed the estimator its confounder. Every column remains a
function of the week index alone, so the leakage guarantee of rule 3 is untouched.

### D23 — 2026-08-05 — C0 fails all five gates at 200 seeds. The 13.8% is reconciled.

**K1 stands, on the full seed count and with D22 in place.**

| Gate | C0 | Threshold | |
|---|---|---|---|
| G1 median \|relative bias\| | 0.540 | < 0.20 | fail |
| G2 coverage of the 90% interval | 0.417 | ≥ 0.80 | fail |
| G3 median Spearman rank | 0.650 | ≥ 0.80 | fail |
| G4 median allocation regret | 2.355 | < 0.20 | fail |
| G5 beats status quo | 0.200 | ≥ 0.90 | fail |

Only 9 of 200 seeds pass G1 individually, and **160 of 200 have regret above 100%** — the
model's advice is worse than not acting. The decision gates fail harder than the estimation
gates, which is the ordering the study was built to detect, arriving at the clean condition.

**D19's rule applied:** median regret of 2.355 is 2.7% of total sales in absolute terms
(p10–p90: 0.8%–6.1%), because C0's achievable lift is only 1.10% of sales. The correct reading
is "loses more than twice what was available there", not "destroys 235% of the business".

**The 13.8% is reconciled and does not survive.** It was `(noiseless, one particular draw
stream, seed 0 alone)` — the single most favourable cell measured. The ten per-seed values on
that same row are 0.138, 0.408, 0.544, 0.617, 0.785, 1.681, 0.507, 0.798, 0.275, 0.494, whose
median is 0.526. It was a minimum over ten noisy draws quoted as though it were a
configuration, which is the same selection-on-noise error that produced the first wrong
diagnosis in Step 4. Re-run properly — every penalty × 10 seeds × the noisy series the grid
actually uses — **no configuration passes G1**: 0.509 at ρ=1e-6, 0.515 at 1e-4, 0.557 at
1e-2, 0.893 at 1e-1, 0.987 at 1.0. K3 remedy 1 is refuted on a proper basis rather than on
one seed.

### D24 — 2026-08-05 — The Meridian anchor on C0. It agrees on G1 and disagrees on G2.

**Run as §4 specifies: C0, 10 seeds, default priors, no tuning.** 4 chains × (500 adapt, 500
burn-in, 1000 keep). All ten seeds converged, worst R-hat 1.008 against a 1.2 ceiling, so no
result here is an unmixed chain being reported as a failure of the method. Raw output is
`results/meridian_c0.json`. 31 minutes total.

| Gate | RidgeMMM (200 seeds) | Meridian (10 seeds) | Threshold |
|---|---|---|---|
| G1 median \|relative bias\| | 0.540 fail | **0.456 fail** | < 0.20 |
| G2 coverage of the 90% interval | 0.417 fail | **0.820 pass** | ≥ 0.80 |

**§4's verify-not-trust purpose is served, and the answer is that C0's contribution failure is
a property of the method.** Two estimators sharing no code, no optimiser and no inference
paradigm — penalised least squares with a random search against Hamiltonian Monte Carlo with
ROI priors — land 8 percentage points apart on the same gate, both roughly 2.5× outside it.
Meridian passes G1 on 1 of 10 seeds. Its median per-channel bias is +49.7% on TV, +107.1% on
OOH and −29.5% on search, and at seed 0 it puts media at **41.9% of sales against a true
25.0%**. Priors did not rescue it, so §4's rescue clause does not trigger for G1.

**The disagreement on G2 is reported as a primary result, per §4.** It is also explained. A
Bayesian posterior integrates over the transform parameters; §4's bootstrap conditions on one
point of the hyperparameter plateau and resamples residuals around it, which prices the
smallest component of the error and omits the largest. Meridian's intervals therefore cover
while `RidgeMMM`'s do not.

This localises the two failures differently, which matters for what the study claims:

* **G1's failure is structural** and survives a change of estimator, of optimiser and of
  inference paradigm.
* **G2's failure is `RidgeMMM`-specific**, an artefact of §4's fixed-hyperparameter bootstrap
  rather than of MMM. §4 already flagged that construction as understating uncertainty; the
  anchor shows the understatement is severe enough to flip a gate.

Note what Meridian's G2 pass does and does not mean. Coverage of 0.820 alongside a median
bias of 0.456 is an interval wide enough to contain a badly wrong point estimate — honest
uncertainty, not accuracy. Interval widths were not recorded, so the attribution to posterior
integration is inferred from the coverage-versus-bias pair rather than measured directly.

### D25 — 2026-08-05 — Prior art found post hoc; the claim is downgraded accordingly

**Positioning correction, made before publication.** A literature check run after the C0 result
found Dew, Padilla & Shchetkina (2024), *Your MMM is Broken: Identification of Nonlinear and
Time-varying Effects in Marketing Mix Models* (Wharton / London Business School, arXiv
2408.07678). It substantially anticipates the qualitative finding.

They establish that nonlinear and time-varying specifications are frequently not separately
identifiable; that standard model-selection metrics including cross-validation cannot
distinguish them; and that the two imply materially different optimal allocations, because
equivalence under status-quo spending does not imply equivalence under intervention. They
quantify a dollar "conflation cost" on Nielsen data and propose experimental separation
tests. They further cite Jin et al. (2017) for the point that the Hill function is poorly
identified, with different parameter combinations yielding effectively the same function over
a finite range — which is the plateau documented in D21, described in a 2017 paper that was
already in this study's own prior-art section.

**§8's prediction that C0 would pass was therefore made without connecting a citation already
in §1.** That is recorded as an error of literature review, not of measurement.

**What is still claimed, and nothing beyond it:**

1. **Within-family non-identification.** Dew et al. conflate two model *classes*. This study
   finds failure with the functional form exactly correct and the baseline recovered to
   1e-13 — a different and starker claim.
2. **Production tooling.** Their framework is their own Gaussian-process construction. This
   study ran Google's shipping Meridian on default priors, which failed G1 at 0.456.
3. **Pre-registration.** Conditions, metrics, thresholds and kill criteria committed before
   any code, with every deviation dated in this log.
4. **Distributional decision metrics.** Their conflation cost is a dollar figure for one
   brand-week. This study reports a beats-status-quo rate of 20% across 200 seeds.

Any framing that presents the identification failure as a discovery is overclaiming and must
be cut in review. Dew et al. is cited in the README's opening paragraph.

### D26 — 2026-08-05 — Exploratory spend-variation sweep added, labelled non-confirmatory

**Post hoc, and reported separately from the confirmatory results.** Identification depends on
curvature across the observed spend range, and `spend_log_sd = 0.30` was an assumption
recorded at the DGP stage rather than a specification. Sweeping it across 0.15 / 0.30 / 0.60 /
1.00 answers a question the confirmatory study cannot: how much deliberate spend variation a
team needs before MMM identifies anything.

This condition was **not pre-registered**, was added after C0 failed, and is reported in a
separate section under an explicit exploratory heading. It sets no gate and revises no
prediction. Its purpose is practitioner guidance and it converges on the same recommendation
as Dew et al.'s separation tests, from a different direction.

### D27 — 2026-08-05 — Absolute lift lost leads; normalised regret follows

**Reporting order, not a metric change.** C0's achievable lift is 1.10% of total sales, which
invites the objection that failing to capture so small a gain is unimpressive. The objection
is answered by the absolute figure: regret of 2.355 means the recommended allocation does not
merely miss a 1.10% gain, it costs **2.7% of total sales** against doing nothing.

Per D19 both are reported everywhere. This entry fixes which one leads: the absolute figure,
with the ratio immediately after. Neither number changes.

**Superseded in one respect by D29:** the figure is right and the comparator named in the
sentence above is wrong. 2.7% is the shortfall from the *optimum*, not the loss against doing
nothing. The ordering rule this entry sets is unaffected.

**Scoped by D38:** the clause immediately above stays true of the absolute-vs-ratio ordering this
entry sets, and says nothing about which of the two *absolute* figures leads. From D38 the
status-quo figure leads and the optimum figure follows. The 1.10% quoted twice above is also
retired — the 200-seed median is 1.16% (D29).

### D28 — 2026-08-05 — The optimiser start seed is fixed at 0. Recovered, not recorded.

**Reproducibility repair, found while building D26's sweep.** D23's five-gate table was
produced by a harness that was never committed. `sweep.py`'s control column is the first
independent re-derivation of it, and it is the reason the sweep can be compared to the
confirmatory result at all.

G1, G2 and G3 reproduced to the printed digit on the first attempt. G4 and G5 did not: passing
the *data* seed to the recommendation solve gives 2.386 and 0.195 against D23's 2.355 and
0.200. The whole of the difference is which seed reaches the SLSQP starting points. Measured
across six conventions, 200 seeds each:

| Convention | G4 median regret | G5 beats-status-quo |
|---|---|---|
| truth = seed, recommendation = seed | 2.3856 | 0.1950 |
| truth = 0, recommendation = 0 | **2.3546** | **0.2000** |
| truth = seed, recommendation = 0 | **2.3546** | **0.2000** |
| truth = 0, recommendation = seed | 2.3856 | 0.1950 |
| truth = seed, recommendation = seed, 16 starts | 2.4060 | 0.1850 |
| truth = seed, recommendation = seed, 32 starts | 2.4173 | 0.1850 |

D23 used a **fixed** seed on the recommendation solve. The truth-side solve is insensitive to
the choice — the structured starts win on the true surface and the single screened random start
never decides the optimum there — which is why the two conventions that fix the recommendation
seed agree exactly with each other and with D23. `sweep.py` fixes `OPTIMISER_SEED = 0` for both
solves and refuses to write results unless the control reproduces D23 to 5e-4.

Two things this makes visible, neither comfortable:

- **A published decision number rested on an undocumented convention.** G1–G3 are properties of
  the generating process and the fit, and were never at risk. G4 and G5 are properties of a
  non-concave optimisation, and were.
- **More starting points make regret worse, not better** — 2.386 → 2.406 → 2.417 at 8, 16 and 32
  starts. This is D21's anti-strawman result appearing a second time, in a second place: a
  better search on the *fitted* surface finds allocations that are better by the model's own
  reckoning and worse against the truth.

The sensitivity is small — 1.3% relative on median regret, one seed in 200 on beats-status-quo —
and belongs in the limitations, stated, rather than here, buried.

### D29 — 2026-08-05 — D27's 2.7% names the wrong comparator. The figure stands, the sentence does not.

**Correction to a reporting decision, made before publication.** D27 states that regret of 2.355
means the recommendation "costs 2.7% of total sales *against doing nothing*". Both quantities,
measured at 200 seeds on C0:

| Quantity | Median | p10 | p90 |
|---|---|---|---|
| Shortfall from the optimum, `(S_opt − S_model) / S_sq` | **2.73%** | 0.77% | 6.15% |
| Loss against doing nothing, `(S_sq − S_model) / S_sq` | **1.58%** | −0.42% | 4.99% |

D23's "2.7% of total sales (p10–p90: 0.8%–6.1%)" matches the first row to two decimals and does
not match the second. The 2.7% is the gap to the **best available allocation**, not the damage
done relative to the status quo.

Both numbers are real, both are damning, and they answer different questions. The write-up uses
both, in this form:

- the recommended allocation falls **2.7% of total sales short of the best available one**,
  against an achievable lift of 1.16%;
- and at the median it **destroys 1.6% of total sales against doing nothing**.

D27's ordering rule is unchanged: the absolute figure leads, the ratio follows. Only the
comparator in its prose is corrected, and nothing that was computed changes.

**Scoped by D38, and one prescription above superseded by it.** The clause immediately above remains
true of the absolute-vs-ratio axis. The bullet pair prescribed above — optimum shortfall first,
status-quo loss second — is **reversed** from D38 onward: the status-quo figure leads. This entry's
correction, its measured table and its percentiles all stand. D38 also corrects this entry's own
attribution: "§4" contains no achievable-lift figure, and the third site quoting 1.10% is **D23**.

**A second number in the same family does not reproduce.** §4, D19 and D27 all quote C0's
achievable lift as 1.10% of sales. The 200-seed median is **1.16%** (p10 1.11%, p90 1.22%,
min 1.03%, max 1.26%, seed 0 = 1.17%). 1.10% is not the median, not seed 0 and not an extreme;
14 of 200 seeds fall below it. Its provenance is not recorded and is not recoverable from the
repository. The write-up uses 1.16% and says which figure it is using.

### D30 — 2026-08-05 — Two sweep cells fail SLSQP. They are recorded, not absorbed.

**Measured, bounded, and excluded with the exclusion stated.** Across the 800 cells of D26's
sweep, two fail the optimiser: `spend_log_sd` 0.15 seed 78, and 1.00 seed 160. Both fail on the
**fitted** surface, never the true one; both fail from a *structured* start; both have healthy
coefficients (min |β̂| of 95.1 and 12.4, no zeros). This is a local linesearch failure on a
non-concave estimate, not a degenerate model, and not the unscaled-objective failure CLAUDE.md
describes.

`truth.optimal_allocation` raises on any failed start and that does not change. Every number
already published rests on that strictness, and CLAUDE.md names "convergence failures reported
as low regret" as a specific failure mode. The handling is therefore in the harness, not the
instrument: the cell is recorded as a `SolveFailure`, written to the CSV as `solve_failed = 1`
so that it cannot be mistaken for a seed that was never run, and excluded from the medians.

Excluding cells is a selection, so the adverse bound travels with it. `g5_worst_case` counts
every failure as a loss: at sd 0.15 it is 0.0950 against the reported 0.0950, and at sd 1.00
0.4600 against the reported 0.4623. One cell in two hundred cannot move a median, and the
bound is reported rather than asserted.

The control column at sd 0.30 has **zero** failures, which `verify_control` requires before any
result is written.

### D31 — 2026-08-05 — The novelty claim for the decision metrics is withdrawn entirely

**Second downgrade, from the novelty sweep D25 required.** D25 item 4 claimed "distributional
decision metrics" as a surviving contribution: a beats-status-quo rate across 200 seeds against
a single dollar figure for one brand-week. A search across academic marketing science,
industry and vendor material, adjacent literatures and brute-force phrasing — roughly 130
queries and three independent adversarial passes — establishes that the claim as worded does
not survive. **It is withdrawn. No novelty is claimed for the metric.**

What was found, in descending order of how badly it damages the original wording:

| Source | What it reports | Verified |
|---|---|---|
| Haus, *Fast, Confident, and Wrong* (haus.io blog) | "acting on noisy results left the business **worse off than doing nothing 38% of the time**"; 36 scenarios × 1,000,000 runs; "The true performance of every channel is **held fixed**" | Re-fetched directly, quotes confirmed |
| Dew et al. (2024) Table 1, p.19 | A rate over 2,187 settings × 100 datasets against known truth — 81–99% any-conflation, 27–47% major | Four independent readers |
| Agarwal et al. (2021), NeurIPS | Standardised *probability of improvement* across seeds, with bootstrap intervals, as a reporting convention | Abstract |
| Thomas et al. (2015, 2019) | Safe policy improvement is *built* around bounding P(learned policy worse than incumbent) | Abstract |
| DeMiguel, Garlappi & Uppal (2009); Smith & Winkler (2006) | The finding's shape — an optimiser losing to a naive incumbent once estimation error swamps the optimisation gain — and the mechanism that names it | Abstract |

**Haus is the one that matters and it is not academic**, which is exactly why the first sweep
missed it and a practitioner referee would not. It differs from this study in a way that is
evidence rather than an excuse: **Haus fits no model.** Measurement is truth plus stipulated
noise — no adstock, no saturation, no confounded demand, no design matrix, no identification
problem. Their quantity is the cost of *noise* in an unbiased readout, which more data removes.
This study's is the cost of *non-identification* in a correctly specified estimator, which more
data does not remove. Both are worth knowing and they are not the same object.

It is also worth stating without flinching: **this study's 20% is worse than Haus's worst arm,
which beat the baseline 62% of the time.**

**What the write-up says instead.** G4 and G5 are presented as measurements, not as a
contribution. The statistic is credited where it comes from. The residual claim — that this is
the rate for a *fitted* MMM evaluated by intervention on a known generating process, rather
than for a stipulated-noise readout or against the model's own fitted surface — is stated once,
in plain language, and not called novel.

**One thing found in the sweep runs the other way and is recorded here so it is not lost.**
Pathak, Jeunen & Lambert (2026), *Auditing Marketing Budget Allocation with Hindsight Regret*
(arXiv 2604.25977), name this study's design as their own open problem: "An important next step
is a semi-synthetic benchmark with **known response structure and oracle regret**, which would
enable direct recovery-based validation of the framework." That is a dated, independent
statement from a competing team that the gap exists. It is a better opening for the related-work
section than any novelty claim would have been.

**Disclosed gap.** One of three adversarial refuters — the pre-1990 marketing-science pass over
Little, Lodish, Hanssens, Leeflang and the normative decision-model literature — terminated on a
session limit and did not complete. That angle is unswept.

### D32 — 2026-08-05 — D26's claim of convergence with Dew et al.'s separation tests is corrected

**D26's text stands as written and is corrected here rather than edited, per the log's own
rule.** D26 closes: "it converges on the same recommendation as Dew et al.'s separation tests,
from a different direction." Having now read §7 in full, that sentence is not defensible on
three counts.

1. **Dew et al. prescribe a spending *policy*, not an amount of variation.** §7.1's maximal
   separation sets each period's spend to the grid value maximising the disagreement between
   the two candidate models, refits, and reads off which predicted better; §7.2's seesaw
   alternates between the highest and lowest candidate spend "one period at a time". The only
   quantity they give is a **duration** — "a firm need only conduct this test for 1–2 periods".
   There is no amplitude prescription anywhere to converge on.
2. **Their variation is experimental; this sweep's is observational.** They choose future spend
   and refit. This sweep raises the dispersion of spend the DGP generates. Their §5.2 is built
   on precisely the distinction between status-quo and interventional evidence, so blurring it
   hands away the study's own central point.
3. **They ran the observational analogue and it did nothing.** Their spend-variance parameter τ
   ∈ {1, 5, 10} is insignificant in all four cells of Table 2 (p.19; largest |coefficient| 0.47,
   smallest p 0.37). A claim of convergence would put this study in visible tension with their
   own table.

**The corrected relationship, which is stronger than the claim it replaces.** Dew et al. reach a
related conclusion from the model-selection side: their remedy for non-identification is a
deliberate spending policy, not the hope that observed variation will suffice. This sweep is the
observational counterpart and reaches the same practical conclusion by failing: even at
`spend_log_sd` = 1.00 — weekly spend routinely swinging by a factor of about 2.7, well beyond
anything a planner produces — four of five gates still fail and 53.8% of runs remain worse than
doing nothing. **That is an argument for their experimental route, not a convergence with it.**

Why this sweep moves identification at all where their τ sweep did not, stated so it is not
mistaken for a contradiction: this DGP's log spend is serially uncorrelated, so raising its
dispersion raises *local* week-to-week variability and marginal spread together, which is closer
to a randomly timed seesaw than to their τ sweep. Their outcome is holdout prediction parity,
which extra variance need not break; this study's is recovery and allocation regret, which extra
curvature coverage directly serves.

**Carried into the write-up as a caveat**, per their p.33 note that carryover blunts seesaw
tests: adstock decays of 0.10–0.70 mean the estimator sees less week-to-week swing than is
injected — most of all on TV, whose λ is 0.70, the highest in §2's table — so a recommendation
phrased in raw spend variation will not transfer unchanged to a team with heavy TV adstock.

**Correction to this entry's own arithmetic.** An earlier version of the sentence above, and of
the README paragraph derived from it, gave the decay range as 0.10–0.60. §2's table is 0.70 (tv),
0.45 (video), 0.10 (search), 0.30 (social), 0.60 (ooh). The upper end is TV's 0.70 — which is the
channel the sentence uses as its own example, so the wrong bound excluded the case it was making.

**A second limit on this sweep, recorded because it bounds what D32 may conclude.** The sweep
varies the *amplitude* of spend jitter and never its *shape*. The spend process is lognormal and
never reaches zero, so a flighted plan — weeks on, weeks dark, which is the closest cheap
analogue of Dew et al.'s deliberate high-low policy and is already how many advertisers buy TV —
is **outside the family swept here**. The claim "no realistic amount of observational variation is
enough" is therefore established for amplitude and untested for shape. Whether a flighted plan
restores identification is the most valuable outstanding run in this project.

### D33 — 2026-08-05 — Post-hoc optimiser-bound robustness check. The failure survives it.

**Not pre-registered, added because the prior art demanded it.** §3 allows `m_c` ∈ [0, 3.0], so
the model may recommend tripling a channel on a response curve fitted from data that never went
there. Dew et al. impose the opposite restriction twice and say why — "we remove the candidate
allocations that assign to at least one of the channels a level of spending that falls outside
of the historical range" (p.29) and "This ensures that our optimal solution is not relying purely
on extrapolation of the response function" (fn.13). A median regret of 2.355 with 160 of 200
seeds above 100% is exactly what an optimiser extrapolating an S-shaped Hill curve would produce,
and that is the most likely way the headline gets dismissed.

Both solves take the same bound, so D18's requirement that truth and recommendation be
identically configured is preserved. C0, 200 seeds:

| Upper bound on m_c | G4 median regret | G5 beats status quo | 95% CI | Achievable lift | Share regret > 1 | At upper bound (of 5) | Zeroed, model | Zeroed, truth | n |
|---|---|---|---|---|---|---|---|---|---|
| 1.3 | 1.407 | 0.303 | [0.243, 0.370] | 1.126% | 0.697 | 1.530 | 0.75 | 1.00 | 198 |
| 1.5 | 1.501 | 0.270 | [0.213, 0.337] | 1.160% | 0.730 | 0.867 | 0.88 | 1.00 | 196 |
| 2.0 | 2.284 | 0.216 | [0.165, 0.278] | 1.160% | 0.784 | 0.291 | 1.16 | 1.00 | 199 |
| 3.0 (§3) | 2.355 | 0.200 | [0.150, 0.261] | 1.160% | 0.800 | **0.035** | 1.24 | 1.00 | 200 |

**Precise wording of what was run, because a first version of this entry got it wrong.** The
bound applied is `m_c ∈ [0, B]` — an upper cap only. `truth.py` fixes the lower bound at 0.0 at
every setting, so a channel can always be switched off. The row labelled 1.3 is therefore "cap
every channel at 130% of current spend", **not** the two-sided ±30% planning rule it was first
described as. A genuine two-sided guardrail is a different and untested configuration.

**The extrapolation objection is answered.** At §3's own bound, the model's recommendation puts
on average **0.035 of 5 channels** at `m_c = 3.0` — seven of a thousand channel-slots. There is
essentially nothing at 3× spend for the regret to be an artefact of. Capping at 1.3 instead
leaves the advice still worse than doing nothing at the median (1.407) and beating the status quo
in under a third of runs, 0.303 [0.243, 0.370] against a threshold of 0.90.

**A claim withdrawn.** The first version of this entry concluded from that 0.035 that "the
recommendation is an interior solution". That was wrong, and it was wrong because
`on_upper_bound` counts only one of the two bounds. The binding constraint is the **lower** one:
the model zeroes **1.24 of 5 channels** on average at bound 3.0. `robustness.py` now counts both,
and the CSV carries both columns.

**What that actually shows, once counted properly, is more interesting than the withdrawn claim.**
The *true* optimum also zeroes a channel — exactly **1.00**, in every seed, always OOH, the one
channel whose true ROAS is below break-even at 0.82 against 1.73–2.30 elsewhere. So defunding a
channel is not the error. Defunding **more** channels than the truth does is, and the model
defunds 1.24 where the truth defunds 1.00. Regret is measured like for like; the model is simply
wrong about which channels deserve it.

**One consequence for how regret should be read, recorded because it is a real limitation.** Since
the true optimum's entire 1.16%-of-sales headroom is essentially "delete OOH and move it to TV",
and no marketing team switches a channel off on model evidence alone, the benchmark is not an
implementable decision. Regret here is measured against an optimum a governed team could not
reach, which raises every regret figure relative to one computed inside a realistic action space.
Stated in the README's limitations.

The mechanism is worth stating because it is the interesting part: the **true** optimum barely
needs the extra room, since achievable lift falls only from 1.160% to 1.126% when the bound drops
from 3.0 to 1.3. The **model's** regret nearly halves over the same change. The damage is
concentrated in recommendations the truth never needed, which is what an estimation-error
maximiser looks like.

Counts below 200 are the D30 solve failures at the tighter bounds, recorded there.

### D34 — 2026-08-05 — Flighted spend as a validity check on C0. Design and readings pre-committed.

**Written before the run. No flighting result existed when this entry was committed.**

**This is not a third exploratory arm; it is an anti-strawman check on C0 itself**, in the same
class as D9 and D12. D26's sweep varies the *amplitude* of spend jitter, and jitter of any size
samples a narrow band of the response curve — it moves spend around its mean without ever
approaching the origin. A flighted plan does something categorically different: it traces the
curve **near zero**, and it makes adstock decay directly observable during dark weeks, because
sales in a dark week are the carryover and nothing else.

So the question is not whether the sweep was complete. It is whether **C0's spend process was
unrepresentatively hard**. Real media plans may carry identifying information that a lognormal
jitter simply does not contain. If so, C0 understates what a real analyst has, and the headline
weakens accordingly. That possibility has to be tested by the people making the claim.

**Design, fixed before running and chosen to be defensible rather than favourable:**

- **TV, video and OOH flight. Search and social stay always-on.** That is how these channels
  actually behave — search and social are budget-capped and pacing-automated, brand channels are
  bought in bursts. It is not a split chosen to produce an outcome.
- **Bursts of 2–6 weeks live, dark blocks between, targeting a 40–60% duty cycle.**
- **Channels flight independently, not in sync.** Synchronised flighting would induce cross-channel
  spend correlation and confound this check with C1, which tests collinearity on purpose.
- **Each channel's total spend is preserved**, by scaling live weeks up by the reciprocal of its
  realised duty cycle. A planner who flights has the same budget and concentrates it; leaving
  totals to fall would confound "flighted" with "spent half as much".
- **Realised media share is reported** (D4). Concentrating a fixed budget into fewer weeks pushes
  those weeks further along the saturation curve, so the 25.0% of C0 will drift and the drift is
  a result, not a defect.
- Truth is recomputed by intervention on the new spend path, as §3 requires. Nothing is carried
  over from C0.

**Both readings are committed here, before the number is known:**

- **If G1 still fails under flighting** — C0's spend process was not the cause of the failure.
  The headline stands, with one fewer attack surface, and the "unrepresentatively hard DGP"
  objection is answered on the record rather than argued.
- **If G1 passes under flighting** — the conclusion is **not** "MMM works". It is that
  identification depends on spend having gone dark, which is a property of the **media plan**,
  not of the model. A team whose channels are always-on cannot identify its own response curves
  no matter how good its modelling is, and a team that flights can. That is a **more** actionable
  finding than the current one, it becomes the lead recommendation in
  `docs/WHEN-TO-TRUST-YOUR-MMM.md`, and it is reported without hedging.

Recorded explicitly: **this check was run because it could overturn the headline, not because it
was expected to support it.**

Post-hoc and non-confirmatory, like D26. It sets no gate and revises no §8 prediction.

### D35 — 2026-08-05 — Two-sided allocation guardrail. Pre-committed, run in the same pass.

**Written before the run.** D33 capped `m_c` above and left the floor at zero, so its rows do not
answer the separate objection that regret is driven by the optimiser making recommendations no
planner would accept. D20 records that the true optimum **defunds OOH entirely in every
condition**, and on C0 that single move is most of the available headroom — so both the benchmark
and the recommendation depend on a decision a governed team would never sign off.

This adds `m_c ∈ [0.7, 1.3]`: a planning rule that forbids switching any channel off and caps
change at ±30% either way. It needs no refit — the same fitted surfaces are re-solved — so it runs
in the same pass as D34.

`truth.optimal_allocation` gains a `min_multiplier` argument defaulting to 0.0. **The default
reproduces present behaviour exactly**, so no number already reported moves. Both the truth solve
and the recommendation solve take the same floor, preserving D18.

**Pre-committed reading:** the interesting quantity is G5, not G4. Regret's denominator shrinks
under the guardrail — less headroom is reachable — so G4 is not comparable across bound sets and
will be reported without cross-comparison. G5 is a rate and is comparable. If the advice still
loses to the status quo more often than not under a rule that forbids the destructive move, the
objection is closed.

### D36 — 2026-08-06 — D34's result. Flighting fixes estimation and does nothing for the decision.

**Run as D34 specified. The baseline arm reproduces D23 exactly — 0.540 / 0.417 / 0.650 / 2.355 /
0.200 — which is what licenses the comparison: only the spend process changed.** 200 seeds per
arm. Raw output is `results/flighting_check.csv`.

| Gate | C0 baseline | C0 flighted | Threshold | |
|---|---|---|---|---|
| G1 median absolute relative error | 0.540 | **0.309** | < 0.20 | still fails |
| G2 coverage | 0.417 | 0.323 | ≥ 0.80 | fails, and worse |
| G3 median Spearman | 0.650 | **0.800** | ≥ 0.80 | **passes** |
| G4 median regret | 2.355 | 2.894 | < 0.20 | fails |
| G5 beats status quo | 0.200 [0.150, 0.261] | **0.207 [0.156, 0.269]** | ≥ 0.90 | fails |
| Seeds passing G1 individually | 4.5% | **25.3%** | — | 5.6× more |
| Share of runs worse than doing nothing | 0.800 | 0.793 | — | unchanged |
| Realised media share (D4) | 0.2500 | 0.2382 | — | drifts down 1.2 points |
| Achievable lift | 1.16% | 0.54% | — | halves |
| Duty cycle, flighted channels | 1.00 | 0.50 | — | as designed |

**D34's pre-committed reading applies: G1 still fails, so C0's spend process was not the cause of
the failure and the headline stands with one fewer attack surface.** The objection that a
lognormal jitter is an unrepresentatively hard world has been tested rather than argued, and it
does not overturn the result.

**But the referee's intuition was half right and that half is reported without spin.** Flighting
carries real identifying information: G1 nearly halves, the number of individual seeds passing it
rises more than fivefold, and G3 crosses its threshold — channel *ranking* becomes reliable when
spend goes dark. Anyone claiming this study shows spend variation is irrelevant would be
misreading it.

**The finding is what happens to the two halves separately, and it restates the study's thesis in
a second, independent place.** Flighting substantially improves **estimation** and leaves the
**decision** exactly where it was: G5 moves from 0.200 to 0.207 with intervals that almost
entirely overlap, and the share of runs worse than doing nothing is 0.800 against 0.793. A team
that flights its buys gets a better-estimated model and an equally bad budget recommendation.
That is *attributable ≠ incremental* appearing again, now as *estimable ≠ actionable*.

Two mechanical notes. G4 rises to 2.894, but the two arms' achievable lift differs (1.16% against
0.54%), so regret's denominator is not the same quantity and the two G4 figures must not be
compared — the same caveat D35 records. And two flighted cells failed SLSQP and are excluded per
D30.

### D37 — 2026-08-06 — D35's result. The guardrail helps a great deal, and the gates still fail.

**Run as D35 specified, on the same fitted surfaces, 200 seeds.** Raw output is
`results/optimiser_bound_check.csv`.

| m_c range | G4 median regret | G5 beats status quo | 95% CI | Achievable lift | Share worse than nothing | n |
|---|---|---|---|---|---|---|
| [0.0, 3.0] (§3) | 2.355 | 0.200 | [0.150, 0.261] | 1.160% | 0.800 | 200 |
| [0.0, 1.3] | 1.407 | 0.303 | [0.243, 0.370] | 1.126% | 0.697 | 198 |
| **[0.7, 1.3] (D35)** | 0.501 | **0.760** | **[0.696, 0.814]** | 0.623% | **0.240** | 200 |

**D35's pre-committed reading does not go this study's way, and it is recorded as stated rather
than reinterpreted.** The commitment was: "if the advice still loses to the status quo more often
than not under a rule that forbids the destructive move, the objection is closed." It does not.
Under a two-sided ±30% guardrail the advice **beats** the status quo 76.0% of the time. **The
objection is therefore sustained in part**, and the write-up says so.

What survives, and it is the thing the gates were written to test: **G5 of 0.760 still fails the
pre-registered threshold of 0.90, with a 95% interval of [0.696, 0.814] that excludes it**, and
G4 of 0.501 still fails 0.20 by two and a half times. A guardrailed MMM is far less destructive
and still not good enough by the standard fixed before any of this ran.

**What this changes in the write-up.** The line "worse than leaving the budget alone in 160 of 200
worlds" is true of §3's pre-registered action space and **not** of a governed one, where the
figure is 48 of 200. Both are reported, the action space is named each time, and the unconstrained
figure never appears alone.

**Why the guardrail helps so much is itself the mechanism.** It removes the decision that carries
most of C0's headroom: the truth's optimum defunds OOH entirely (D20), and forbidding that halves
achievable lift from 1.160% to 0.623%. Under the guardrail the *truth* sits 1.97 channels on the
floor and the model 1.52. The model is not failing because it makes wild recommendations — it is
failing because it is wrong about which channels deserve the money, and a guardrail limits how
much damage being wrong can do. It bounds the consequence, not the error.

### D38 — 2026-08-06 — The status-quo comparator leads. D29's ordering-by-example is reversed; its correction stands.

**Raised as an uncorrected arithmetic error in D27, and it is not one.** For the record, because
anyone reading D27 in isolation will raise it again: D27 says the recommendation "costs 2.7% of
total sales against doing nothing"; 2.7% is the shortfall from the **optimum**; and **D29 already
corrected exactly that** on 2026-08-05, measured both quantities at 200 seeds, annotated D27 in
place with a pointer, and the corrected form is already carried in `README.md` and
`docs/WHEN-TO-TRUST-YOUR-MMM.md`. No arithmetic changes here and no published number moves. This
entry exists because something else in the same sentence does change.

**What changes is which of the two *absolute* figures leads, and the distinction matters because
two live entries say the ordering is untouched.** D27's rule is an ordering over {absolute figure,
normalised ratio}, and 1.58% is also an absolute figure, so that rule survives the swap intact — the
ratio still follows either way. What the swap actually overturns is **D29's ordering of the two
absolutes, which D29 fixed by example and never argued**: "the recommended allocation falls 2.7% of
total sales short of the best available one … and at the median it destroys 1.6% of total sales
against doing nothing." That order is reversed from here.

**One prescription is superseded and two clauses are scoped, and the difference between those matters
enough to separate them.**

- **Superseded — D29:881–886.** "The write-up uses both, in this form: — the recommended allocation
  falls 2.7% of total sales short of the best available one …; — and at the median it destroys 1.6%
  of total sales against doing nothing." That is the live instruction the new order breaks. It is
  **retracted and replaced** by the order below. Nothing else in D29 changes; its correction and its
  measured table stand entirely.
- **Scoped, not retracted — D27:824 and D29:888.** "The ordering rule this entry sets is unaffected"
  and "D27's ordering rule is unchanged: the absolute figure leads, the ratio follows" are both still
  **true**, because they concern the absolute-vs-ratio axis and 1.58% is also an absolute. Read them
  as saying nothing about the order between the two absolute figures, which is what this entry sets.

Both D27 and D29 carry a one-line pointer to this entry, added in place. That is the same mechanism
D29 used on D27, and declining it here while citing it there as the reason the earlier correction is
discoverable would have been having it both ways.

The new order:

1. **The loss against doing nothing leads**, with its comparator named in the same sentence.
2. The shortfall from the optimum follows, also named.
3. The regret ratio follows that.

**The rationale is not only that practitioners face the status quo. There is a measurement argument,
and it is the stronger of the two — but it holds over a narrower set of comparisons than the first
draft of this entry claimed, and the narrowing is recorded rather than quietly dropped.**

**Where it holds: the dataset fixed, the action space varied.** That is D33 against D35. `S_sq` is
then literally the same number in every arm — the same fitted surfaces are re-solved under different
bounds — while `S_opt` moves, because D33 records that regret is measured against an optimum a
governed team could not reach and D35 that regret's denominator shrinks under the guardrail. So
across action spaces the status-quo-referenced figure is comparable and the optimum-referenced one
is not. No entry in this log had made that point, and it is the reason the reordering improves the
write-up rather than merely re-angling it.

**Where it does NOT hold, stated because the first draft cited it as support and it is the
counterexample.** D36's two arms share an action space and differ in the *spend process*, which
moves `S_sq` as well — D36:1171 records realised media share falling 0.2500 → 0.2382 at the status
quo, and D34:1111 gives the mechanism, a fixed budget concentrated into fewer weeks sitting further
along the saturation curve. Across the flighting arms **neither** absolute figure is comparable, and
only the rates are. Worse, the newly-mandated leading figure cannot even be produced for the flighted
arm from committed data: `flighting_check.csv` carries no achievable-lift column, so the per-seed
change against the status quo is not derivable. The README says all three of these things where the
flighting table is.

**Three constraints from other entries, each honoured rather than overridden.**

- **D19.2** mandates "absolute lift lost, as a percentage of total sales, reported alongside regret",
  and absolute lift lost is `regret × achievable lift` — the *optimum* comparator, the 2.73%. Leading
  with the status-quo figure complies with D19.2 only while 2.73% still appears alongside. It does,
  everywhere, and that is now a requirement rather than a habit.
- **D19.3** is the one that needed thinking about, and the honest answer is that **nothing about it
  is re-pointed** — an earlier draft of this entry claimed it was, and that claim is withdrawn. Its
  warning reads "**Cross-condition regret comparisons** read as 'share of what was achievable there,'
  not 'damage done'", so it was always scoped to *regret*, and the status-quo loss is a different
  quantity that may legitimately be read as damage precisely because its baseline does not move.
  What D19.3 does leave unresolved is its second sentence: "the absolute figure settles which is
  meant" had exactly **one** referent when it was written, and this entry introduces a second
  absolute and leads with it. **The disambiguating role stays with lift lost — the optimum
  comparator — because that is the quantity commensurate with regret.** That is precisely why every
  figure must name its comparator in the same sentence: with two absolutes in play, position no
  longer identifies which one is meant, and only the label does.
- **D37** requires that the action space be named at every use and that "the unconstrained figure
  never appears alone". A leading figure inherits that rule, and the guardrailed counterpart of
  −1.58% did not exist anywhere in this log, the README or the docs before this entry. It is in the
  table below, and it is **+0.30%** — a gain, not a loss. That is a materially weaker headline
  pairing than "destroys 1.58%" alone, and the decision to lead with the status-quo comparator is
  taken in full view of it.

**Both figures, in three action spaces, as signed change in total sales against leaving the budget
alone** (negative is worse). Recomputed this session from `results/optimiser_bound_check.csv`, which
carries no loss column — the quantity is derived per seed as `achievable_lift_share × (1 − regret)`
and the median taken over seeds. The §3 row reproduces D29's second row with the sign flipped and
the percentiles correspondingly exchanged; the shortfall column is D29's first row unchanged.

| Action space | Change vs doing nothing | p10 | p90 | Shortfall from the optimum | Regret |
|---|---|---|---|---|---|
| §3, `m_c ∈ [0, 3.0]`, n=200 | **−1.58%** | −4.99% | +0.42% | 2.73% | 2.355× |
| D33, `m_c ∈ [0, 1.3]`, n=198 | −0.46% | −2.01% | +0.66% | 1.60% | 1.407× |
| D35, `m_c ∈ [0.7, 1.3]`, n=200 | **+0.30%** | −0.27% | +0.55% | 0.32% | 0.501× |

A `loss_vs_status_quo_share` metric does exist, but in `results/spend_variation_sweep.csv`, not in
the bound check; it and D29's table state the same quantity with the opposite sign convention
(positive = loss). Prose in the write-up avoids the ambiguity by naming the direction in words.

**Two estimators of "the median", differing in the second decimal, recorded so neither looks like
an error.** Applying the identity to the two published medians gives (1 − 2.355) × 1.160% = −1.57%
in §3's space and (1 − 0.501) × 0.623% = +0.31% guardrailed. Taking the median of the per-seed
quantity gives −1.58% and +0.30%. The median of a ratio is not the ratio of medians. **The measured
medians are what is reported**, because they are the median of the thing being described and
because the p10/p90 quoted beside them come from the same distribution; the derived figures are
noted in the README so that a reader who reproduces them from the headline medians does not
conclude there is a mistake. The gap is under a hundredth of a percentage point and no conclusion
turns on it.

**Four claims checked while making this change, and corrected before they reached the write-up.**
Every headline figure in `README.md` was recomputed from the committed CSVs in the same pass; all
of the gate tables reproduced exactly, and these four did not.

1. **"G3 crosses its threshold for the first time in the study" under flighting — it does not.**
   G3's median Spearman is 0.900 at `spend_log_sd = 1.00` in D26's sweep, run and logged a day
   earlier. The defensible form, and the one the README now uses, is that flighting is the only
   place in this study where **ranking recovers under an intervention a media team would actually
   choose**; `spend_log_sd = 1.00` is a level of jitter no planner would introduce.
2. **`docs/WHEN-TO-TRUST-YOUR-MMM.md` called G3 "the only gate that ever passed" — also wrong.**
   Meridian passes G2 at 0.820 (D24).
3. **"a factor of about 16 between its 10th and 90th percentile" at `spend_log_sd = 1.00` — it is
   about 13.** Measured over 30 seeds × 5 channels the p90/p10 spend ratio has mean 13.07 and
   median 12.98, against a per-channel-seed range of 10.0 to 17.2. **16 is near the top of that
   range quoted as though it were typical** — the same failure CLAUDE.md names for extrema over
   noisy tries, appearing here as a maximum rather than a minimum. The companion figure, "40–45% of
   weeks below half that channel's own average", holds at a mean of 42.4% and is restated as ~42%.
4. **"an interquartile range of roughly ±100 percentage points" on the signed per-channel bias —
   overstated by about a factor of two.** The IQR *widths* at C0 are 94.4 (search), 103.2 (video),
   105.5 (social) and 109.9 (tv) percentage points, so ±100 should have been ±50, or a width of
   about 100. OOH's is 258.6 and is now stated separately rather than folded into "each".

**Five more, found by an adversarial pass over the rewritten documents before they were shown to
anyone.**

5. **"G3 crosses its threshold" overstates a statistic that lands exactly on it.** The flighted
   median Spearman is 0.8000, the interquartile range is [0.700, 0.900], and **104 of 198 seeds** sit
   at or above 0.80. Spearman on five channels moves in steps of 0.1, so "crosses" implies a
   precision the metric does not have. The documents now say **reaches**, and carry the count.
6. **"The budget decision does not move" / "exactly as bad as before" accepted a null from an
   underpowered comparison.** Paired over the 198 seeds both arms solved, the G5 difference is
   **+1.0 percentage point, 95% CI [−7.1, +9.1]**, McNemar *p* = 0.90 (34 discordant one way, 32 the
   other). The data are consistent with flighting raising the beat rate to 29%, and this design could
   not have detected it. The claim is now **"no measurable improvement"** with the interval printed,
   which is what the evidence supports.
7. **OOH's true ROAS of 0.82, and the 1.73–2.30 range for the others, are seed-0 realisations quoted
   as properties of the generating process.** The 50-seed medians are 0.84 and 1.73–2.32. The
   load-bearing claim survives — OOH is below break-even in all 50 seeds checked, maximum 0.852 — but
   the figures are now labelled.
8. **"7 of 995" cells fail in the bound check — the denominator is 1000.** Five arms × 200 seeds;
   993 rows survive.
9. **"Halves achievable lift from 1.160% to 0.623%" is a 46% cut**, not a halving.

None of these nine changes a gate, a conclusion or a pre-registered number. They are recorded because
most were about to be re-published in a document being reordered for emphasis, and the reordering is
what caused them to be re-examined. Items 5 and 6 are the two that would have mattered to a referee:
both were places where a hedge had quietly hardened into a claim.

**`README.md` is reordered in the same pass, and the reordering is a framing choice rather than a
claim.** D36's result leads the document: a controlled intervention halves estimation error and
moves the decision by 0.007 on overlapping intervals. Level identification stays in the body as the
mechanism. Nothing about D31's withdrawal is disturbed — no novelty is claimed for anything, and
leading with the flighting result asserts nothing about priority. The one hazard this creates is
that an **exploratory** result now frames a document whose primary results are confirmatory. It is
mitigated three ways and disclosed rather than finessed: D34 pre-committed both readings before the
numbers existed, the exploratory part of the README carries its own heading saying none of it is
pre-registered, and the limitation is stated in the limitations list.

**Two uncorrected residues in the log itself, found by the same sweep and fixed here rather than in
place.**

1. **D29's own attribution is wrong.** It says "§4, D19 and D27 all quote C0's achievable lift as
   1.10% of sales." §4 is *Estimator under test* and contains no achievable-lift figure. The entry
   that does quote 1.10% and is not named is **D23** (line 712). Read D29's list as **D19, D23 and
   D27**.
2. **The retired 1.10% survives in six occurrences across four entries plus one live docstring.**
   D19:577 and its worked example at :588 ("a 40% regret at C0 is 0.44% of sales", which is
   0.40 × 1.10% and would be 0.46% at 1.16%); D23:712; and **D27 twice**, at :814 and again at :817
   inside the very sentence this entry reorders — an earlier draft of this list said "four places"
   and missed the second D27 occurrence. All of those stand as written under this log's own rule,
   and D27 now carries an in-place pointer. **`src/mmm_recovery/estimator.py:801` is not in the
   log** — it states "achievable lift, which runs from 1.10% of sales at C0 to 11.18% at C7" in a
   live docstring with no correction pointer, and it is the only surviving copy of the retired
   figure outside this file. It is recorded here and left for the same pass that resolves the
   stale-docstring item below, because both are code changes and this entry is not.

**Three defects found outside the write-up and deliberately not fixed in this pass**, recorded so
they are not lost. (a) `estimator.py:616` states "G2 coverage on C0 is 32.0%" with no qualifier;
32.0% is D21's pre-D22 ten-seed figure and the current 200-seed value is 0.417, and
`tests/test_estimator.py:655` asserts the literal string `"32.0%"`, so **a green test is holding a
superseded number in place and will fail the moment the docstring is corrected**. (b)
`meridian_anchor.py` has neither a `main()` nor a `__main__` block, so the command the README
documents as a 31-minute run imports the module and exits silently. (c) The bound check and the
flighting check exclude failed solves without writing a `solve_failed` marker, unlike the sweep;
the README now discloses the affected `n` for every arm, but the harnesses should record it.

### D39 — 2026-08-06 — The plateau sweep regenerated. It does not reproduce, and the new figure is authoritative.

**A deliberate, single exception to a scope freeze, with the handling pre-committed in writing
before the run.** The README's centrepiece — "177 of 780 grid points within 1% of the true
parameters' CV score, implied TV contribution £43,938k to £240,522k, a 5.5× spread" — rested on a
harness that was never committed. The exception was justified on the ground that regenerating an
already-published number can only expose an error, never manufacture a finding. The commitment was:
if it fails to reproduce, **the regenerated number is authoritative**, the sweep is *not* tuned to
recover 177, and the discrepancy is logged with both values and the likely cause. It failed to
reproduce. `src/mmm_recovery/plateau.py` is the committed module; `results/plateau_sweep.csv` is its
output.

| | Original (uncommitted harness) | Regenerated (`plateau.py`, noisy) |
|---|---|---|
| Truth's CV RMSE | 3.63160 | **30.94153** |
| Bias at the true hyperparameters | 2.4% | **−7.5%** |
| Grid points within 1% of it | 177 of 780 | **639 of 780** |
| TV contribution across that set | £43,938k – £240,522k | **£15,138k – £248,075k** |
| Spread | 5.5× | **16.4×** |
| \|bias\| across that set | 0.5% – 362.6% | **0.2% – 377.2%** |

**The likely cause is identified rather than guessed: the original predates D22.** Its quoted truth
CV of 3.63160 is reproducible under neither current configuration — with D22's ten-column controls
the noiseless truth scores 0.00002 and the noisy series 30.94153 — and 3.63 £k per week is the size
of the structured residual D22 records for §4's original six-column control list (2.45 £k per week).
The original also reports 2.4% bias at the true hyperparameters, where D22's controls make recovery
*exact*. So the published plateau describes a control block the study abandoned, and it has been
quoted as the current mechanism ever since.

**What is the same, established rather than assumed.** C0, seed 0: the original's maximum bias of
362.6% against £240,522k implies a true TV contribution of £51,989k, and
`truth.incremental_contribution` on C0 seed 0 returns exactly £51,989k. Grid size 780 is matched at
26 α × 30 κ. The 1% band is the original's, applied multiplicatively to the truth's CV.

**What differs, listed because the original cannot be re-read.** The control block (ten columns
against six). The series — the regeneration's primary arm is the **noisy** one, which is what the
estimator sees and what every gate is computed on; the original's CV of 3.63 implies noiseless.
α is swept linearly across `SearchBounds.hill_shape` and κ log-uniformly across
`half_saturation_ratio`, matching how `_draw_hyperparameters` draws them; the original's grid
spacing is not recorded. TV's λ is held at truth along with all four other channels' full triples.

**The regenerated result is stronger than the one it replaces, which is worth saying plainly because
it would be convenient to claim and is therefore the part to check hardest.** Two arms, same grid:

- **Noiseless, correct controls: the truth is uniquely identified.** The best competing grid point
  scores 0.051795 against the truth's 0.00002 — **2,590× worse** — and is only 3.3% wrong. Zero of
  780 fall within 1% of the truth, so *the 1% band is degenerate on this arm* and that number should
  not be quoted as though it were comparable to 639. The honest statement is that with the correct
  functional form, the correct controls and no noise, nothing on the grid comes close.
- **Noisy, at the study's own noise level: the objective goes flat.** 639 of 780 within 1%, and
  **116 of 780 fit strictly better than the truth**. The single best-fitting point on the whole grid
  is **42.8% wrong** (£29,757k against £51,989k). Across the near-tied set the rank correlation
  between CV score and \|bias\| is only +0.416.

**So the mechanism is sharper than "the functional form is unidentified".** It is that noise of the
size §2 specifies — sd 29.32 £k per week against a media series whose own sd is 16.8 — erases the
curvature the level is identified through. The form is identifiable in principle and is not
identifiable from this data. That is a more precise claim than the one it replaces and it is now
reproducible from a clean checkout.

**Downstream edits made in the same pass**, since the retired figures were load-bearing in three
places: `README.md`'s mechanism section and its reading-order note about the missing harness; the
strict-xfail reason string in `tests/test_estimator.py`, which quoted 177 and the £43,938–240,522
range; and `docs/WHEN-TO-TRUST-YOUR-MMM.md`, which quoted "177 of 780" and "£44m to £241m". D23's
and D21's own text stands as written per this log's rule; this entry is their forward pointer.

**Scope closes again here.** No further runs.
