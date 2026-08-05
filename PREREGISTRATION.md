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
