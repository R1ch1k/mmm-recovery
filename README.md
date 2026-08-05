# mmm-recovery

**A pre-registered test of whether Marketing Mix Modelling produces a budget decision worth
acting on. Under the cleanest conditions the study could construct, it does not.**

Ten years of weekly data. No confounding. No collinearity. The estimator given the exactly
correct functional form and a control block that reproduces the true baseline to a residual of
1.4 × 10⁻¹³ £k per week. Under those conditions the fitted model's recommended budget allocation
is **worse than leaving the budget alone in 160 of 200 simulated worlds** (80.0%, 95% CI
73.9–85.0%) when the optimiser has the action space this study pre-registered — and in **48 of
200** (24.0%) under a two-sided ±30% planning guardrail that forbids switching a channel off.

Both figures fail the gate that was fixed before any code ran. The recommendation beats the
status quo 20.0% of the time unconstrained and 76.0% guardrailed, against a pre-registered
threshold of 90%; the guardrailed interval, [69.6%, 81.4%], excludes it. **A governed MMM is far
less destructive and still not good enough.** Every figure below names its action space, because
the difference between them is large and quoting the unconstrained one alone would overstate the
case.

Every way in which this dataset is unrealistic makes the problem **easier** than a real one:
five channels rather than the fifteen to twenty a live model carries once price, promotion,
distribution and competitor pressure are in; ten years of stable channel definitions rather than
the two or three that survive a re-org; no confounded demand; no collinearity. The pre-registered
conditions that would have removed those advantages are the ones the kill criterion blocked.

That MMM has an identification problem is not news, and this study does not claim it as one.
Dew, Padilla and Shchetkina (2024), *Your MMM is Broken*, established that nonlinear and
time-varying specifications are frequently not separately identifiable, that cross-validation
cannot tell them apart, and — the sentence this whole project is a footnote to — that "just
because two models give equivalent predictions under status quo spending patterns does not imply
that they will give the same predictions under intervention" (p. 20). They cite Jin et al. (2017)
for the Hill function being "poorly identified, with different combinations of the parameters
yielding effectively the same function, especially over a finite range" (p. 6) — a paper that was
already in this study's own prior-art section when it predicted the clean condition would pass.
That prediction was wrong, and it was wrong for a reason already in print. It is scored as wrong
below.

What this repository adds is narrower: the same failure measured **inside** a single model family
with the functional form held exactly correct, carried through to a **budget decision**, on a
**pre-registered** grid with thresholds fixed before any code ran, and reproduced by **Google's
shipping Meridian** on default priors.

Third in a series on measurement validity, after [`priced-in`](https://github.com/R1ch1k/priced-in)
(detectable ≠ tradeable) and `marketplace-mispricing` (detectable ≠ actionable). This one is
**attributable ≠ incremental**.

---

## The headline number

The pre-registered decision metric is **allocation regret**: how much of the available gain the
model's advice throws away, where 100% means "no better than doing nothing" and values above
100% mean actively worse. Both the absolute and the normalised form are reported everywhere, per
D19; the absolute figure leads, per D27.

**One unit to fix before the numbers.** Allocation regret is measured in **multiples of the gain
that was available**, and on this dataset the available gain is small — 1.16% of sales. A regret
of 2.355× therefore means "the advice gave up about 2.4 times everything there was to win here."
It does **not** mean 235% of anything was lost. Every regret figure below carries a × for that
reason.

On C0 — the clean condition — at 200 seeds:

- The recommended allocation falls **2.73% of total sales short of the best available
  allocation** (p10–p90: 0.77%–6.15%), against an achievable gain of only 1.16% of sales.
- At the median, following the model's advice leaves sales **1.58% lower than leaving the budget
  alone** — with the middle 80% of runs spanning a 0.42% *gain* to a 4.99% loss.
- As a ratio, that is a **median allocation regret of 2.355× the available gain**.
- It **beats the status quo in 20.0% of runs** (95% CI 15.0–26.1%), against a pre-registered
  threshold of 90%.

## All five gates fail on the clean condition

| Gate | Threshold | C0, 200 seeds | |
|---|---|---|---|
| G1 median \|relative bias\| of contribution | < 0.20 | 0.540 | fail |
| G2 coverage of the nominal 90% interval | ≥ 0.80 | 0.417 | fail |
| G3 median Spearman ρ(true, estimated) | ≥ 0.80 | 0.650 | fail |
| G4 median allocation regret (× the achievable gain) | < 0.20× | 2.355× | fail |
| G5 beats status quo | ≥ 0.90 | 0.200 [0.150, 0.261] | fail |
| G6 placebo | flag rate ≤ 0.10 **and** placebo spend share ≤ 0.02 | not applicable — C0 has no placebo channel | — |

Nine of 200 seeds pass G1 individually. **The decision gates fail harder than the estimation
gates**, which is the ordering the study was built to detect, arriving at the condition that was
supposed to be easy.

**Read G1 as dispersion, not as systematic inflation.** It is a median *absolute* relative error,
so it says nothing about direction, and the direction turns out to be mostly innocent: across 200
seeds the median *signed* error is −6.8% on TV, −11.6% on video, −15.5% on search and −15.0% on
social, with an interquartile range of roughly ±100 percentage points on each. Only OOH is
directionally wrong, at +86.1%. The right reading is "the typical channel's number is off by about
half its own size, in an unpredictable direction" — not "MMM overstates media by 54%."

## The mechanism: a plateau, not a bug

The media *level* is identified only through curvature across the observed spend range. Hill
saturation contains a near-constant as a limiting case, and a constant is collinear with a free
intercept. So a wide set of very different worlds fit the data equally well.

That is directly measurable, and it is the study's centrepiece figure rather than a footnote.
Holding four channels at their true values and sweeping only TV's saturation parameters,
**177 of 780 grid points sit within 1% of the true parameters' cross-validation score, and across
that near-tied set the implied TV contribution ranges from £43,938k to £240,522k** — a 5.5-fold
spread in the answer, with essentially nothing in the data to choose between them.

Three findings rule out the comfortable explanations:

- **It is not a search-power problem, and better optimisation makes recovery *worse*.**
  Nelder–Mead started at *exactly* the true hyperparameters walks away from them: cross-validation
  score improves from 3.63160 to 3.06272 while median contribution bias goes from 2.4% to 57.3%.
  Independently, giving the allocation optimiser more starting points also degrades the decision
  (median regret 2.386 → 2.406 → 2.417 at 8, 16 and 32 starts). The objective is being optimised
  correctly. The objective is the problem.
- **It is not a misspecified control block.** §4's original control list could not represent §2's
  own baseline — a structured residual of 2.45 £k per week. Adding the four missing
  `trend × Fourier` columns drops the projection residual to 1.4 × 10⁻¹³ and makes contribution
  recovery exact at the true hyperparameters. It moved median relative bias from 0.641 to 0.640.
  The last implementation-level explanation was fixed, and the gates still fail (D22).
- **It is not the optimiser extrapolating upward.** At §3's bound of `m_c ≤ 3.0` the model's
  recommendation puts on average **0.035 of 5 channels** at that bound — 7 of 1000 channel-slots —
  so there is essentially nothing at 3× spend for the regret to be an artefact of. Capping every
  channel at 130% of current spend, as Dew et al. do explicitly to avoid extrapolation (p. 26;
  p. 29 fn. 13), leaves the advice still worse than doing nothing at the median (regret 1.407×,
  beats status quo 30.3% [24.3, 37.0]).
  **The bound that does bind is the lower one**, and it is reported here because an earlier
  version of this section wrongly called these solutions interior on the strength of an
  upper-bound count alone. The model zeroes **1.24 of 5 channels** on average. So does the truth —
  it zeroes exactly **1.00**, always OOH, the one channel whose true ROAS is below break-even at
  0.82. Defunding a channel is therefore not the error; defunding more channels than the truth
  does is. See D33.

## Two validity checks that could have overturned this, with their readings fixed in advance

Both were run *because* they might sink the headline, and both had their interpretations written
into the deviations log and committed to git before the numbers existed (D34, D35). One went this
study's way. One did not, and is reported as it landed.

**Does the failure survive a realistic media plan?** The sweep above varies how much spend jitters
and never how it is *shaped* — the simulated spend never goes dark. A flighted plan is
categorically different: it traces the response curve near zero and makes adstock observable,
because sales in a dark week are carryover and nothing else. So the question was whether C0's
spend process was unrepresentatively hard. TV, video and OOH were flighted in independent 2–6 week
bursts at a 50% duty cycle, with each channel's total budget preserved; search and social stayed
always-on. 200 seeds, baseline arm reproducing D23 exactly.

| Gate | C0 baseline | C0 flighted | Threshold |
|---|---|---|---|
| G1 median absolute relative error | 0.540 | **0.309** | < 0.20, still fails |
| G3 median Spearman | 0.650 | **0.800** | ≥ 0.80, **passes** |
| Seeds passing G1 individually | 4.5% | **25.3%** | — |
| **G5 beats status quo** | **0.200** [0.150, 0.261] | **0.207** [0.156, 0.269] | ≥ 0.90, fails |
| Share worse than doing nothing | 0.800 | 0.793 | — |

**Flighting substantially fixes the estimation and does nothing whatsoever for the decision.**
Contribution error nearly halves, five and a half times as many individual runs pass G1, and
channel *ranking* becomes reliable for the first time. And the budget recommendation is exactly as
bad as before: 0.207 against 0.200, intervals almost entirely overlapping. A team that flights its
buys gets a better-estimated model and an equally poor recommendation. That is this project's
thesis appearing a second time, from a direction it was not looking — **estimable ≠ actionable**.

**Does the failure survive a governed action space?** §3 let the optimiser take any channel to
zero or to 3× spend. Real teams operate under planning rules. Adding a two-sided ±30% guardrail —
no channel cut or raised by more than 30% — on both the truth solve and the recommendation:

| m_c range | G4 median regret | G5 beats status quo | 95% CI | Achievable lift | Worse than nothing |
|---|---|---|---|---|---|
| [0.0, 3.0] (§3) | 2.355× | 0.200 | [0.150, 0.261] | 1.160% | 0.800 |
| [0.0, 1.3] | 1.407× | 0.303 | [0.243, 0.370] | 1.126% | 0.697 |
| **[0.7, 1.3]** | 0.501× | **0.760** | **[0.696, 0.814]** | 0.623% | **0.240** |

**This one did not go the study's way and the pre-commitment is honoured rather than
reinterpreted.** D35 committed to "if the advice still loses to the status quo more often than
not, the objection is closed." It does not — guardrailed, it *wins* 76% of the time. The objection
is sustained in part. What survives is that G5 of 0.760 still fails the 0.90 threshold with an
interval excluding it, and G4 still fails by two and a half times.

The mechanism is the interesting part. The guardrail helps so much because it removes the decision
carrying most of C0's headroom — the true optimum defunds OOH entirely, and forbidding that halves
achievable lift from 1.160% to 0.623%. Under the guardrail the *truth* puts 1.97 channels on the
floor and the model 1.52. **The model is not failing by making wild recommendations. It is failing
by being wrong about which channels deserve the money, and a guardrail bounds the consequence of
being wrong without reducing the error.**

## Google's Meridian agrees on the estimation failure and disagrees on the intervals

Run exactly as §4 specifies — C0, 10 seeds, default priors, no tuning, 4 chains — all ten seeds
converged with a worst R-hat of 1.008 against a 1.2 ceiling. Raw output is
`results/meridian_c0.json`.

| Gate | RidgeMMM (200 seeds) | Meridian (10 seeds) | Threshold |
|---|---|---|---|
| G1 median \|relative bias\| | 0.540 fail | **0.456 fail** | < 0.20 |
| G2 coverage | 0.417 fail | **0.820 pass** | ≥ 0.80 |

Two estimators sharing no code, no optimiser and no inference paradigm — penalised least squares
with a random search, against Hamiltonian Monte Carlo with ROI priors — land eight percentage
points apart on G1 and both roughly 2.5× outside it. Meridian's median per-channel bias is +49.7%
on TV, +107.1% on OOH and −29.5% on search; at seed 0 it puts media at **41.9% of sales against a
true 25.0%**.

**Read that as "default priors", not "priors did not help".** Meridian's defaults are only weakly
informative about ROI. They are *not* the experiment-calibrated ROI priors (`roi_calibration`)
that are the tool's main answer to exactly this problem — and which are the mechanism by which
the experiments recommended below would enter an MMM at all. Whether calibrating one channel's
prior from a geo-lift test repairs the allocation is the obvious next experiment, and this study
does not run it.

**The disagreement on G2 is reported as a primary result and it constrains what this study
claims.** G1's failure survives a change of estimator, of optimiser and of inference paradigm, so
it is a property of the method. **G2's failure does not.** §4's bootstrap holds hyperparameters
fixed and resamples residuals around a single point of the plateau, which prices the smallest
component of the error and omits the largest; a Bayesian posterior integrates over the transform
and therefore covers. Reporting both as "MMM fails" would overclaim. Note also what Meridian's
pass does *not* mean: coverage of 0.820 alongside a median bias of 0.456 is an interval wide
enough to contain a badly wrong point estimate — honest uncertainty, not accuracy.

## The kill criterion fired, so the degradation grid was never run

K1 states that if C0 fails any of G1–G5 the harness is presumed wrong and C1–C7 must not be
reported. C0 fails all five. Everything that could make that an artefact was ruled out first — a
13-agent adversarial audit raised nine candidate implementation defects and all nine were refuted;
the control-block error was found and fixed rather than argued away; the one configuration that
appeared to pass turned out to be a minimum over ten noisy draws quoted as though it were a
configuration, and no configuration passes when re-run properly (D23).

**So this is not the study that was designed.** It is a pre-registered stress test that never
reached its stress conditions, because under ten years of clean weekly data the specified model
could not identify how much of sales the media caused. C1 through C7 — collinearity, short
samples, confounded demand, misspecification, placebo channels — are **moot rather than
negative**. Nothing in this repository reports them, and no one should infer them.

## Scoring the pre-registered predictions

§8 recorded directional predictions so the study could be wrong in public. It was.

| Condition | Prediction | Outcome |
|---|---|---|
| **C0** | **Passes all gates** | **WRONG. Fails all five.** |
| C1–C7 | various | Not run. K1 blocked them; they are moot, not negative. |

The C0 prediction is the one that matters and it was wrong on the merits, not by a hair. Worse
than that, it was avoidable: §1 already cited Jin et al. (2017), whose result on Hill
identifiability over a finite range is the mechanism documented above. The prediction was made
without connecting a citation the study had itself listed. That is an error of literature review,
recorded as D25, and it is the single most useful thing in this repository — a pre-registered
prediction, scored wrong, by the person who made it.

---

## Exploratory: how much spend variation would it take? (not pre-registered)

Everything in this section was added **after** C0 failed. It sets no gate, revises no prediction,
and is reported separately for that reason (D26). Identification depends on curvature across the
observed spend range, and the volatility of weekly spend was an assumption of the generating
process rather than a specification — so it is worth asking how much of it a team would need.

C0 with `spend_log_sd` swept, 200 seeds per level. 0.30 is C0 itself and serves as a control that
must reproduce the confirmatory table exactly; it does.

| Metric | sd = 0.15 | **sd = 0.30** | sd = 0.60 | sd = 1.00 | Threshold |
|---|---|---|---|---|---|
| G1 median \|relative bias\| | 0.620 | **0.540** | 0.486 | 0.412 | < 0.20 |
| G2 coverage | 0.529 | **0.417** | 0.258 | 0.184 | ≥ 0.80 |
| G3 median Spearman | 0.600 | **0.650** | 0.700 | 0.900 pass | ≥ 0.80 |
| G4 median regret (× available gain) | 2.944× | **2.355×** | 1.424× | 1.087× | < 0.20× |
| G5 beats status quo | 0.095 | **0.200** | 0.330 | 0.462 | ≥ 0.90 |
| Share of runs worse than doing nothing | 0.905 | **0.800** | 0.670 | 0.538 | — |
| Shortfall from optimum, % of sales | 3.44% | **2.73%** | 1.53% | 1.12% | — |

**The answer is that no realistic amount of observational variation is enough.** At
`spend_log_sd = 1.00` — a single channel's weekly spend spanning a factor of about **16** between
its 10th and 90th percentile, with 40–45% of weeks below half that channel's own average — only
the *ranking* of channels recovers. The advice is still worse than doing nothing in 53.8% of runs
(95% CI 46.8–60.6%).

Note precisely what that level is and is not. It is far more week-to-week *jitter* than any
planner would deliberately introduce. It is **not** more total variation than a flighted plan,
which goes dark and reaches zero; this spend process never does, so flighting is outside the
family swept here entirely. One lognormal is also applied identically to all five channels, and
real portfolios are not homogeneous — always-on search and social run smoother than this, flighted
TV and OOH rougher. The sweep brackets an average; it does not represent a mix.

That converges on Dew et al.'s conclusion by failing rather than by agreeing with it. Their remedy
is a deliberate **spending policy** — their "seesaw" and "maximal separation" tests, alternating
high and low spend for one to two periods (§7) — not the hope that observed variation will
suffice. Their own sweep over spend variance leaves conflation rates unmoved (Table 2, p. 19).
This sweep is the observational counterpart and it points the same way: **if you want an MMM you
can act on, you have to run an experiment, not collect more history.** See D32 for why "converges"
had to be stated this carefully.

One caveat carried from their p. 33: carryover blunts this. Adstock decays of 0.10–0.70 mean the
estimator sees less week-to-week swing than is injected — most of all on TV, at 0.70 — so a
recommendation phrased in raw spend variation will not transfer unchanged to a team with heavy TV
adstock.

Two qualifications a practitioner will raise immediately, and both are fair. This is **not new
advice**: calibrating MMM to lift tests is standard practice and is why Meridian exposes ROI
calibration at all. What is new here is a measurement of what happens when you do not. And
experiments are **unevenly available** — geo holdouts are routine for search and social, and
expensive, slow and noisy for TV and OOH. The cheaper version of the same remedy is a deliberate
flighting pattern, which many brands already run on precisely the channels that are hardest to
test.

---

## What this study does and does not claim

**No novelty is claimed for the decision metrics.** An earlier draft claimed the
beats-status-quo rate as a contribution. A systematic prior-art sweep refuted that and it is
withdrawn (D31). The statistic is imported: probability of improvement across replications is
Agarwal et al.'s (2021) reporting convention, and bounding the probability that a learned policy
is worse than its incumbent is the defining quantity of safe policy improvement (Thomas et al.
2015, 2019). The finding's *shape* is older still — DeMiguel, Garlappi and Uppal (2009) showed an
optimiser losing to a naive incumbent once estimation error swamps the optimisation gain, and
Smith and Winkler (2006) named the mechanism the optimizer's curse. In marketing, Haus's
incrementality-test Monte Carlo already reports the analogous rate: acting on noisy readouts left
the business worse off than doing nothing 38% of the time, across 36 scenarios run a million times
each with true channel performance held fixed. **This study's 20% is worse than Haus's worst arm,
which beat the baseline 62% of the time.**

Haus differs in a way that is the point rather than an excuse: **they fit no model.** Measurement
there is truth plus stipulated noise — no adstock, no saturation, no confounded demand, no design
matrix. Their quantity is the cost of *noise* in an unbiased readout, which more data removes.
This study's is the cost of *non-identification* in a correctly specified estimator, which more
data does not remove.

What is left, stated plainly and without the word "novel":

1. **Within-family.** The failure occurs with the functional form exactly correct and the
   baseline recovered to 1.4 × 10⁻¹³ — not as conflation between two model classes. Dew et al.
   state within-family non-identification as established background (citing Jin et al. 2017) and
   act on it by fixing adstock rather than estimating it; they name the correctly-specified case
   as open (p. 34). This measures its decision consequences.
2. **Production tooling.** Google's shipping Meridian on default priors, not a bespoke framework.
3. **Pre-registration**, with every deviation dated in a log that now runs to 33 entries.
4. **Ground truth by intervention.** True contribution is `sales(spend) − sales(spend_c := 0)`
   evaluated on the generating process, never a comparison of β̂ to β.

Pathak, Jeunen and Lambert (2026) independently name this design as their own open problem: "An
important next step is a semi-synthetic benchmark with known response structure and oracle regret."

## Limitations

Read these as binding, not as ritual.

- **One DGP family.** A different generating process may well be identifiable. Nothing here
  licenses a claim about MMM in general beyond this family.
- **`spend_log_sd = 0.30` was an assumption**, not a specification. The exploratory sweep above
  shows how much it matters — enough to move every gate, nowhere near enough to pass one.
- **Meridian was run national, single-geo, with no control columns, at default priors.** Its
  baseline spline is more flexible than the generating process's, and — more importantly — the
  geo hierarchy that is the package's main identifying mechanism is switched off, because this
  DGP produces one national series. That is a supported configuration but not the one Meridian is
  designed around. Read the anchor as "a Bayesian MMM with ROI priors agrees on G1", not as
  "Meridian as deployed agrees".
- **Experiment-calibrated priors are untested.** This study's own recommendation — run an
  experiment — enters an MMM through a tightened ROI prior. Meridian supports exactly that and it
  was not tried, so the recommendation is argued rather than demonstrated.
- **The pre-registered action space is not an implementable one, and this materially moves the
  headline.** §3 lets the optimiser zero a channel, and the true optimum does exactly that to OOH
  (true ROAS 0.82 against 1.73–2.30 elsewhere). Under a governed ±30% guardrail the
  worse-than-nothing rate falls from 80% to 24%. The gates still fail, but any single number
  quoted from this study must name which action space produced it.
- **One DGP family, and now one spend *shape* beyond it.** Flighting was tested (above) and did
  not rescue the decision. Other plan shapes — seasonal pulsing, always-on with occasional
  blackouts, geo-staggered launches — were not.
- **The flighted arm's achievable lift is half the baseline's** (0.54% against 1.16%), so its G4
  is not comparable to C0's and is not compared. Only G1, G3 and G5 carry across the two arms.
- **Robyn is untested.** The Python port is an LLM-translated beta, so a failure there would be
  uninterpretable and no claim is made about it.
- **Global optimality is empirically supported, not proven** (D17). The surface is non-concave; the
  returned optimum matches a 256-start reference on 100 of 100 trials across five representative
  cells. That is a strong prior, not a guarantee, and every regret figure inherits its status.
- **Hyperparameter search ranges were assumed**, not specified in the pre-registration.
- **G2's failure is estimator-specific**, not structural. Meridian passes it. See above.
- **The decision gates are sensitive to an optimiser convention that was not recorded at the
  time** (D28). D23's harness was never committed; a faithful re-implementation reproduces G1–G3
  to the printed digit and lands G4 at 2.386 versus 2.355 depending on the SLSQP start seed. The
  convention is now fixed and asserted, but the sensitivity — 1.3% relative on median regret — is
  the honest error bar on that class of number.
- **Two of 800 cells in the exploratory sweep fail to solve** and are excluded, with the adverse
  bound reported alongside (D30).
- **The prior-art sweep has a known hole.** The pass over pre-1990 marketing science — Little,
  Lodish, Hanssens, and the normative decision-model literature — did not complete. If the rate
  form turns out to be fifty years old, that is where it will be found.
- **Xu, Thomadsen and Zhang (2025)** argue the ordinary bootstrap is invalid for post-optimisation
  quantities. G5's interval here is a Wilson score interval on a proportion over independent
  seeds, which sidesteps that objection, but G2's block bootstrap does not and the paper is unread.

## Reproducing

No network calls, no API keys, no data downloads. Every stochastic step takes an explicit seed.

```bash
uv sync
uv run pytest                              # 416 tests: 415 pass, 1 strict xfail
uv run python -m mmm_recovery.sweep        # the exploratory sweep, ~1 min
uv run python -m mmm_recovery.robustness   # bound check + the ±30% guardrail, ~2 min
uv run python -m mmm_recovery.flighting    # the flighted-spend validity check, ~2 min
```

The sweep refuses to write results unless its `spend_log_sd = 0.30` column reproduces D23's
published table to within 5 × 10⁻⁴. That assertion is the reason the other three columns can be
believed.

The Meridian anchor needs the optional extra and takes about 31 minutes:

```bash
uv sync --extra meridian
uv run python -m mmm_recovery.meridian_anchor
```

## Reading order

1. [`PREREGISTRATION.md`](PREREGISTRATION.md) — the binding specification, and the deviations log
   D1–D33 that records every departure from it with a date and a reason.
2. [`docs/WHEN-TO-TRUST-YOUR-MMM.md`](docs/WHEN-TO-TRUST-YOUR-MMM.md) — one page, no equations,
   for a marketing reader.
3. [`results/`](results/) — the raw per-seed output behind the C0 table, the exploratory sweep
   and the bound check. **Not** behind the mechanism section: the plateau sweep and the
   Nelder–Mead diagnostic were run by a harness that was never committed, and they survive only
   as numbers in the deviations log and in the strict-xfail reason string that `pytest -rx`
   prints. That is a real reproducibility gap and re-running them as a committed module is the
   first outstanding task.

## References

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. (2021). *Deep
  Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS. arXiv:2108.13264.
- Chan, D., & Perry, M. (2017). *Challenges and Opportunities in Media Mix Modeling.* Google Inc.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). *Optimal Versus Naive Diversification: How
  Inefficient Is the 1/N Portfolio Strategy?* Review of Financial Studies, 22(5), 1915–1953.
- Dew, R., Padilla, N., & Shchetkina, A. (2024). *Your MMM is Broken: Identification of Nonlinear
  and Time-varying Effects in Marketing Mix Models.* arXiv:2408.07678.
- Haus. *Fast, Confident, and Wrong: The Risk of Noisy Incrementality Tests.*
- Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). *Bayesian Methods for Media Mix
  Modeling with Carryover and Shape Effects.* Google Inc.
- Pathak, N., Jeunen, O., & Lambert, E. (2026). *Auditing Marketing Budget Allocation with
  Hindsight Regret.* arXiv:2604.25977.
- Smith, J. E., & Winkler, R. L. (2006). *The Optimizer's Curse: Skepticism and Postdecision
  Surprise in Decision Analysis.* Management Science, 52(3), 311–322.
- Thomas, P. S., Theocharous, G., & Ghavamzadeh, M. (2015). *High Confidence Policy Improvement.*
  ICML.
- Thomas, P. S., Castro da Silva, B., Barto, A. G., Giguere, S., Brun, Y., & Brunskill, E. (2019).
  *Preventing undesirable behavior of intelligent machines.* Science, 366(6468), 999–1004.
- Xu, S., Thomadsen, R., & Zhang, D. (2025). *The Winner's Curse in Data-Driven Decision-Making:
  Evidence and Solutions.* SSRN 5930537. **Cited from its abstract; the full text was not
  obtainable and the claim below is unverified.**
- Zhang, S., & Vaver, J. (2017). *Introduction to the Aggregate Marketing System Simulator.*
  Google Inc.
