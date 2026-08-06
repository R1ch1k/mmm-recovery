# mmm-recovery

**A pre-registered test of whether Marketing Mix Modelling produces a budget decision worth acting
on. Under the cleanest conditions the study could construct it does not — and the one intervention
that most improves the model's estimates leaves the budget advice where it was.**

Flight three of five channels — two-to-six-week bursts, dark weeks between, the same annual budget
concentrated into fewer weeks, the way brand media is actually bought — and *contribution error*
improves more than anywhere else in this study. On a flighted variant of the clean condition, the
median absolute relative error in channel contribution falls from **0.540 to 0.309** and the share of
runs whose median channel error lands inside the pre-registered 20% rises from **4.5% to 25.3%**.
Two counterweights belong in the same breath rather than in a footnote: interval coverage gets
*worse* (G2 0.417 → 0.323), and channel ranking **reaches** its threshold rather than clearing it —
G3 lands exactly on 0.800, with 104 of 198 seeds at or above 0.80, on a statistic that moves in
steps of 0.1 across five channels.

The budget decision shows **no measurable improvement**. The recommendation beats leaving the budget
alone in 20.7% of runs against 20.0%. Paired over the 198 seeds both arms solved — where the
baseline is 19.7% — the difference is **+1.0 percentage point, 95% CI [−7.1, +9.1]**, McNemar
*p* = 0.90. It is worse than doing nothing in 79.3% of runs against 80.0%. Both of those are §3's
unconstrained action space — under the ±30% guardrail described below the worse-than-nothing rate is
24%. This is a null from a design that could not have detected a change smaller than about eight
points, and it is reported as that rather than as a demonstration of no effect.

**Under flighting you can learn which channels are better without learning how much better, and
better ranking did not buy a better budget.** That is this series' framing — *attributable ≠
incremental* — one step further in: **estimable ≠ actionable**. What this does **not** establish is
the tidier mechanism, that the decision fails *because* the sizing fails: elsewhere in the study a
world with better ranking and worse sizing produces a markedly better decision. That reconciliation
is set out in full where the sweep is, not buried.

**What licenses leading with this, and what does not.** D34 fixed *before the numbers existed* how a
G1 pass or a G1 failure would be read. It pre-committed **nothing** about the decision metrics, so
the estimation-versus-decision split above is post-hoc, and the "pre-committed" badge attaches to the
G1 half only. The guardrail check below carries a genuine pre-commitment tied to G5 (D35). The
baseline arm reproduces the confirmatory result to the printed digit, so the spend process is the
only thing that differs between the two arms.

**The identification failure underneath it is not new, and nothing in this repository claims to be
first.** Dew, Padilla and Shchetkina (2024), *Your MMM is Broken*, established that nonlinear and
time-varying specifications are frequently not separately identifiable, that cross-validation cannot
tell them apart, and — the sentence this whole project is a footnote to — that "just because two
models give equivalent predictions under status quo spending patterns does not imply that they will
give the same predictions under intervention" (p. 20). They cite Jin et al. (2017) for the Hill
function being "poorly identified, with different combinations of the parameters yielding
effectively the same function, especially over a finite range" (p. 6). Chan and Perry (2017) set out
the same class of problem for Google's own MMM practice and recommended evaluating MMM through
simulators; Zhang and Vaver's (2017) Aggregate Marketing System Simulator, open-sourced as
`google/amss`, is that recommendation carried out, and this project is a Python-native descendant of
it. What is added here is narrower: the failure measured **inside a single model family** with the
functional form held exactly correct, carried through to a **budget decision**, on a
**pre-registered** grid with thresholds fixed before any code ran, and reproduced by **Google's
shipping Meridian** on default priors.

**Every figure in this document names its action space, because the difference between them is
large.** The pre-registered optimiser (§3) may take any channel to zero or to 3× its current spend,
and under it the advice is worse than doing nothing in **160 of 200** simulated worlds (80.0%, 95%
CI 73.9–85.0%). Under a two-sided ±30% planning guardrail — no channel switched off, no channel
raised by more than a third — it is worse in **48 of 200**, and it *beats* the status quo 76.0% of
the time. That guardrail check was pre-committed and it did not go this study's way; it is reported
as it landed (D35, D37). What survives it is that 76.0% still fails the pre-registered 90% threshold, on an interval of
[69.6%, 81.4%] that excludes it. **A governed MMM is far less destructive and still not good
enough.** The unconstrained figure never appears here without the governed one beside it.

Third in a series on measurement validity, after [`priced-in`](https://github.com/R1ch1k/priced-in)
(detectable ≠ tradeable) and `marketplace-mispricing` (detectable ≠ actionable).

---

# Confirmatory results

Everything in this part of the document is the pre-registered study: condition **C0**, the
pre-registered action space `m_c ∈ [0, 3.0]`, 200 seeds, against thresholds fixed before any code
ran. C1 through C7 were never run; [the kill criterion](#the-kill-criterion-fired-so-the-degradation-grid-was-never-run)
below explains why.

## What C0 gives the model

Ten years of weekly data. No confounding. No collinearity. The estimator given the exactly correct
functional form and a control block that reproduces the true baseline to a residual of
1.4 × 10⁻¹³ £k per week.

Every way in which this dataset is unrealistic makes the problem **easier** than a real one: five
channels rather than the fifteen to twenty a live model carries once price, promotion, distribution
and competitor pressure are in; ten years of stable channel definitions rather than the two or three
that survive a re-org; no confounded demand; no collinearity. The pre-registered conditions that
would have removed those advantages are the ones the kill criterion blocked.

## The headline number

The pre-registered decision metric is **allocation regret**: how much of the available gain the
model's advice throws away, where 1.0× means "no better than doing nothing" and values above 1.0×
mean actively worse. Both the absolute and the normalised form are reported everywhere, per D19.

**One unit to fix before the numbers.** Allocation regret is measured in **multiples of the gain
that was available**, and on this dataset the available gain is small — 1.16% of sales. A regret of
2.355× therefore means "the advice gave up about 2.4 times everything there was to win here." It
does **not** mean 235% of anything was lost. Every regret figure below carries a × for that reason.

**And two absolute figures, which answer different questions and are not interchangeable.** The
status-quo figure leads throughout this document, because *act on the model or leave the budget
alone* is the decision a practitioner actually faces; the optimum is not on the menu, since if it
were knowable there would be no study (D27, D29, D38). On C0, at 200 seeds, in §3's action space:

- Following the model's advice leaves sales **1.58% of total sales lower than leaving the budget
  alone** — with the middle 80% of runs spanning a 0.42% *gain* to a 4.99% loss.
- The same allocation falls **2.73% of total sales short of the best available allocation**
  (p10–p90: 0.77%–6.15%), against an achievable gain of only 1.16% of sales.
- As a ratio, that is a **median allocation regret of 2.355× the available gain**.
- It **beats the status quo in 20.0% of runs** (95% CI 15.0–26.1%), against a pre-registered
  threshold of 90%.

> Both absolute figures are medians of the per-seed quantity across 200 seeds, not quantities derived
> from the two headline medians — the median of a ratio is not the ratio of medians. It shows in the
> status-quo figure: measured 1.585%, against (2.355 − 1) × 1.160% = 1.572% derived, which is the
> difference between printing 1.58% and 1.57%. The shortfall is unaffected at this precision (2.727%
> measured, 2.732% derived, both printing 2.73%). No conclusion turns on 0.013 of a percentage point,
> but the printed figures are the measured ones (D38).

## All five gates fail on the clean condition

| Gate | Threshold | C0, 200 seeds, `m_c ∈ [0, 3.0]` | |
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

**Read G1 as dispersion, not as systematic inflation.** It is a median *absolute* relative error, so
it says nothing about direction, and the direction turns out to be mostly innocent: across 200 seeds
the median *signed* error is −6.8% on TV, −11.6% on video, −15.5% on search and −15.0% on social,
with an interquartile range between 94 and 110 percentage points wide on each of those four. Only
OOH is directionally wrong, at +86.1%, and its spread is wider again at 259 points. The right
reading is "the typical channel's number is off by about half its own size, in an unpredictable
direction" — not "MMM overstates media by 54%."

## The mechanism: a plateau, not a bug

The media *level* is identified only through curvature across the observed spend range. Hill
saturation contains a near-constant as a limiting case, and a constant is collinear with a free
intercept. So a wide set of very different worlds fit the data equally well. This is the
within-family non-identification Dew et al. treat as established background, citing Jin et al.
(2017) on the Hill function over a finite range; they act on it by fixing adstock rather than
estimating it, and name the correctly-specified case as open (p. 34).

That is directly measurable, and it is the study's centrepiece figure rather than a footnote.
Holding four channels at their true values and sweeping only TV's saturation parameters over a
26 × 30 grid, on C0 seed 0:

| | Noiseless sales | Noisy sales, as the estimator sees them |
|---|---|---|
| Truth's CV RMSE | 0.00002 | 30.94153 |
| Best competing grid point | 0.051795 — **2,590× worse** | 30.91330 — **better than the truth** |
| Grid points fitting better than the truth | 0 of 780 | **116 of 780** |
| Grid points within 1% of the truth's score | 0 of 780 (band degenerate) | **639 of 780** |
| Implied TV contribution across the near-tied set | — | **£15,138k to £248,075k, a 16.4× spread** |
| Error of the single best-fitting point | −3.3% | **−42.8%** |

**Read the two columns against each other, because the contrast is the mechanism, and start with the
noiseless control.** With the correct functional form, the correct controls and no noise, the truth
is uniquely identified — the best competing grid point scores **2,590× worse**, and the runner-up is
only **3.3%** wrong. **So the functional form is identifiable in principle, and simply not
identifiable from this data.** That control is what forecloses the obvious objection, which is that
a sweep over Hill parameters was always going to look flat.

Now add noise at the level §2 specifies — sd 29.32 £k per week, against a true media series whose own
standard deviation is 16.8. **116 of the 780 grid points fit the data strictly *better* than the
truth does, and the best-fitting point on the entire grid is 42.8% wrong.** The objective around them
has gone flat: 639 of 780 transforms sit within 1% of the truth's score, and across that near-tied
set the correlation between CV score and absolute error is only +0.416. Zero of 780 in the noiseless
column is a degenerate band — 1% of a score that is essentially zero — and is reported as such rather
than as a figure comparable to 639.

So the claim is not "Hill saturation is unidentifiable". It is the sharper and more uncomfortable
one: noise of an entirely ordinary size erases the curvature the level is identified through, in a
world where the analyst has the functional form exactly right.

> **These numbers replace the ones this section carried until D39, and the replacement was
> pre-committed.** The README previously reported "177 of 780 … £43,938k to £240,522k, a 5.5-fold
> spread" from a harness that was never committed. Regenerating it as
> [`plateau.py`](src/mmm_recovery/plateau.py) did not reproduce those figures. The cause is
> identified: the original's quoted truth CV of 3.63160 with 2.4% bias matches neither current arm,
> and 3.63 £k per week is the size of the structured residual D22 records for the **superseded
> six-column control block** — so the published plateau described a configuration the study
> abandoned. The regenerated figure is authoritative, the sweep was not tuned to recover 177, and
> both values are in D39 with the full list of what differs.

Three findings rule out the comfortable explanations:

- **It is not a search-power problem, and better optimisation makes recovery *worse*.**
  Nelder–Mead started at *exactly* the true hyperparameters walks away from them: cross-validation
  score improves from 3.63160 to 3.06272 while median contribution bias goes from 2.4% to 57.3%.
  Independently, giving the allocation optimiser more starting points also degrades the decision
  (median regret 2.386× → 2.406× → 2.417× at 8, 16 and 32 starts). The objective is being optimised
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
  **The bound that does bind is the lower one**, and it is reported here because an earlier version
  of this section wrongly called these solutions interior on the strength of an upper-bound count
  alone. The model zeroes **1.24 of 5 channels** on average. So does the truth — it zeroes exactly
  **1.00** in all 200 seeds, and that one is OOH, the only channel whose true ROAS is below
  break-even — 0.82 at seed 0, with a 50-seed median of 0.84 and a maximum of 0.85. (The committed CSV records the count, not the identity; that it is always OOH
  follows from the ROAS ordering, per D20.) Defunding a channel is therefore not the error;
  defunding more channels than the truth does is. See D33.

## Google's Meridian agrees on the estimation failure and disagrees on the intervals

Run exactly as §4 specifies — C0, 10 seeds, default priors, no tuning, 4 chains — all ten seeds
converged with a worst R-hat of 1.008 against a 1.2 ceiling. Raw output is
`results/meridian_c0.json`.

| Gate | RidgeMMM (200 seeds) | Meridian (10 seeds) | Threshold |
|---|---|---|---|
| G1 median \|relative bias\| | 0.540 fail | **0.456 fail** | < 0.20 |
| G2 coverage | 0.417 fail | **0.820 pass** | ≥ 0.80 |

Two estimators sharing no code, no optimiser and no inference paradigm — penalised least squares
with a random search, against Hamiltonian Monte Carlo with ROI priors — land eight percentage points
apart on G1 and both roughly 2.5× outside it. Meridian's median per-channel bias is +49.7% on TV,
+107.1% on OOH and −29.5% on search; at seed 0 it puts media at **41.9% of sales against a true
25.0%**. That last figure is the one number in this section not recoverable from
`results/meridian_c0.json`, which stores per-seed G1, coverage, R-hat and the per-channel relative
bias but no media share; it was read from the run's console output.

**Read that as "default priors", not "priors did not help".** Meridian's defaults are only weakly
informative about ROI. They are *not* the experiment-calibrated ROI priors (`roi_calibration`) that
are the tool's main answer to exactly this problem — and which are the mechanism by which the
experiments recommended below would enter an MMM at all. Whether calibrating one channel's prior
from a geo-lift test repairs the allocation is the obvious next experiment, and this study does not
run it.

**The disagreement on G2 is reported as a primary result and it constrains what this study claims.**
G1's failure survives a change of estimator, of optimiser and of inference paradigm, so it is a
property of the method. **G2's failure does not.** §4's bootstrap holds hyperparameters fixed and
resamples residuals around a single point of the plateau, which prices the smallest component of the
error and omits the largest; a Bayesian posterior integrates over the transform and therefore
covers. Reporting both as "MMM fails" would overclaim. Note also what Meridian's pass does *not*
mean: coverage of 0.820 alongside a median bias of 0.456 is an interval wide enough to contain a
badly wrong point estimate — honest uncertainty, not accuracy.

---

# Exploratory results

**Nothing in this part of the document is pre-registered.** All three checks were added *after* C0
failed. They set no gate, revise no §8 prediction, and are reported separately for that reason (D26,
D34, D35). Two of the three — flighting and the guardrail — had their readings written into the
deviations log and committed before the numbers existed, which is what licenses the opening section
of this README to lead with one of them; that is a stronger warrant than ordinary post-hoc analysis,
and it is still not pre-registration.

## How much spend variation would it take? Not an achievable amount

Identification depends on curvature across the observed spend range, and the volatility of weekly
spend was an assumption of the generating process rather than a specification — so it is worth
asking how much of it a team would need. C0 with `spend_log_sd` swept, 200 seeds per level. 0.30 is
C0 itself and serves as a control that must reproduce the confirmatory table exactly; it does.

| Metric | sd = 0.15 | **sd = 0.30** | sd = 0.60 | sd = 1.00 | Threshold |
|---|---|---|---|---|---|
| Loss against doing nothing, % of sales | 2.24% | **1.58%** | 0.46% | 0.09% | — |
| Shortfall from the optimum, % of sales | 3.44% | **2.73%** | 1.53% | 1.12% | — |
| G1 median \|relative bias\| | 0.620 | **0.540** | 0.486 | 0.412 | < 0.20 |
| G2 coverage | 0.529 | **0.417** | 0.258 | 0.184 | ≥ 0.80 |
| G3 median Spearman | 0.600 | **0.650** | 0.700 | 0.900 pass | ≥ 0.80 |
| G4 median regret (× available gain) | 2.944× | **2.355×** | 1.424× | 1.087× | < 0.20× |
| G5 beats status quo | 0.095 | **0.200** | 0.330 | 0.462 | ≥ 0.90 |
| Share of runs worse than doing nothing | 0.905 | **0.800** | 0.670 | 0.538 | — |

**The answer is that no realistic amount of observational variation is enough.** At
`spend_log_sd = 1.00` — a single channel's weekly spend spanning a factor of about **13** between
its 10th and 90th percentile, with about 42% of weeks below half that channel's own average — only
the *ranking* of channels recovers. The advice is still worse than doing nothing in 53.8% of runs (95%
CI 46.8–60.6%), and it still loses money against the status quo at the median.

Note precisely what that level is and is not. It is far more week-to-week *jitter* than any planner
would deliberately introduce. It is **not** more total variation than a flighted plan, which goes
dark and reaches zero; this spend process never does, so flighting is outside the family swept here
entirely, and it is tested separately below. One lognormal is also applied identically to all five
channels, and real portfolios are not homogeneous — always-on search and social run smoother than
this, flighted TV and OOH rougher. The sweep brackets an average; it does not represent a mix.

That converges on Dew et al.'s conclusion by failing rather than by agreeing with it. Their remedy
is a deliberate **spending policy** — their "seesaw" and "maximal separation" tests, alternating high
and low spend for one to two periods (§7) — not the hope that observed variation will suffice. Their
own sweep over spend variance leaves conflation rates unmoved (Table 2, p. 19). This sweep is the
observational counterpart and it points the same way: **if you want an MMM you can act on, you have
to run an experiment, not collect more history.** See D32 for why "converges" had to be stated this
carefully.

One caveat carried from their p. 33: carryover blunts this. Adstock decays of 0.10–0.70 mean the
estimator sees less week-to-week swing than is injected — most of all on TV, at 0.70 — so a
recommendation phrased in raw spend variation will not transfer unchanged to a team with heavy TV
adstock.

Two qualifications a practitioner will raise immediately, and both are fair. This is **not new
advice**: calibrating MMM to lift tests is standard practice and is why Meridian exposes ROI
calibration at all. What this adds is a measurement of what happens when you do not — a
measurement, not a priority claim; see the claims section below. And
experiments are **unevenly available** — geo holdouts are routine for search and social, and
expensive, slow and noisy for TV and OOH. The cheaper version of the same remedy is a deliberate
flighting pattern, which many brands already run on precisely the channels that are hardest to test.

## Does a real media plan rescue it? It rescues the estimate and not the decision

This is the check the opening of this README leads with, and it was run *because* it might sink the
headline. The sweep above varies how much spend jitters and never how it is *shaped* — the simulated
spend never goes dark. A flighted plan is categorically different: it traces the response curve near
zero and makes adstock observable, because sales in a dark week are carryover and nothing else. So
the question was whether C0's spend process was unrepresentatively hard.

TV, video and OOH were flighted in independent 2–6 week bursts at a 50% duty cycle, with each
channel's total budget preserved; search and social stayed always-on. 200 seeds per arm, with the
baseline arm reproducing D23 exactly. Two flighted cells failed SLSQP and their rows are blank
throughout, so **every** flighted figure below — estimation as well as decision — is over 198 seeds,
not 200. Three rows — G2 coverage, achievable lift and realised media share — are computed and
printed by the module but **not** persisted to `results/flighting_check.csv`, which carries only
bias, Spearman, regret and beats-status-quo. Those three are reproducible by re-running the check,
not by reading the committed file.

| Gate | C0 baseline | C0 flighted | Threshold |
|---|---|---|---|
| G1 median absolute relative error | 0.540 | **0.309** | < 0.20, still fails |
| G2 coverage | 0.417 | 0.323 | ≥ 0.80, fails and worse |
| G3 median Spearman | 0.650 | **0.800** | ≥ 0.80, **passes** |
| Seeds passing G1 individually | 4.5% | **25.3%** | — |
| G4 median regret | 2.355× | 2.894× | < 0.20×, not comparable across arms |
| **G5 beats status quo** | **0.200** [0.150, 0.261] | **0.207** [0.156, 0.269] | ≥ 0.90, fails |
| Share worse than doing nothing | 0.800 | 0.793 | — |
| Achievable lift | 1.16% | 0.54% | — |
| Realised media share | 0.2500 | 0.2382 | — |

**D34's pre-committed reading applies: G1 still fails, so C0's spend process was not the cause of
the failure, and the headline stands with one fewer attack surface.** The objection that a lognormal
jitter is an unrepresentatively hard world has been tested rather than argued.

**But the referee's intuition was half right, and that half is the finding.** Flighting carries real
identifying information — G1 nearly halves, five and a half times as many individual seeds pass it,
and G3 reaches its threshold. Anyone claiming this study shows spend variation is irrelevant would
be misreading it. G3 also passes at `spend_log_sd = 1.00` in the sweep above, so flighting is not the
only place in this study where ranking recovers; it is the only place where ranking recovers under
something a media team would actually choose to do.

And the budget recommendation shows no measurable improvement: 0.207 against 0.200, and over the 198
seeds both arms solved a paired difference of +1.0 percentage point (34 seeds flip toward the
flighted arm, 32 against it) with a 95% interval of [−7.1, +9.1] and McNemar *p* = 0.90. The
worse-than-nothing share is 0.793 against 0.800. A team that flights its buys gets a
better-estimated model and a recommendation this design cannot show to be any better.

**The honest reconciliation, because the sweep two sections up cuts against the tidy version of this
story.** At `spend_log_sd = 1.00` the *ranking* is better than under flighting (G3 0.900 against
0.800) and the *sizing* is worse (G1 0.412 against 0.309), and yet the decision there is markedly
better: G5 0.462 against 0.207, worse-than-nothing 0.538 against 0.793. So "the decision fails
because the sizing fails" is **not** a mechanism this study has established — decision quality does
not track either gate monotonically across arms. What is established is narrower and is what the
opening claims: under flighting specifically, ranking became reliable and the beat-the-status-quo
rate did not measurably move. Anyone wanting the mechanism would have to run the decomposition
directly — true ranking with estimated magnitudes, and the converse — and this study does not.

One more caveat on comparing the arms in absolute terms. Flighting changes the spend process, so the
status-quo world itself differs between the arms — realised media share falls 0.2500 → 0.2382 — and
the flighted arm's achievable lift is not persisted per seed. **Only the rates carry across the two
arms; neither absolute loss figure does.**

The two arms' achievable lift differs (1.16% against 0.54%) because concentrating a fixed budget
into fewer weeks pushes those weeks further along the saturation curve, so regret's denominator is
not the same quantity in the two arms and the two G4 figures **must not be compared**. Only G1, G2,
G3 and G5 carry across.

## Does a governed action space rescue it? Substantially, and not enough

§3 let the optimiser take any channel to zero or to 3× spend. Real teams operate under planning
rules. Adding a two-sided ±30% guardrail — no channel cut or raised by more than 30% — on both the
truth solve and the recommendation, using the same fitted surfaces and no refit:

| m_c range | G4 median regret | G5 beats status quo | 95% CI | Achievable lift | Worse than nothing | n |
|---|---|---|---|---|---|---|
| [0.0, 3.0] (§3) | 2.355× | 0.200 | [0.150, 0.261] | 1.160% | 0.800 | 200 |
| [0.0, 2.0] | 2.284× | 0.216 | [0.165, 0.278] | 1.160% | 0.784 | 199 |
| [0.0, 1.5] | 1.501× | 0.270 | [0.213, 0.337] | 1.160% | 0.730 | 196 |
| [0.0, 1.3] | 1.407× | 0.303 | [0.243, 0.370] | 1.126% | 0.697 | 198 |
| **[0.7, 1.3]** | 0.501× | **0.760** | **[0.696, 0.814]** | 0.623% | **0.240** | 200 |

The two intermediate upper bounds are shown because the ladder is monotone: every step that narrows
the action space improves the decision, which is the mechanism claim below stated as a gradient
rather than as two endpoints. The varying `n` is the SLSQP failures described in the disclosure
note at the end of this section.

**This one did not go the study's way and the pre-commitment is honoured rather than
reinterpreted.** D35 committed to "if the advice still loses to the status quo more often than not
under a rule that forbids the destructive move, the objection is closed." It does not — guardrailed,
it *wins* 76% of the time, and at the median it gains **0.30% of total sales against leaving the
budget alone** rather than losing 1.58% — while still falling **0.32% of total sales short of the
best allocation reachable inside the guardrail**. The objection is **sustained in part**. What
survives is that
G5 of 0.760 still fails the 0.90 threshold with an interval excluding it, and G4 still fails 0.20× by
two and a half times.

The mechanism is the interesting part. The guardrail helps so much because it removes the decision
carrying most of C0's headroom — the true optimum defunds OOH entirely (D20), and forbidding that
cuts achievable lift from 1.160% to 0.623%, a reduction of 46%. Under the guardrail the *truth* puts
1.97 channels on the floor and the model 1.52. **The model is not failing by making wild recommendations. It is
failing by being wrong about which channels deserve the money, and a guardrail bounds the
consequence of being wrong without reducing the error.**

That is also why the guardrail is not a fix. Of the 1.160% of sales that reallocation could win in
§3's action space, only **0.623%** remains reachable once a planner's rules are applied — so the
downside of acting on a badly identified model dominates the upside of acting on a well identified
one.

**Solve failures across all three exploratory harnesses, disclosed together.** SLSQP fails on a
small number of cells, always on the fitted surface and never on the true one, and every failed cell
is excluded from its medians (D30). The sweep loses 2 of 800 cells and writes a `solve_failed`
marker for them. The bound check loses 7 of 1000 and the flighting check 2 of 400, and **neither of
those two writes a marker** — their failures are visible only as missing or blank seed rows, which
is why the `n` column is printed above. The affected arms are `[0.0, 1.3]` (n=198), `[0.0, 1.5]`
(n=196), `[0.0, 2.0]` (n=199) and the flighted arm (n=198). Every published figure is computed on
the reduced denominator, so 25.3% of the flighted arm is 50 of 198 rather than of 200; a reader
recomputing from the raw file at face value would get 25.0%.

## Three arms side by side: what actually moved the decision

Each section above changes one thing at a time. Laid alongside each other and read on **G5** — the
rate D35 fixed in advance as the quantity that is comparable across arms, where regret is not — the
three interventions separate cleanly.

| Arm | What was changed | G1 | G5 beats status quo | 95% CI | n |
|---|---|---|---|---|---|
| C0 baseline | — | 0.540 | 0.200 | [0.150, 0.261] | 200 |
| Flighted | the **shape** of spend | **0.309** | 0.207 | [0.156, 0.269] | 198 |
| `spend_log_sd = 1.00` | the **amount** of spend variation | 0.412 | **0.462** | [0.394, 0.532] | 199 |
| `m_c ∈ [0.7, 1.3]` | the optimiser's **action space** | 0.540 *(identical by construction: re-solves the same fitted surfaces)* | **0.760** | [0.696, 0.814] | 200 |

**The last row's G1 is not a fourth measurement.** The guardrail re-solves the **same fitted
surfaces** with no refit, so its 0.540 is the baseline's own number reappearing — not an estimate
that happened to land in the same place. The G1 column therefore holds three measurements across
four rows, and the table is a controlled contrast rather than four independent readings.

**Two interventions moved the decision, and better estimation was not one of them.** More spend
variation took G5 from 0.200 to 0.462 and bounding the optimiser took it from 0.200 to 0.760; both
intervals clear the baseline's entirely. Flighting produced the **best contribution estimates
anywhere in this study** — G1 0.540 → 0.309, a 43% reduction — and moved G5 by 0.007, on intervals
that almost entirely overlap.

Put the two extremes side by side and the point is as sharp as this study can make it. **The arm
that moved the decision most changed the estimate by exactly zero. The arm that improved the
estimate most moved the decision by 0.007, on overlapping intervals.** That is *estimable ≠
actionable* as measured arms rather than as an argument.

**The honest limit on the middle row, in the same breath.** `spend_log_sd = 1.00` is not a setting
any team can choose. It is weekly spend spanning a factor of about 13 between its quiet and its heavy
weeks, with about 42% of weeks below half the channel's own average — an implausibly large
*observational* spend distribution, well beyond anything a planner produces — and even there the
advice is still worse than doing nothing in 53.8% of runs and G5 fails its threshold by a wide
margin. So that row demonstrates **that variation is what identifies**. It is not a recommendation.
The actionable form of the same finding is D32's: the remedy is deliberate intervention — an
experiment, a geo holdout, a seesaw — not waiting for passive variation to accumulate. Dew et al.'s
own sweep over observational spend variance moved nothing (Table 2, p. 19); this one moves the
decision only at a level of jitter nobody would introduce on purpose, which is an argument for their
experimental route rather than an alternative to it.

Of the three, only the bottom row is something a team can adopt on Monday, and it works by bounding
the consequence of an error rather than by removing it.

---

# The kill criterion fired, so the degradation grid was never run

K1 states that if C0 fails any of G1–G5 the harness is presumed wrong and C1–C7 must not be
reported. C0 fails all five. Everything that could make that an artefact was ruled out first — a
13-agent adversarial audit raised nine candidate implementation defects and all nine were refuted;
the control-block error was found and fixed rather than argued away; the one configuration that
appeared to pass turned out to be a minimum over ten noisy draws quoted as though it were a
configuration, and no configuration passes when re-run properly (D23).

**So this is not the study that was designed.** It is a pre-registered stress test that never reached
its stress conditions, because under ten years of clean weekly data the specified model could not
identify how much of sales the media caused. C1 through C7 — collinearity, short samples, confounded
demand, misspecification, placebo channels — are **moot rather than negative**. Nothing in this
repository reports them, and no one should infer them. K1 says "no exceptions, no partial
publication", and running them under an exploratory label would be exactly the exception it forbids.

The cost of that is concrete and worth naming: **RQ3 is unanswered.** Whether MMM assigns credit to
a channel with exactly zero true effect, and routes real budget into it, was the finding a marketing
team could most directly have acted on. G6 was never evaluated on anything.

# Scoring the pre-registered predictions

§8 recorded directional predictions so the study could be wrong in public. It was. Every prediction
is listed, including the seven that can no longer be scored, because a table showing only the scored
row would understate what the kill criterion cost.

| Condition | Prediction (§8, fixed in advance) | Outcome |
|---|---|---|
| **C0** | **Passes all gates** | **WRONG. Fails all five: 0.540 / 0.417 / 0.650 / 2.355× / 0.200.** |
| C1 (ρ=0.95) | G1 and G3 fail; total-media bias under 10%; G4 may still pass | Not run — K1. Moot, not negative. |
| C2 (T=104) | G2 fails first; G1 marginal; G4 marginal | Not run — K1. Moot, not negative. |
| C3 (φ=0.6) | Every channel biased upward; total media overstated by >30%; G1 fails | Not run — K1. Moot, not negative. |
| C4 | Rung 1 badly wrong while G1 and G4 pass | Not run — K1. Moot, not negative. |
| C5 | Passes G6 comfortably (flag rate < 10%) | Not run — K1. Moot, not negative. |
| C6 | **Fails G6** — flag rate > 30%, placebo spend share > 5% | Not run — K1. Moot, not negative. |
| C7 | Fails G1–G4; regret above 50%; G5 in genuine doubt | Not run — K1. Moot, not negative. |

The C0 prediction is the one that matters and it was wrong on the merits, not by a hair. Worse than
that, it was avoidable: §1 already cited Jin et al. (2017), whose result on Hill identifiability over
a finite range is the mechanism documented above. The prediction was made without connecting a
citation the study had itself listed. That is an error of literature review, recorded as D25, and it
is the single most useful thing in this repository — a pre-registered prediction, scored wrong, by
the person who made it.

§8 also recorded which prediction its author was least confident in: C4 passing G4, "the
diagnostic/decision divergence". That was the most interesting thing the grid would have tested and
it is among what was forfeited.

---

# What this study does and does not claim

**No novelty is claimed for the decision metrics.** An earlier draft claimed the beats-status-quo
rate as a contribution. A systematic prior-art sweep refuted that and it is withdrawn (D31). The
statistic is imported: probability of improvement across replications is Agarwal et al.'s (2021)
reporting convention, and bounding the probability that a learned policy is worse than its incumbent
is the defining quantity of safe policy improvement (Thomas et al. 2015, 2019). The finding's *shape*
is older still — DeMiguel, Garlappi and Uppal (2009) showed an optimiser losing to a naive incumbent
once estimation error swamps the optimisation gain, and Smith and Winkler (2006) named the mechanism
the optimizer's curse. In marketing, Haus's incrementality-test Monte Carlo already reports the
analogous rate: acting on noisy readouts left the business worse off than doing nothing 38% of the
time, across 36 scenarios run a million times each with true channel performance held fixed. **This
study's 20% is worse than Haus's worst arm, which beat the baseline 62% of the time.**

Haus differs in a way that is the point rather than an excuse: **they fit no model.** Measurement
there is truth plus stipulated noise — no adstock, no saturation, no confounded demand, no design
matrix. Their quantity is the cost of *noise* in an unbiased readout, which more data removes. This
study's is the cost of *non-identification* in a correctly specified estimator, which more data does
not remove.

What is left, stated plainly and without the word "novel":

1. **Within-family.** The failure occurs with the functional form exactly correct and the baseline
   recovered to 1.4 × 10⁻¹³ — not as conflation between two model classes. Dew et al. state
   within-family non-identification as established background (citing Jin et al. 2017) and act on it
   by fixing adstock rather than estimating it; they name the correctly-specified case as open
   (p. 34). This measures its decision consequences.
2. **Production tooling.** Google's shipping Meridian on default priors, not a bespoke framework.
3. **Pre-registration**, with every deviation dated in a log that now runs to 39 entries.
4. **Ground truth by intervention.** True contribution is `sales(spend) − sales(spend_c := 0)`
   evaluated on the generating process, never a comparison of β̂ to β.

Pathak, Jeunen and Lambert (2026) independently name this design as their own open problem: "An
important next step is a semi-synthetic benchmark with known response structure and oracle regret."

# Limitations

Read these as binding, not as ritual.

- **One DGP family.** A different generating process may well be identifiable. Nothing here licenses
  a claim about MMM in general beyond this family.
- **`spend_log_sd = 0.30` was an assumption**, not a specification. The exploratory sweep shows how
  much it matters — enough to move every gate, nowhere near enough to pass one.
- **The pre-registered action space is not an implementable one, and this materially moves the
  headline.** §3 lets the optimiser zero a channel, and the true optimum does exactly that to OOH
  (true ROAS 0.84 against 1.73–2.32 elsewhere, 50-seed medians). Under a governed ±30% guardrail the
  worse-than-nothing rate falls from 80% to 24%. The gates still fail, but any single number quoted
  from this study must name which action space produced it.
- **Meridian was run national, single-geo, with no control columns, at default priors.** Its baseline
  spline is more flexible than the generating process's, and — more importantly — the geo hierarchy
  that is the package's main identifying mechanism is switched off, because this DGP produces one
  national series. That is a supported configuration but not the one Meridian is designed around.
  Read the anchor as "a Bayesian MMM with ROI priors agrees on G1", not as "Meridian as deployed
  agrees".
- **Experiment-calibrated priors are untested.** This study's own recommendation — run an experiment
  — enters an MMM through a tightened ROI prior. Meridian supports exactly that and it was not tried,
  so the recommendation is argued rather than demonstrated.
- **One DGP family, and now one spend *shape* beyond it.** Flighting was tested and did not rescue
  the decision. Other plan shapes — seasonal pulsing, always-on with occasional blackouts,
  geo-staggered launches — were not.
- **The flighted arm's achievable lift is half the baseline's** (0.54% against 1.16%), so its G4 is
  not comparable to C0's and is not compared. Only G1, G2, G3 and G5 carry across the two arms.
- **The exploratory results lead this document's framing but are not pre-registered.** Flighting and
  the guardrail had their readings pre-committed in the deviations log before the numbers existed,
  which is why leading with one of them is defensible; it is not equivalent to pre-registration, and
  the confirmatory result is the one the thresholds were written for.
- **Robyn is untested.** The Python port is an LLM-translated beta, so a failure there would be
  uninterpretable and no claim is made about it.
- **Global optimality is empirically supported, not proven** (D17). The surface is non-concave; the
  returned optimum matches a 256-start reference on 100 of 100 trials across five representative
  cells. That is a strong prior, not a guarantee, and every regret figure inherits its status.
- **Hyperparameter search ranges were assumed**, not specified in the pre-registration.
- **G2's failure is estimator-specific**, not structural. Meridian passes it.
- **The decision gates are sensitive to an optimiser convention that was not recorded at the time**
  (D28). D23's harness was never committed; a faithful re-implementation reproduces G1–G3 to the
  printed digit and lands G4 at 2.386× versus 2.355× depending on the SLSQP start seed. The
  convention is now fixed and asserted, but the sensitivity — 1.3% relative on median regret — is the
  honest error bar on that class of number.
- **Eleven cells across the three exploratory harnesses fail to solve** — 2 of 800 in the sweep,
  7 of 1000 in the bound check, 2 of 400 in the flighting check — and all are excluded, with the
  adverse bound reported for the sweep (D30). Only the sweep records a `solve_failed` marker; in the
  other two the exclusions are inferable only from missing seed rows, which is a defect in those
  harnesses rather than in the numbers, and is set out in full above.
- **The prior-art sweep has a known hole.** The pass over pre-1990 marketing science — Little,
  Lodish, Hanssens, and the normative decision-model literature — did not complete. If the rate form
  turns out to be fifty years old, that is where it will be found.
- **Xu, Thomadsen and Zhang (2025)** argue the ordinary bootstrap is invalid for post-optimisation
  quantities. G5's interval here is a Wilson score interval on a proportion over independent seeds,
  which sidesteps that objection, but G2's block bootstrap does not and the paper is unread.

---

# Reproducing

No network calls, no API keys, no data downloads. Every stochastic step takes an explicit seed.

```bash
uv sync
uv run pytest                              # 434 tests: 433 pass, 1 strict xfail
uv run python -m mmm_recovery.sweep        # the exploratory sweep, ~1 min
uv run python -m mmm_recovery.robustness   # bound check + the ±30% guardrail, ~2 min
uv run python -m mmm_recovery.flighting    # the flighted-spend validity check, ~2 min
uv run python -m mmm_recovery.plateau      # the identification plateau (D39), ~3 min
```

The dashboard is one self-contained HTML file with `plotly.js` vendored inline, so it opens from a
clean checkout with nothing to fetch. Building it twice produces the same bytes, and the suite
asserts both that and that the committed copy still matches the code:

```bash
uv sync --extra report
make report                                # results/dashboard.html, ~5 MiB, ~10 s
```

The sweep refuses to write results unless its `spend_log_sd = 0.30` column reproduces D23's published
table to within 5 × 10⁻⁴. That assertion is the reason the other three columns can be believed.

The Meridian anchor needs the optional extra and takes about 31 minutes — 186 s per seed on CPU,
ten seeds:

```bash
uv sync --extra meridian
uv run python -m mmm_recovery.meridian_anchor          # --condition C0 --seeds 10
```

That command had no entry point until D38: the module imported and exited silently, so the
reproduction step this section documents could not be tested. Its output is
`results/meridian_c0.json`, which the suite now checks is byte-reproducible by the writer that
produces it. The `seconds` field is wall-clock and is the one value that does not reproduce; nothing
is computed from it.

# Reading order

1. [`PREREGISTRATION.md`](PREREGISTRATION.md) — the binding specification, and the deviations log
   D1–D39 that records every departure from it with a date and a reason.
2. [`docs/WHEN-TO-TRUST-YOUR-MMM.md`](docs/WHEN-TO-TRUST-YOUR-MMM.md) — one page, no equations, for a
   marketing reader.
3. [`results/dashboard.html`](results/dashboard.html) — the plateau and the three-arm comparison
   as figures. Self-contained; open it directly.
4. [`results/`](results/) — the raw per-seed output behind the C0 table, the exploratory sweep, the
   bound check, the flighting check and, since D39, the plateau grid. The **Nelder–Mead diagnostic**
   is the one thing here still without a committed harness: it survives only as numbers in the
   deviations log and in the strict-xfail reason string that `pytest -rx` prints. Given what
   regenerating the plateau turned up — a published centrepiece describing a control block the study
   had abandoned — that remaining gap should be read as a live risk rather than a formality.

# References

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. (2021). *Deep
  Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS. arXiv:2108.13264.
- Chan, D., & Perry, M. (2017). *Challenges and Opportunities in Media Mix Modeling.* Google Inc.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). *Optimal Versus Naive Diversification: How
  Inefficient Is the 1/N Portfolio Strategy?* Review of Financial Studies, 22(5), 1915–1953.
- Dew, R., Padilla, N., & Shchetkina, A. (2024). *Your MMM is Broken: Identification of Nonlinear and
  Time-varying Effects in Marketing Mix Models.* arXiv:2408.07678.
- Haus. *Fast, Confident, and Wrong: The Risk of Noisy Incrementality Tests.*
- Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). *Bayesian Methods for Media Mix
  Modeling with Carryover and Shape Effects.* Google Inc.
- Pathak, N., Jeunen, O., & Lambert, E. (2026). *Auditing Marketing Budget Allocation with Hindsight
  Regret.* arXiv:2604.25977.
- Smith, J. E., & Winkler, R. L. (2006). *The Optimizer's Curse: Skepticism and Postdecision Surprise
  in Decision Analysis.* Management Science, 52(3), 311–322.
- Thomas, P. S., Theocharous, G., & Ghavamzadeh, M. (2015). *High Confidence Policy Improvement.*
  ICML.
- Thomas, P. S., Castro da Silva, B., Barto, A. G., Giguere, S., Brun, Y., & Brunskill, E. (2019).
  *Preventing undesirable behavior of intelligent machines.* Science, 366(6468), 999–1004.
- Xu, S., Thomadsen, R., & Zhang, D. (2025). *The Winner's Curse in Data-Driven Decision-Making:
  Evidence and Solutions.* SSRN 5930537. **Cited from its abstract; the full text was not obtainable
  and the claim below is unverified.**
- Zhang, S., & Vaver, J. (2017). *Introduction to the Aggregate Marketing System Simulator.* Google
  Inc.
