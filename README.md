# mmm-recovery

**Can you trust Marketing Mix Modeling enough to move your budget on it? A pre-registered test says no, not on observational data alone. Google's own Meridian reproduces the failure.**

Marketing Mix Modeling (MMM) is how teams decide where ad spend should go. This project asks a narrower question: are its budget recommendations good enough to actually act on? And it asks that under conditions far kinder than reality.

## The setup

I gave the model every advantage. The correct model form. Ten years of clean weekly data. Five channels, no confounding, no collinearity. Then, against thresholds I fixed before running any code, I checked two things: whether it recovers how much each channel really contributed, and whether the budget it recommends beats leaving spend where it is.

## The finding: it fails every gate

| Gate (fixed in advance) | Threshold | Result | |
|---|---|---|---|
| Contribution error (median relative bias) | < 0.20 | 0.540 | fail |
| Interval coverage | ≥ 0.80 | 0.417 | fail |
| Channel ranking (Spearman) | ≥ 0.80 | 0.650 | fail |
| Allocation regret | < 0.20x | 2.355x | fail |
| Beats doing nothing | ≥ 0.90 | 0.200 | fail |

The recommended budget beats leaving spend alone in only **20% of simulated worlds**, against a 90% bar. Then I ran Google's Meridian on the same data, independently, on its default settings, and it reproduced the estimation failure (median bias 0.456, also far outside the 0.20 gate).

## Why: the signal it needs gets erased by ordinary noise

The model can only tell how much a channel contributed by reading the curvature in how sales respond to spend. Ordinary week-to-week noise flattens that curve. To show it, I held four channels at their true values and swept one channel's response parameters over a grid, once on clean data and once with realistic noise added:

| | Clean data | Realistic noise added |
|---|---|---|
| How much better the true answer fits than its best rival | 2,590x better | **rivals fit better than the truth** |
| Grid points fitting better than the truth | 0 of 780 | **116 of 780** |
| Spread in implied contribution across the near-tied set | none (truth is unique) | **16x** |

Read the two columns against each other. On clean data the truth is recovered cleanly, so the method is not broken in principle. Add noise of an entirely ordinary size and 116 different "worlds" fit the data *better* than the truth does, and what one channel supposedly contributed swings by a factor of 16. The signal the method depends on is simply gone.

## The sharp version: better estimates did not buy a better decision

I ran three controlled interventions, changing one thing each time, and measured each on the rate at which the model's budget beats doing nothing:

| Intervention | What it changed | Contribution error | Beats doing nothing |
|---|---|---|---|
| Baseline | nothing | 0.540 | 0.200 |
| Flighted spend | the *shape* of spend | **0.309** (best in the study) | 0.207 |
| More spend variation | the *amount* of variation | 0.412 | **0.462** |
| Planning guardrail | the optimiser's *action space* | 0.540 (unchanged) | **0.760** |

The arm that most improved the *estimates* (flighting) moved the *decision* by essentially zero, 0.200 to 0.207. The arm that most improved the decision (a planning guardrail) changed the estimates not at all. You can learn which channels rank better without learning how much better, and better ranking did not buy a better budget. Estimable is not the same as actionable.

## What it means in practice

More history will not fix this, because more history is just more of the same passive variation. You need a deliberate signal: an experiment such as a geo lift test, or a flighting pattern that switches channels off and on so the response curve becomes visible. This is exactly why tools like Meridian let you calibrate to lift tests. This study measures what happens when you do not.

## Honesty notes

The identification failure itself is well established (Dew et al. 2024; Jin et al. 2017), and I claim no novelty for it. When I ran a prior-art sweep, four things I had thought were original turned out to be occupied, and I withdrew those claims rather than soften them. What is left is narrower and specific: the failure carried all the way through to a budget decision, on a pre-registered grid with thresholds fixed in advance, and reproduced by production tooling rather than a bespoke framework. Full method, deviations log, and limitations are below and in `docs/`.

*Fourth in a series on measurement validity, after priced-in, pitchvalue, and marketplace-mispricing: a signal you can detect or estimate is not the same as one you can act on.*
