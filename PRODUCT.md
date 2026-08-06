# Product

<!-- impeccable:product-schema 1 -->

Scope: `results/dashboard.html`, and **only** as produced by its generator
`src/mmm_recovery/report.py`. The HTML is a build artefact. It is committed so it can be opened
from a clean checkout, and it is never edited.

## Platform

web

## Stack

Python 3.12 generator (`src/mmm_recovery/report.py`) emitting one static HTML file. `plotly` is
vendored into the output at build time and is never a run-time dependency of the study. No
framework, no bundler, no dev server. `make report` is the whole build.

## Users

- **Hiring managers in marketing measurement.** Skimming to judge whether the work is rigorous.
  They will read the headline, the cards, and one figure.
- **Practitioners deciding whether to trust an MMM.** They want to know whether this applies to
  their model. They will read the gate table and the plateau panel.

Neither has read `README.md` and neither will. Both are reading **once**, **probably on a phone**,
**probably from a link** someone sent them. Nothing may depend on prior context, hovering, or
scrolling back up.

## Product Purpose

**Surface mode: READ.** The page's job is to build understanding. Not Persuade, not Operate.

There is nothing to sign up for, nothing to configure, no next step. A reader arrives, forms an
accurate picture of what was measured and what it means, and leaves. Success is that they could
restate the finding correctly to someone else. Failure is that they remember it was impressive.

The understanding to build, in order:

1. Acting on this model's budget advice was worse than doing nothing in 160 of 200 simulated
   worlds; under a ±30% planning guardrail, 48 of 200.
2. The reason is identification, not a bug: 639 of 780 candidate transforms fit the data as well
   as the truth does, spanning a 16.4× range of implied contribution.
3. Improving the estimate did not improve the decision. That is the point the page exists for.

## Positioning

A neighbouring dashboard cannot truthfully copy three things:

- **Ground truth by intervention.** True contribution is `sales(spend) − sales(spend_c := 0)`
  evaluated on the generating process, never `β̂` compared to `β`. Almost nothing in this category
  has a known answer to be wrong against.
- **Pre-registration with a kill criterion that actually fired.** Thresholds were fixed before any
  code ran; K1 fired on the clean condition and the degradation grid was never run.
- **It reports its own failure.** The page's headline is that the method under test did not work,
  published by the person who built the test.

## Operating Context

Opened from a shared link, cold, once. Most often on a phone. The reader has no prior context, no
onboarding, and no intention to return. Adjacent artefacts they may or may not reach:
`README.md` (the full method), `PREREGISTRATION.md` (binding spec plus the D1–D42 deviations log),
and `docs/WHEN-TO-TRUST-YOUR-MMM.md` (a one-page plain-language explainer). The dashboard must
stand alone without any of them.

## Capabilities and Constraints

**Hard constraints, asserted by `tests/test_report.py`.** Any proposal that breaks one is refused,
not adapted.

| # | Constraint | Asserted by |
|---|---|---|
| a | **Byte-determinism.** `make report` produces identical bytes on repeat builds. No clock, no uuid, no dict-ordering dependence, no environment dependence. | `test_the_output_is_byte_deterministic` |
| b | **Zero external fetches.** No script, stylesheet, font, image or XHR from any host. **No webfonts** — system stack, or embed it. A stated reproducibility property of the study, not a preference. | `test_the_page_fetches_nothing` |
| c | **No new runtime dependencies** in the core package. Anything required goes in the `[report]` optional extra or nowhere. | `pyproject.toml`; the grid runs without `[report]` |

Two further properties, cheap to break and expensive to notice:

- **The vendored bundle must stay in `<head>`.** Figure scripts call `Plotly.newPlot` at parse
  time; a deferred or end-of-body bundle renders every panel blank with no malformed HTML and no
  other symptom. `test_plotly_is_defined_before_the_first_figure_script`.
- **Every figure `<div>` id is fixed.** Plotly seeds ids from a uuid otherwise, destroying (a).

**Settled — do not re-litigate:**

- **Palette** is the data-viz reference palette, both modes rendered and CSS-selected, because a
  plotly figure bakes colours into JSON and cannot re-theme itself without JavaScript.
- **Two layouts per figure** (wide, narrow) because a plotly subplot grid cannot reflow with CSS.
  This is why there are eight figure divs and not four.
- **Marks are SVG, not WebGL.** 780 points does not need `Scattergl`, and requiring it would make
  the figure blank on a machine without it and unprintable everywhere.
- **Text is full-width.** No `max-width` measure cap on text containers.

## Brand Commitments

**Voice: clinical, understated, no hype.** Short declaratives. Numbers carry their own weight and
are never introduced with an adjective. The subject is a study that failed its own pre-registered
validity gate and reports that honestly; the design has to be worthy of that, not dress it up. If a
treatment would look equally at home on a page announcing a success, it is wrong here. Restraint is
part of the argument: a study about honest measurement must not arrive looking like a launch.

Concretely — every number states its action space; a verdict never appears without its threshold; a
figure never implies precision the study does not have; "fail" is stated plainly and not softened.

**Anti-references.** Not stylistic preferences. Any of these is a defect on this surface:

gradients of any kind · glassmorphism, frosted panels, blur · decorative motion, scroll animation,
parallax, count-up numbers · hero eyebrow chips ("PRE-REGISTERED STUDY" pills above the title) ·
italic serif display type · nested cards · icon tile stacks, feature-grid iconography · anything
that reads as a product launch: badges, testimonial framing, CTA buttons, "trusted by".

## Evidence on Hand

Real, in the repository, and the source of every number on the page:

- `results/spend_variation_sweep.csv`, `results/flighting_check.csv`,
  `results/optimiser_bound_check.csv`, `results/plateau_sweep.csv` — every figure and card is
  recomputed from these at build time.
- `results/meridian_c0.json` — Google Meridian's independent run on the same condition.
- `PREREGISTRATION.md` — the binding spec and the D1–D42 deviations log.

**Absences that must never be fabricated.** There is no degradation grid: conditions C1–C7 were
never run because K1 fired, and they are moot rather than negative. There is no `experiment.py`,
`conditions.py` or `metrics.py`. The guardrail arm has **no contribution-bias column at all**,
because it re-solves the same fitted surfaces and never refits — its G1 is the baseline's number,
not a fourth measurement. The Nelder–Mead diagnostic still has no committed harness and is
unverified.

## Product Principles

1. **A number without its warrant is not a result.** Every figure carries its action space, every
   verdict its threshold, every extremum its N and spread.
2. **The artefact is the claim.** A figure with no committed harness behind it is treated as
   unverified regardless of how confidently it is recorded.
3. **Show the contrast, do not assert it.** Where the finding is a comparison, both terms are on
   the page as marks — not one term drawn and the other described in a caption.
4. **Render it and look at it.** Every defect this surface has actually shipped passed every
   structural check. A browser is the only instrument that has caught them.
5. **Restraint is evidence.** The design's credibility comes from what it declines to do.

## Accessibility & Inclusion

WCAG 2.1 AA, **measured rather than assumed** — 0 of 64 text nodes below AA in both themes, checked
in-browser after every palette change.

- `--muted` is the **text** token; `deemphasis` is the **mark** token. They are separate because
  WCAG asks 4.5:1 of small text and only 3:1 of a graphical object, and collapsing them once made
  the receding point cloud compete with the series colour it exists to sit behind.
- Each plot is `aria-hidden` with a static `sr-only` description standing in for it. A plotly
  figure is thousands of unlabelled SVG nodes; exposing it produces noise, not meaning. Those
  descriptions restate **no measured quantity** — a test enforces it — because a build-time string
  derived from the CSVs is both a drift surface and a determinism surface.
- `<main>` landmark, `scope` on all ten table headers, a table caption, and a
  `prefers-reduced-motion` block for plotly's hover transitions, which are the only motion present.

## Detector findings reviewed and declined — do not re-raise

| Finding | Date | Why declined |
|---|---|---|
| `overused-font` — "Open Sans" | 2026-08-06 | The match is at line 29 of the output, inside the **vendored plotly bundle**; the page's own CSS starts around line 3900. `report.py` names no such face, and the page measurably *renders* in `system-ui, -apple-system, "Segoe UI"` — body, headings, tiles and plotly's own SVG text. |
| `layout-transition` — `transition: height` ×2 | 2026-08-06 | Both matches are inside the vendored bundle. `report.py` contains no `transition` at all. |
| `em-dash-overuse` — "15 in body text" | 2026-08-06 | The rendered page has **9** in 3,318 characters, not 15; the static scan double-counts because every figure is emitted twice for light and dark. Six of the nine are title–subtitle separators in chart headings, which is standard typography and carries none of the prose cadence the rule screens for. Fixing because a rule fired, rather than because the thing it detects is present, is the wrong instinct. |

The first two share a cause worth stating once: **this page inlines a 5.6 MB third-party bundle, so
any whole-file scanner will attribute the library's source to the design.** Check the line number
against the `<style>` block before believing a finding.

## The failure mode this surface has actually had

Four times, a structurally perfect page was wrong in a way only a browser showed:

1. The vendored bundle sat after the calls that use it. Valid HTML, correct figure JSON,
   byte-deterministic, four silent `Plotly is not defined` errors, two blank panels.
2. Contributions were divided by 1000 under an axis titled £k, so the axis read 50 beside an
   annotation reading £51,989k.
3. A log-axis annotation was given linear units and sat four decades from its own line; and the
   noiseless panel's own subject — the truth, the entire reason the panel exists — was clipped off
   the bottom of its axis while every assertion passed.
4. Mixing full and short row labels gave a categorical axis eight categories instead of four; the
   explicit range clipped to the first four and the bottom two arms silently drew nothing.

**Therefore: render it and look at it.** Any change to the generator is verified by screenshot in
both themes at desktop and phone widths before it is called done. A detector that runs against the
rendered page is doing the job that structural checks demonstrably cannot.
