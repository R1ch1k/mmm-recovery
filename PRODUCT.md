# PRODUCT.md — the dashboard surface

Scope: `results/dashboard.html`, and **only** as produced by its generator
`src/mmm_recovery/report.py`. The HTML is a build artefact. It is committed so it can be opened
from a clean checkout, and it is never edited.

---

## Surface mode: READ

The page's job is to **build understanding**. It is not Persuade and it is not Operate.

There is nothing to sign up for, nothing to configure, no next step. A reader arrives, forms an
accurate picture of what was measured and what it means, and leaves. Success is that they could
restate the finding correctly to someone else. Failure is that they remember it was impressive.

The specific understanding to build, in order:

1. Acting on this model's budget advice was worse than doing nothing in 160 of 200 simulated
   worlds; under a ±30% planning guardrail, 48 of 200.
2. The reason is identification, not a bug: 639 of 780 candidate transforms fit the data as well
   as the truth does, spanning a 16.4× range of implied contribution.
3. Improving the estimate did not improve the decision. That is the point the page exists for.

## Users

- **Hiring managers in marketing measurement.** Skimming to judge whether the work is rigorous.
  They will read the headline, the cards, and one figure.
- **Practitioners deciding whether to trust an MMM.** They want to know if this applies to their
  model. They will read the gate table and the plateau panel.

Neither has read `README.md` and neither will. Both are reading **once**, **probably on a phone**,
**probably from a link** someone sent them. Nothing may depend on prior context, hovering, or
scrolling back up.

## Voice

Clinical, understated, no hype. Short declaratives. Numbers carry their own weight and are never
introduced with an adjective.

The subject is a study that **failed its own pre-registered validity gate and reports that
honestly**. The design has to be worthy of that, not dress it up. If a treatment would look equally
at home on a page announcing a success, it is wrong for this page. Restraint is part of the
argument: a study about honest measurement must not arrive looking like a launch.

Concretely — every number states its action space; a verdict never appears without its threshold;
a figure never implies precision the study does not have; "fail" is stated plainly and not softened.

## Anti-references

Not stylistic preferences. Any of these is a defect on this surface:

- gradients of any kind
- glassmorphism, frosted panels, blur
- decorative motion, scroll animation, parallax, count-up numbers
- hero eyebrow chips ("PRE-REGISTERED STUDY" pills above the title)
- italic serif display type
- nested cards — a card inside a card
- icon tile stacks, feature-grid iconography
- anything that reads as a product launch: badges, testimonial framing, CTA buttons, "trusted by"

## Hard constraints — asserted by `tests/test_report.py`, not negotiable

Any proposal that breaks one of these is refused, not adapted.

| # | Constraint | Asserted by |
|---|---|---|
| a | **Byte-determinism.** `make report` produces identical bytes on repeat builds. No clock, no uuid, no dict-ordering dependence, no environment dependence. | `test_the_output_is_byte_deterministic` |
| b | **Zero external fetches.** No script, stylesheet, font, image or XHR from any host. **No webfonts** — system stack, or embed it. This is a stated reproducibility property of the study, not a preference. | `test_the_page_fetches_nothing` |
| c | **No new runtime dependencies** in the core package. Anything required goes in the `[report]` optional extra or nowhere. `plotly` is vendored into the output at build time and is never a run-time dependency of the grid. | `pyproject.toml`; core grid runs without `[report]` |

Two further properties that are cheap to break and expensive to notice:

- **The vendored bundle must stay in `<head>`.** Figure scripts call `Plotly.newPlot` at parse
  time; a deferred or end-of-body bundle renders every panel blank with no malformed HTML and no
  other symptom. `test_plotly_is_defined_before_the_first_figure_script`.
- **Every figure `<div>` id is fixed.** Plotly seeds ids from a uuid otherwise, which destroys (a).

## What is already settled — do not re-litigate

- **Palette** is the data-viz reference palette, both modes rendered and CSS-selected, because a
  plotly figure bakes colours into JSON and cannot re-theme itself without JavaScript.
- **Two layouts per figure** (wide, narrow) exist because a plotly subplot grid cannot reflow with
  CSS. This is why there are eight figure divs and not four.
- **Marks are SVG, not WebGL.** 780 points does not need `Scattergl`, and requiring it would make
  the figure blank on a machine without it and unprintable everywhere.
- **Text is full-width.** No `max-width` measure cap on text containers.

## Detector findings reviewed and declined — do not re-raise

| Finding | Date | Why declined |
|---|---|---|
| `overused-font` — "Open Sans" | 2026-08-06 | The match is at line 29 of the output, inside the **vendored plotly bundle**; the page's own CSS starts around line 3900. `report.py` names no such face, and the page measurably *renders* in `system-ui, -apple-system, "Segoe UI"` — body, headings, tiles and plotly's own SVG text. Unactionable without forking plotly, and there is nothing to action. |
| `layout-transition` — `transition: height` ×2 | 2026-08-06 | Both matches are inside the vendored bundle. `report.py` contains no `transition` at all. |
| `em-dash-overuse` — "15 in body text" | 2026-08-06 | The rendered page has **9** em-dashes in 3,318 characters, not 15; the static scan double-counts because every figure is emitted twice for light and dark. Six of the nine are title–subtitle separators in chart headings, which is standard typography and carries none of the prose cadence the rule screens for. Fixing because a rule fired, rather than because the thing it detects is present, is the wrong instinct. |

The first two share a cause worth stating once: **this page inlines a 5.6 MB third-party bundle, so
any whole-file scanner will attribute the library's source to the design.** Check the line number
against the `<style>` block before believing a finding.

## The failure mode this surface has actually had

Twice, a structurally perfect page was wrong in a way only a browser showed:

1. The vendored bundle sat after the calls that use it. Valid HTML, correct figure JSON,
   byte-deterministic, four silent `Plotly is not defined` errors, two blank panels.
2. Contributions were divided by 1000 under an axis titled £k, so the axis read 50 beside an
   annotation reading £51,989k.
3. A log-axis annotation was given linear units and sat four decades from its own line; and a
   panel's own subject was clipped off the bottom of its axis while every assertion passed.

**Therefore: render it and look at it.** Any change to the generator is verified by screenshot in
both themes at desktop and phone widths before it is called done. A detector that runs against the
rendered page is doing the job that structural checks demonstrably cannot.
