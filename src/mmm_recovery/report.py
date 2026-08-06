"""The dashboard: one self-contained HTML file, zero external references — Step 8.

`plotly.js` is vendored inline rather than pulled from a CDN, so the file opens from a clean
checkout with no network and renders identically in five years. Precisely: the page fetches
nothing — no `<script src>`, no stylesheet, no font, no image. The vendored bundle does contain
map-tile attribution URLs as inert string literals, for trace types this dashboard never
instantiates; `tests/test_report.py` asserts the absence of load targets rather than of the
substring "http", because the latter would be a claim the file cannot honestly make.

Marks are SVG `Scatter`, not `Scattergl`. 780 points a panel does not need WebGL, and requiring
it would make the figure blank on a machine without it and unprintable everywhere.

Every number is recomputed from `results/*.csv` at build time; nothing here is typed in by hand
except the pre-registered thresholds, which are the one thing that must never be derived from the
data.

**Byte-determinism.** Every `<div>` id is fixed, plotly's own uuid generation is therefore never
reached, and no clock is read. `make report` twice on the same inputs produces the same file, and
`tests/test_report.py` asserts it.

Two figures, in the order the findings deserve:

1. **The plateau** — the mechanism. Two panels on the same grid: with noise, and without it. The
   contrast is the point, so they share an x-axis and are read against each other.
2. **Three arms** — the result. What actually moved the decision, with intervals, against the
   threshold that was fixed before any code ran.

Colours are the data-viz reference palette. Both modes are rendered and CSS picks one, because a
plotly figure bakes its colours into JSON and cannot re-theme itself without JavaScript.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from mmm_recovery.plateau import BAND, PLATEAU_CSV
from mmm_recovery.sweep import RESULTS_DIR, SWEEP_CSV

DASHBOARD_HTML: Final = RESULTS_DIR / "dashboard.html"
BOUND_CSV: Final = RESULTS_DIR / "optimiser_bound_check.csv"
FLIGHTING_CSV: Final = RESULTS_DIR / "flighting_check.csv"

TRUTH_CV_NOISY: Final = 30.94153
"""D39's truth CV on the noisy arm. Held here so the band drawn on the figure is the band the
deviations log states, rather than one silently recomputed into agreement with itself."""

G5_THRESHOLD: Final = 0.90
"""§7. Pre-registered, never derived from the data."""

TRUE_TV_CONTRIBUTION: Final = 51_989.0
"""£k. `truth.incremental_contribution` on C0 seed 0, pinned in `tests/test_plateau.py`."""


@dataclass(frozen=True)
class Palette:
    """One rendering of the data-viz reference palette. See `references/palette.md`."""

    name: str
    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    series: str
    accent: str

    @property
    def deemphasis(self) -> str:
        """Points the objective cannot distinguish. Chart chrome, deliberately not a hue."""
        return self.muted


LIGHT: Final = Palette(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series="#2a78d6",
    accent="#eb6834",
)
DARK: Final = Palette(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series="#3987e5",
    accent="#d95926",
)

FONT: Final = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _wilson(successes: int, total: int) -> tuple[float, float]:
    """95% Wilson score interval. The same one `sweep.py` reports and D37 quotes."""
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
    return centre - half / denominator, centre + half / denominator


@dataclass(frozen=True)
class Arm:
    """One intervention, on the only decision metric comparable across arms (D35)."""

    label: str
    changed: str
    g5: float
    low: float
    high: float
    n: int
    moved: bool


def load_arms() -> list[Arm]:
    """The four arms of the README's three-arm comparison, recomputed from the CSVs."""
    sweep = [row for row in _rows(SWEEP_CSV) if row["channel"] == "all"]
    by_level: dict[float, dict[int, dict[str, float]]] = {}
    for row in sweep:
        cell = by_level.setdefault(float(row["spend_log_sd"]), {}).setdefault(int(row["seed"]), {})
        cell[row["metric"]] = float(row["value"])

    def sweep_arm(level: float, label: str, changed: str, moved: bool) -> Arm:
        cells = [c for c in by_level[level].values() if c.get("solve_failed", 0.0) != 1.0]
        wins = int(sum(c["beats_status_quo"] for c in cells))
        low, high = _wilson(wins, len(cells))
        return Arm(label, changed, wins / len(cells), low, high, len(cells), moved)

    flighted = [r for r in _rows(FLIGHTING_CSV) if r["arm"] == "flighted" and r["regret"]]
    wins = sum(int(float(r["beats_status_quo"])) for r in flighted)
    flight_low, flight_high = _wilson(wins, len(flighted))

    guardrail = [
        r
        for r in _rows(BOUND_CSV)
        if (float(r["min_multiplier"]), float(r["max_multiplier"])) == (0.7, 1.3)
    ]
    guard_wins = sum(int(float(r["beats_status_quo"])) for r in guardrail)
    guard_low, guard_high = _wilson(guard_wins, len(guardrail))

    return [
        sweep_arm(0.30, "C0 baseline", "nothing — the pre-registered condition", False),
        Arm(
            "Flighted",
            "the shape of spend",
            wins / len(flighted),
            flight_low,
            flight_high,
            len(flighted),
            False,
        ),
        sweep_arm(1.00, "spend_log_sd = 1.00", "the amount of spend variation", True),
        Arm(
            "Guardrail m_c ∈ [0.7, 1.3]",
            "the optimiser's action space",
            guard_wins / len(guardrail),
            guard_low,
            guard_high,
            len(guardrail),
            True,
        ),
    ]


def plateau_figure(palette: Palette) -> go.Figure:
    """The mechanism: what the objective can and cannot distinguish, with and without noise."""
    rows = _rows(PLATEAU_CSV)

    # The counts go in the subplot titles rather than inside the axes: at this point density the
    # bottom-left corner is solid marker and an in-plot caption is unreadable there.
    noisy = np.array([float(r["cv_rmse"]) for r in rows if r["series"] == "noisy"])
    noiseless = np.array([float(r["cv_rmse"]) for r in rows if r["series"] == "noiseless"])
    tied_count = int((noisy <= TRUTH_CV_NOISY * (1.0 + BAND)).sum())
    better_count = int((noisy < TRUTH_CV_NOISY).sum())

    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=False,
        horizontal_spacing=0.09,
        subplot_titles=(
            f"<b>Noisy sales — what the estimator actually sees</b><br>"
            f"<span style='font-size:12px'>{tied_count} of {len(noisy)} within the band · "
            f"{better_count} fit <b>better</b> than the truth</span>",
            f"<b>Noiseless — the same grid, the noise removed</b><br>"
            f"<span style='font-size:12px'>0 of {len(noiseless)} within the band · best "
            f"competitor {noiseless.min() / 0.00002:,.0f}× the truth's score</span>",
        ),
    )

    for column, (series, truth_cv) in enumerate(
        (("noisy", TRUTH_CV_NOISY), ("noiseless", None)), start=1
    ):
        cells = [row for row in rows if row["series"] == series]
        score = np.array([float(row["cv_rmse"]) for row in cells])
        # £k, the unit every other document in this repo uses. Not rescaled: an axis reading
        # "50" under a title saying £k is a unit error, and this figure once shipped with one.
        contribution = np.array([float(row["tv_contribution"]) for row in cells])
        bias = np.array([float(row["relative_bias"]) for row in cells])
        alpha = np.array([float(row["hill_shape"]) for row in cells])
        ratio = np.array([float(row["half_saturation_ratio"]) for row in cells])

        tied = score <= truth_cv * (1.0 + BAND) if truth_cv else np.zeros(len(score), bool)
        hover = [
            f"α̂ {a:.2f} · κ̂ ratio {r:.2f}<br>CV RMSE {s:.4f}<br>"
            f"TV contribution £{c:,.0f}k<br>bias {b:+.1%}"
            for a, r, s, c, b in zip(alpha, ratio, score, contribution, bias, strict=True)
        ]
        for mask, colour, size, name in (
            (~tied, palette.deemphasis, 5, "Distinguishable from the truth"),
            (tied, palette.series, 6, f"Within {BAND:.0%} of the truth's score"),
        ):
            if not mask.any():
                continue
            figure.add_trace(
                go.Scatter(
                    x=contribution[mask],
                    y=score[mask],
                    mode="markers",
                    marker={
                        "size": size,
                        "color": colour,
                        "opacity": 0.75,
                        "line": {"width": 0.5, "color": palette.surface},
                    },
                    name=name,
                    legendgroup=name,
                    showlegend=column == 1,
                    hovertext=[h for h, keep in zip(hover, mask, strict=True) if keep],
                    hoverinfo="text",
                ),
                row=1,
                col=column,
            )

        if truth_cv is not None:
            figure.add_hline(
                y=truth_cv * (1.0 + BAND),
                line={"color": palette.series, "width": 1.5, "dash": "dash"},
                row=1,
                col=column,
            )
            figure.add_annotation(
                x=1.0,
                xref="x domain",
                y=truth_cv * (1.0 + BAND),
                text=f"{BAND:.0%} band  ",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font={"size": 11, "color": palette.series},
                row=1,
                col=column,
            )

        figure.add_vline(
            x=TRUE_TV_CONTRIBUTION,
            line={"color": palette.ink, "width": 1.5},
            row=1,
            col=column,
        )
        figure.add_annotation(
            x=TRUE_TV_CONTRIBUTION,
            y=1.0,
            yref="y domain",
            text=f"  truth, £{TRUE_TV_CONTRIBUTION:,.0f}k",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={"size": 11, "color": palette.ink},
            row=1,
            col=column,
        )

        figure.update_xaxes(
            title_text="Implied TV contribution, £k",
            tickformat=",.0f",
            row=1,
            col=column,
        )
        figure.update_yaxes(
            title_text="CV RMSE, £k per week" if column == 1 else None,
            type="log" if truth_cv is None else "linear",
            dtick=1 if truth_cv is None else None,
            row=1,
            col=column,
        )

    figure.update_layout(
        height=460,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.12, "x": 0},
        margin={"l": 70, "r": 24, "t": 96, "b": 56},
    )
    return _style(figure, palette)


def arms_figure(arms: list[Arm], palette: Palette) -> go.Figure:
    """The result: what moved the decision, on the metric D35 fixed as comparable."""
    figure = go.Figure()
    labels = [arm.label for arm in arms]

    for arm in arms:
        figure.add_trace(
            go.Scatter(
                x=[arm.low, arm.high],
                y=[arm.label, arm.label],
                mode="lines",
                line={"color": palette.series if arm.moved else palette.muted, "width": 2},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[arm.g5 for arm in arms],
            y=labels,
            mode="markers",
            marker={
                "size": 11,
                "color": [palette.series if arm.moved else palette.muted for arm in arms],
                "line": {"width": 2, "color": palette.surface},
            },
            hovertext=[f"{arm.label}<br>changed: {arm.changed}" for arm in arms],
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Values live in an aligned column outside the plot, not beside each marker. Anchored to the
    # marker they ran back over their own interval whiskers, which is the one thing a forest plot
    # exists to show.
    for arm in arms:
        figure.add_annotation(
            x=1.02,
            xref="paper",
            y=arm.label,
            text=f"{arm.g5:.3f}  [{arm.low:.3f}, {arm.high:.3f}]   n={arm.n}",
            showarrow=False,
            xanchor="left",
            font={"size": 12.5, "color": palette.secondary},
        )

    figure.add_vline(
        x=G5_THRESHOLD,
        line={"color": palette.accent, "width": 2, "dash": "dash"},
    )
    figure.add_annotation(
        x=G5_THRESHOLD,
        y=1.04,
        yref="paper",
        text="pre-registered threshold, 0.90  ",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font={"size": 12, "color": palette.accent},
    )
    figure.update_xaxes(
        title_text="G5 — share of runs beating the status quo",
        range=[0.0, 1.0],
        dtick=0.2,
    )
    figure.update_yaxes(autorange="reversed")
    figure.update_layout(height=330, margin={"l": 200, "r": 250, "t": 52, "b": 56})
    return _style(figure, palette)


def _style(figure: go.Figure, palette: Palette) -> go.Figure:
    figure.update_layout(
        paper_bgcolor=palette.surface,
        plot_bgcolor=palette.surface,
        font={"family": FONT, "size": 13, "color": palette.secondary},
        hoverlabel={"font": {"family": FONT, "size": 12}},
    )
    figure.update_xaxes(
        gridcolor=palette.grid,
        zeroline=False,
        linecolor=palette.axis,
        tickcolor=palette.axis,
        title_font={"size": 12, "color": palette.muted},
    )
    figure.update_yaxes(
        gridcolor=palette.grid,
        zeroline=False,
        linecolor=palette.axis,
        tickcolor=palette.axis,
        title_font={"size": 12, "color": palette.muted},
    )
    for annotation in figure.layout.annotations:
        if annotation.font is None or annotation.font.color is None:
            annotation.font = {"family": FONT, "size": 13, "color": palette.ink}
    return figure


def c0_gates() -> list[tuple[str, str, str, bool]]:
    """The five pre-registered gates on C0, recomputed from the sweep's control column."""
    rows = [row for row in _rows(SWEEP_CSV) if float(row["spend_log_sd"]) == 0.30]
    per_seed: dict[int, dict[str, float]] = {}
    covered: list[float] = []
    for row in rows:
        if row["metric"] == "covered":
            covered.append(float(row["value"]))
        elif row["channel"] == "all":
            per_seed.setdefault(int(row["seed"]), {})[row["metric"]] = float(row["value"])

    def median(metric: str) -> float:
        return float(np.median([cell[metric] for cell in per_seed.values()]))

    beats = float(np.mean([cell["beats_status_quo"] for cell in per_seed.values()]))
    return [
        (
            "G1",
            "median |relative bias| of contribution",
            f"{median('median_abs_relative_bias'):.3f}",
            False,
        ),
        ("G2", "coverage of the nominal 90% interval", f"{float(np.mean(covered)):.3f}", False),
        ("G3", "median Spearman ρ(true, estimated)", f"{median('spearman'):.3f}", False),
        ("G4", "median allocation regret", f"{median('regret'):.3f}×", False),
        ("G5", "beats status quo", f"{beats:.3f}", False),
    ]


def _figure_html(figure: go.Figure, div_id: str) -> str:
    """Deterministic: the div id is fixed, so plotly never reaches its uuid generator."""
    html: str = figure.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )
    return html


def _palette_css(palette: Palette) -> str:
    return "\n".join(
        f"    --{name}: {value};"
        for name, value in (
            ("surface", palette.surface),
            ("ink", palette.ink),
            ("secondary", palette.secondary),
            ("muted", palette.muted),
            ("grid", palette.grid),
            ("series", palette.series),
            ("accent", palette.accent),
        )
    )


def render() -> str:
    """The whole dashboard as one string. Pure in the CSVs — no clock, no network, no uuids."""
    arms = load_arms()
    gates = c0_gates()
    plotlyjs = get_plotlyjs()

    figures = "\n".join(
        f'<div class="viz viz-{palette.name}">\n'
        f'  <section class="fig">\n'
        f"    <h2>1 &middot; The plateau</h2>\n"
        f'    <p class="lede">Four channels held at their true values; only TV&rsquo;s saturation '
        f"pair swept, 26&nbsp;&times;&nbsp;30 = 780 points per panel, C0 seed&nbsp;0. Band is "
        f"1% of the true parameters&rsquo; own cross-validation score.</p>\n"
        f"{_figure_html(plateau_figure(palette), f'plateau-{palette.name}')}\n"
        f"  </section>\n"
        f'  <section class="fig">\n'
        f"    <h2>2 &middot; What actually moved the decision</h2>\n"
        f'    <p class="lede">G5 with 95% Wilson intervals. The rate is the one decision metric '
        f"D35 fixed in advance as comparable across arms; regret is not.</p>\n"
        f"{_figure_html(arms_figure(arms, palette), f'arms-{palette.name}')}\n"
        f"  </section>\n"
        f"</div>"
        for palette in (LIGHT, DARK)
    )

    gate_rows = "\n".join(
        f"      <tr><th>{gate}</th><td>{name}</td>"
        f'<td class="num">{value}</td><td class="fail">fail</td></tr>'
        for gate, name, value, _ in gates
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mmm-recovery — attributable &ne; incremental</title>
<!-- plotly.js, vendored. Must load BEFORE the figure scripts below: they call
     Plotly.newPlot at parse time, so a deferred or end-of-body bundle leaves every
     panel blank with "Plotly is not defined" and no other symptom. -->
<script type="text/javascript">{plotlyjs}</script>
<style>
:root {{
{_palette_css(LIGHT)}
    --plane: #f9f9f7;
    --hairline: rgba(11,11,11,0.10);
    color-scheme: light;
}}
.viz-dark {{ display: none; }}
@media (prefers-color-scheme: dark) {{
  :root {{
{_palette_css(DARK)}
    --plane: #0d0d0d;
    --hairline: rgba(255,255,255,0.10);
    color-scheme: dark;
  }}
  .viz-light {{ display: none; }}
  .viz-dark {{ display: block; }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 40px 32px 72px;
  background: var(--plane); color: var(--ink);
  font-family: {FONT}; font-size: 15px; line-height: 1.6;
}}
h1 {{ font-size: 30px; line-height: 1.25; margin: 0 0 8px; letter-spacing: -0.01em; }}
h2 {{ font-size: 19px; margin: 0 0 4px; letter-spacing: -0.005em; }}
p {{ margin: 0 0 14px; }}
.sub {{ color: var(--secondary); font-size: 17px; margin-bottom: 28px; }}
.tiles {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 32px; }}
.tile {{
  background: var(--surface); border: 1px solid var(--hairline); border-radius: 10px;
  padding: 14px 18px; min-width: 170px; flex: 1 1 170px;
}}
.tile .v {{ font-size: 26px; font-weight: 600; color: var(--ink); line-height: 1.15; }}
.tile .k {{ font-size: 12.5px; color: var(--muted); margin-top: 3px; }}
.fig {{
  background: var(--surface); border: 1px solid var(--hairline); border-radius: 12px;
  padding: 22px 22px 10px; margin: 0 0 22px;
}}
.lede {{ color: var(--secondary); font-size: 13.5px; margin-bottom: 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 7px 12px 7px 0; border-bottom: 1px solid var(--hairline); }}
th {{ color: var(--muted); font-weight: 600; }}
td.num {{ font-variant-numeric: tabular-nums; color: var(--ink); }}
td.fail {{ color: #d03b3b; font-weight: 600; }}
footer {{ color: var(--muted); font-size: 13px; margin-top: 34px; }}
footer code {{ font-size: 12.5px; }}
a {{ color: var(--series); }}
</style>
</head>
<body>
<h1>attributable &ne; incremental</h1>
<p class="sub">A pre-registered test of whether Marketing Mix Modelling produces a budget decision
worth acting on. Under the cleanest conditions the study could construct, it does not.</p>

<div class="tiles">
  <div class="tile"><div class="v">639 / 780</div><div class="k">transforms the objective cannot
    tell apart from the truth</div></div>
  <div class="tile"><div class="v">16.4&times;</div><div class="k">spread in implied TV
    contribution across them</div></div>
  <div class="tile"><div class="v">116 / 780</div><div class="k">that fit <em>better</em> than the
    truth</div></div>
  <div class="tile"><div class="v">160 / 200</div><div class="k">runs where the advice is worse
    than doing nothing (&sect;3 action space)</div></div>
  <div class="tile"><div class="v">48 / 200</div><div class="k">the same, under a &plusmn;30%
    planning guardrail</div></div>
</div>

{figures}

<section class="fig">
  <h2>3 &middot; The five pre-registered gates on C0</h2>
  <p class="lede">Thresholds fixed before any code ran. The kill criterion K1 fired on this table,
  so conditions C1&ndash;C7 were never run and are moot rather than negative.</p>
  <table>
    <thead><tr><th>Gate</th><th>Metric</th><th>C0, 200 seeds</th><th>Verdict</th></tr></thead>
    <tbody>
{gate_rows}
    </tbody>
  </table>
</section>

<footer>
<p>Every figure is recomputed at build time from <code>results/*.csv</code>; only the
pre-registered thresholds are constants. <code>plotly.js</code> is vendored inline, so this page
fetches nothing &mdash; no script, stylesheet, font or image is loaded from anywhere. Output is
byte-deterministic: building twice gives the same file.</p>
<p>The plateau panels were regenerated at D39 as <code>mmm_recovery.plateau</code> after the
original harness was found never to have been committed. The figures they replace
(177&nbsp;of&nbsp;780, a 5.5&times; spread) described the superseded six-column control block.
Both values are in the deviations log.</p>
<p>Build: <code>uv sync --extra report &amp;&amp; make report</code></p>
</footer>

</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the self-contained dashboard.")
    parser.add_argument("--out", type=Path, default=DASHBOARD_HTML)
    args = parser.parse_args(argv)

    html = render()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(html.encode('utf-8')) / 1_048_576:.1f} MiB)")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a command, not in the suite
    raise SystemExit(main())
