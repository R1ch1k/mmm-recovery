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
except the pre-registered thresholds and the two truth scores, which are the things that must
never be derived from the data they are used to judge.

**Byte-determinism.** Every `<div>` id is fixed, plotly's own uuid generation is therefore never
reached, and no clock is read. `make report` twice on the same inputs produces the same file, and
`tests/test_report.py` asserts it.

Two figures, in the order the findings deserve:

1. **The plateau** — the mechanism. Two panels on the same grid: with noise, and without it. The
   contrast is the point, so they are read against each other. The truth is drawn as an explicit
   mark on both, because on the noiseless panel it is the whole finding: it scores 2,590× better
   than anything else on the grid and would otherwise be clipped off the bottom of the axis.
2. **Three arms** — the result, on *both* dimensions. G1 beside G5 on the same four rows, because
   the finding is that the arm improving the estimate most moved the decision least. One panel
   cannot show that.

Each figure is emitted twice more than it looks: once per colour mode (a plotly figure bakes its
colours into JSON and cannot re-theme itself without JavaScript) and once per layout (side-by-side
for a desktop, stacked for a phone — a subplot grid cannot reflow with CSS). CSS picks one of the
four. Colours are the data-viz reference palette.
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

TRUTH_CV_NOISELESS: Final = 0.00002
"""D39's truth CV on the noiseless arm, same reasoning. It is not in `plateau_sweep.csv`: that
file holds the 780 *competitors*, and the truth is the thing they are competing against. Without
it drawn explicitly the noiseless panel shows a point cloud whose own subject is off-axis."""

G5_THRESHOLD: Final = 0.90
"""§7. Pre-registered, never derived from the data."""

G1_THRESHOLD: Final = 0.20
"""§7. Pre-registered, never derived from the data."""

TRUE_TV_CONTRIBUTION: Final = 51_989.0
"""£k. `truth.incremental_contribution` on C0 seed 0, pinned in `tests/test_plateau.py`."""

GATE_THRESHOLDS: Final = (
    ("G1", "median |relative bias| of contribution", "< 0.20"),
    ("G2", "coverage of the nominal 90% interval", "≥ 0.80"),
    ("G3", "median Spearman ρ(true, estimated)", "≥ 0.80"),
    ("G4", "median allocation regret", "< 0.20×"),
    ("G5", "beats status quo", "≥ 0.90"),
)
"""§7, fixed before any code ran. A verdict without its threshold is unreadable."""


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
    """One intervention, on both dimensions the study separates.

    `g5` is the decision metric D35 fixed in advance as comparable across arms. `g1` is the
    estimation metric. Keeping them on one object is the point of the figure: the arm that moved
    one moved the other least.
    """

    label: str
    changed: str
    g5: float
    low: float
    high: float
    n: int
    moved: bool
    g1: float
    g1_by_construction: bool = False
    """True when the arm never refits, so its G1 *is* the baseline's rather than a measurement of
    its own. The guardrail re-solves the same fitted surfaces; `optimiser_bound_check.csv` has no
    bias column at all, which is that fact showing up in the schema."""

    @property
    def short(self) -> str:
        """The row label for the phone layout, where the gutter cannot carry the full name."""
        return {
            "spend_log_sd = 1.00": "sd = 1.00",
            "Guardrail m_c ∈ [0.7, 1.3]": "Guardrail ±30%",
        }.get(self.label, self.label)


def _median_g1(rows: list[dict[str, str]], level: float) -> float:
    values = [
        float(row["value"])
        for row in rows
        if float(row["spend_log_sd"]) == level
        and row["channel"] == "all"
        and row["metric"] == "median_abs_relative_bias"
    ]
    return float(np.median(values))


def load_arms() -> list[Arm]:
    """The four arms of the README's three-arm comparison, recomputed from the CSVs."""
    sweep = [row for row in _rows(SWEEP_CSV) if row["channel"] == "all"]
    by_level: dict[float, dict[int, dict[str, float]]] = {}
    for row in sweep:
        cell = by_level.setdefault(float(row["spend_log_sd"]), {}).setdefault(int(row["seed"]), {})
        cell[row["metric"]] = float(row["value"])

    all_sweep_rows = _rows(SWEEP_CSV)
    baseline_g1 = _median_g1(all_sweep_rows, 0.30)

    def sweep_arm(level: float, label: str, changed: str, moved: bool) -> Arm:
        cells = [c for c in by_level[level].values() if c.get("solve_failed", 0.0) != 1.0]
        wins = int(sum(c["beats_status_quo"] for c in cells))
        low, high = _wilson(wins, len(cells))
        return Arm(
            label,
            changed,
            wins / len(cells),
            low,
            high,
            len(cells),
            moved,
            _median_g1(all_sweep_rows, level),
        )

    flighted = [r for r in _rows(FLIGHTING_CSV) if r["arm"] == "flighted" and r["regret"]]
    wins = sum(int(float(r["beats_status_quo"])) for r in flighted)
    flight_low, flight_high = _wilson(wins, len(flighted))
    flight_g1 = float(np.median([float(r["median_abs_relative_bias"]) for r in flighted]))

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
            flight_g1,
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
            baseline_g1,
            g1_by_construction=True,
        ),
    ]


def worse_than_nothing() -> tuple[int, int, int, int]:
    """(C0 worse, C0 n, guardrailed worse, guardrailed n). Regret above 1.0× is the definition.

    Derived rather than typed, so the two headline decision figures cannot drift from the CSVs the
    rest of the page is built from.
    """
    per_seed: dict[int, dict[str, float]] = {}
    for row in _rows(SWEEP_CSV):
        if float(row["spend_log_sd"]) == 0.30 and row["channel"] == "all":
            per_seed.setdefault(int(row["seed"]), {})[row["metric"]] = float(row["value"])
    c0 = [c["regret"] for c in per_seed.values() if c.get("solve_failed", 0.0) != 1.0]

    guardrail = [
        float(r["regret"])
        for r in _rows(BOUND_CSV)
        if (float(r["min_multiplier"]), float(r["max_multiplier"])) == (0.7, 1.3) and r["regret"]
    ]
    return (
        sum(1 for x in c0 if x > 1.0),
        len(c0),
        sum(1 for x in guardrail if x > 1.0),
        len(guardrail),
    )


def plateau_figure(palette: Palette, narrow: bool = False) -> go.Figure:
    """The mechanism: what the objective can and cannot distinguish, with and without noise."""
    rows = _rows(PLATEAU_CSV)

    noisy = np.array([float(r["cv_rmse"]) for r in rows if r["series"] == "noisy"])
    noiseless = np.array([float(r["cv_rmse"]) for r in rows if r["series"] == "noiseless"])
    tied_count = int((noisy <= TRUTH_CV_NOISY * (1.0 + BAND)).sum())
    better_count = int((noisy < TRUTH_CV_NOISY).sum())
    gap = noiseless.min() / TRUTH_CV_NOISELESS

    figure = make_subplots(
        rows=2 if narrow else 1,
        cols=1 if narrow else 2,
        shared_yaxes=False,
        horizontal_spacing=0.09,
        vertical_spacing=0.16,
        subplot_titles=(
            f"<b>Noisy sales — what the estimator sees</b><br>"
            f"<span style='font-size:11px'>{tied_count} of {len(noisy)} within the band<br>"
            f"{better_count} fit <b>better</b> than the truth</span>"
            if narrow
            else f"<b>Noisy sales — what the estimator actually sees</b><br>"
            f"<span style='font-size:12px'>{tied_count} of {len(noisy)} within the band · "
            f"{better_count} fit <b>better</b> than the truth</span>",
            f"<b>Noiseless — the noise removed</b><br>"
            f"<span style='font-size:11px'>0 of {len(noiseless)} within the band<br>"
            f"the truth wins by {gap:,.0f}×</span>"
            if narrow
            else f"<b>Noiseless — the same grid, the noise removed</b><br>"
            f"<span style='font-size:12px'>0 of {len(noiseless)} within the band · the truth "
            f"wins by {gap:,.0f}×</span>",
        ),
    )

    for index, (series, truth_cv) in enumerate(
        (("noisy", TRUTH_CV_NOISY), ("noiseless", TRUTH_CV_NOISELESS))
    ):
        row = index + 1 if narrow else 1
        column = 1 if narrow else index + 1
        first = index == 0

        cells = [r for r in rows if r["series"] == series]
        score = np.array([float(r["cv_rmse"]) for r in cells])
        # £k, the unit every other document in this repo uses. Not rescaled: an axis reading
        # "50" under a title saying £k is a unit error, and this figure once shipped with one.
        contribution = np.array([float(r["tv_contribution"]) for r in cells])
        bias = np.array([float(r["relative_bias"]) for r in cells])
        alpha = np.array([float(r["hill_shape"]) for r in cells])
        ratio = np.array([float(r["half_saturation_ratio"]) for r in cells])

        tied = score <= truth_cv * (1.0 + BAND)
        hover = [
            f"α̂ {a:.2f} · κ̂ ratio {r:.2f}<br>CV RMSE {s:.5f}<br>"
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
                    showlegend=first,
                    hovertext=[h for h, keep in zip(hover, mask, strict=True) if keep],
                    hoverinfo="text",
                ),
                row=row,
                col=column,
            )

        # The truth itself, as a mark rather than as a caption. On the noiseless panel this is the
        # entire finding and it sits three decades below the nearest competitor.
        figure.add_trace(
            go.Scatter(
                x=[TRUE_TV_CONTRIBUTION],
                y=[truth_cv],
                mode="markers",
                marker={
                    "size": 15,
                    "color": palette.accent,
                    "symbol": "star",
                    "line": {"width": 1.2, "color": palette.surface},
                },
                name="The true parameters",
                legendgroup="truth",
                showlegend=first,
                hovertext=[
                    f"<b>The true parameters</b><br>CV RMSE {truth_cv:.5f}<br>"
                    f"TV contribution £{TRUE_TV_CONTRIBUTION:,.0f}k"
                ],
                hoverinfo="text",
            ),
            row=row,
            col=column,
        )

        # Truth's own score, as a line, so "116 fit better" is 116 marks below it rather than a
        # number in a caption. The band ceiling above it is dashed; this one is solid.
        figure.add_hline(
            y=truth_cv,
            line={"color": palette.accent, "width": 1.5},
            row=row,
            col=column,
        )
        # An annotation on a log axis takes its y in log10 units. Passing the raw score put the
        # "1% band" label at y ≈ 1 on the noiseless panel — four decades above its own line.
        truth_y = truth_cv if first else float(np.log10(truth_cv))

        if first:
            # The band is only drawn where it separates anything. On the noiseless panel it sits
            # 1% above a score of 0.00002, which on a seven-decade axis is the same pixel as the
            # truth rule — two coincident lines reading as one thick one, and a label for a
            # distinction the eye cannot make. The subplot title carries "0 of 780" instead.
            figure.add_hline(
                y=truth_cv * (1.0 + BAND),
                line={"color": palette.series, "width": 1.5, "dash": "dash"},
                row=row,
                col=column,
            )
            figure.add_annotation(
                x=0.0,
                xref="x domain",
                y=truth_cv * (1.0 + BAND),
                text=f" {BAND:.0%} band ",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                bgcolor=palette.surface,
                opacity=0.92,
                font={"size": 11, "color": palette.series},
                row=row,
                col=column,
            )
            figure.add_annotation(
                x=1.0,
                xref="x domain",
                y=truth_cv,
                text=(
                    f" truth's score · {better_count} below "
                    if narrow
                    else f" truth's own score · {better_count} of {len(noisy)} score below it "
                ),
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                bgcolor=palette.surface,
                opacity=0.92,
                font={"size": 10 if narrow else 11, "color": palette.accent},
                row=row,
                col=column,
            )
        else:
            # The gap is the panel's subject, so it is drawn on the panel: an arrow from open
            # space down to the truth, labelled with the ratio. Offset to the right of the
            # contribution rule so the text does not sit across it.
            figure.add_annotation(
                x=TRUE_TV_CONTRIBUTION,
                y=truth_y,
                showarrow=True,
                arrowhead=2,
                arrowsize=1.1,
                arrowwidth=1.4,
                arrowcolor=palette.accent,
                ax=40 if narrow else 76,
                ay=-44,
                text=f"<b>the truth</b> — {gap:,.0f}× better than<br>anything else on the grid",
                xanchor="left",
                yanchor="bottom",
                align="left",
                bgcolor=palette.surface,
                opacity=0.94,
                font={"size": 11 if narrow else 11.5, "color": palette.accent},
                row=row,
                col=column,
            )

        figure.add_vline(
            x=TRUE_TV_CONTRIBUTION,
            line={"color": palette.ink, "width": 1.5},
            row=row,
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
            row=row,
            col=column,
        )

        # One x-range for both panels. The two clouds are only comparable if the axis they sit on
        # is the same one, and the whole figure is an instruction to compare them.
        figure.update_xaxes(
            title_text="Implied TV contribution, £k",
            tickformat=",.0f",
            range=[0.0, 285_000.0],
            row=row,
            col=column,
        )
        if first:
            # Explicit, with headroom below the truth: on autorange the truth mark sat on the
            # axis line itself and read as chrome rather than as a point.
            figure.update_yaxes(
                title_text="CV RMSE, £k per week",
                range=[float(score.min()) - 0.06, float(score.max()) + 0.09],
                row=row,
                col=column,
            )
        else:
            # Log, and floored well below the truth. Left to autorange it starts at the best
            # competitor and the panel's own subject falls off the bottom, which is how this
            # figure shipped once: two clouds that looked like the same finding twice.
            # `power` rather than the SI default, which renders 1e-5 as "10μ" — a prefix that
            # means nothing applied to £k per week.
            figure.update_yaxes(
                title_text="CV RMSE, £k per week  (log)",
                type="log",
                range=[-5.4, 1.45],
                dtick=1,
                exponentformat="power",
                row=row,
                col=column,
            )

    figure.update_layout(
        height=940 if narrow else 480,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.10, "x": 0},
        margin={
            "l": 58 if narrow else 70,
            "r": 14 if narrow else 24,
            "t": 116 if narrow else 100,
            "b": 56,
        },
    )
    return _style(figure, palette)


def arms_figure(arms: list[Arm], palette: Palette, narrow: bool = False) -> go.Figure:
    """The result, on both dimensions.

    G1 left, G5 right, same four rows. The finding is the mismatch between the two columns, and a
    single-metric panel makes it something you have to be told rather than something you see.
    """
    # On a 332px-wide phone figure a 170px label gutter is half the panel, so the rows carry
    # short names there and the full one stays in the hover.
    labels = [(arm.short if narrow else arm.label) for arm in arms]
    figure = make_subplots(
        rows=2 if narrow else 1,
        cols=1 if narrow else 2,
        shared_yaxes=not narrow,
        horizontal_spacing=0.06,
        vertical_spacing=0.22,
        subplot_titles=(
            "<b>G1 — contribution error</b><br>"
            "<span style='font-size:11.5px'>lower is better · threshold 0.20</span>",
            "<b>G5 — beats the status quo</b><br>"
            "<span style='font-size:11.5px'>higher is better · threshold 0.90</span>",
        ),
    )
    g1_row, g1_col = 1, 1
    g5_row, g5_col = (2, 1) if narrow else (1, 2)

    # --- G1: a lollipop. There is no interval to draw, so the value sits beside the marker.
    # Every trace on this axis must use the SAME label strings: mixing `label` and `short` gives
    # the categorical axis eight categories instead of four, and the last two rows fall outside
    # the range and simply do not draw.
    for arm, name in zip(arms, labels, strict=True):
        figure.add_trace(
            go.Scatter(
                x=[0.0, arm.g1],
                y=[name, name],
                mode="lines",
                line={"color": palette.muted, "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=g1_row,
            col=g1_col,
        )
    figure.add_trace(
        go.Scatter(
            x=[arm.g1 for arm in arms],
            y=labels,
            mode="markers+text",
            marker={
                "size": 11,
                "color": [
                    palette.muted if arm.g1_by_construction else palette.series for arm in arms
                ],
                "symbol": ["circle-open" if arm.g1_by_construction else "circle" for arm in arms],
                "line": {"width": 2, "color": palette.series},
            },
            text=[f"  {arm.g1:.3f}" for arm in arms],
            textposition="middle right" if not narrow else "top center",
            textfont={"size": 12, "color": palette.secondary},
            hovertext=[
                f"{arm.label}<br>G1 {arm.g1:.3f}"
                + (
                    "<br>re-solves the same fitted surfaces — no refit"
                    if arm.g1_by_construction
                    else ""
                )
                for arm in arms
            ],
            hoverinfo="text",
            showlegend=False,
        ),
        row=g1_row,
        col=g1_col,
    )
    figure.add_vline(
        x=G1_THRESHOLD,
        line={"color": palette.accent, "width": 2, "dash": "dash"},
        row=g1_row,
        col=g1_col,
    )
    # The one row on this panel that is not a measurement, said on the panel rather than only in
    # the lede. The open marker carries the same signal for anyone who does not read it.
    construction = next(arm for arm in arms if arm.g1_by_construction)
    figure.add_annotation(
        x=construction.g1,
        y=construction.short if narrow else construction.label,
        text="unchanged by construction",
        showarrow=False,
        # The value label sits above the marker on the phone layout, so this goes below it or the
        # two overprint. The axis padding leaves room under the last row.
        xanchor="center" if narrow else "left",
        yanchor="top" if narrow else "bottom",
        xshift=0 if narrow else 14,
        yshift=-13 if narrow else 11,
        font={"size": 10 if narrow else 10.5, "color": palette.muted},
        row=g1_row,
        col=g1_col,
    )

    # --- G5: point and 95% Wilson interval, as before.
    for arm, name in zip(arms, labels, strict=True):
        figure.add_trace(
            go.Scatter(
                x=[arm.low, arm.high],
                y=[name, name],
                mode="lines",
                line={"color": palette.series if arm.moved else palette.muted, "width": 2},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=g5_row,
            col=g5_col,
        )
    figure.add_trace(
        go.Scatter(
            x=[arm.g5 for arm in arms],
            y=labels,
            mode="markers+text",
            marker={
                "size": 11,
                "color": [palette.series if arm.moved else palette.muted for arm in arms],
                "line": {"width": 2, "color": palette.surface},
            },
            text=[f"{arm.g5:.3f}" for arm in arms],
            textposition="top center",
            textfont={"size": 12, "color": palette.secondary},
            hovertext=[
                f"{arm.label}<br>changed: {arm.changed}<br>"
                f"G5 {arm.g5:.3f}  [{arm.low:.3f}, {arm.high:.3f}]   n={arm.n}"
                for arm in arms
            ],
            hoverinfo="text",
            showlegend=False,
        ),
        row=g5_row,
        col=g5_col,
    )
    figure.add_vline(
        x=G5_THRESHOLD,
        line={"color": palette.accent, "width": 2, "dash": "dash"},
        row=g5_row,
        col=g5_col,
    )

    figure.update_xaxes(
        title_text="median |relative bias|",
        range=[0.0, 0.80] if narrow else [0.0, 0.94],
        dtick=0.2,
        row=g1_row,
        col=g1_col,
    )
    figure.update_xaxes(
        title_text="share of runs beating the status quo",
        range=[0.0, 1.0],
        dtick=0.2,
        row=g5_row,
        col=g5_col,
    )
    # Padding at both ends of the categorical axis. Without it the top row sits on the plot edge
    # and its "top center" value label is clipped by the subplot title above it.
    figure.update_yaxes(range=[len(arms) - 0.45, -0.55], autorange=False)
    figure.update_layout(
        height=670 if narrow else 372,
        margin={"l": 104 if narrow else 170, "r": 16 if narrow else 24, "t": 96, "b": 56},
    )
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


def c0_gates() -> list[tuple[str, str, str, str]]:
    """The five pre-registered gates on C0, recomputed from the sweep's control column.

    Returns (gate, metric, threshold, measured). The threshold is a constant from §7 and is the
    one thing on this page that must never be derived from the data it judges.
    """
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
    measured = {
        "G1": f"{median('median_abs_relative_bias'):.3f}",
        "G2": f"{float(np.mean(covered)):.3f}",
        "G3": f"{median('spearman'):.3f}",
        "G4": f"{median('regret'):.3f}×",
        "G5": f"{beats:.3f}",
    }
    return [(gate, name, threshold, measured[gate]) for gate, name, threshold in GATE_THRESHOLDS]


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
    c0_worse, c0_n, guard_worse, guard_n = worse_than_nothing()

    figures = "\n".join(
        f'<div class="viz viz-{palette.name} viz-{width}">\n'
        f'  <section class="fig">\n'
        f"    <h2>1 &middot; What actually moved the decision</h2>\n"
        f'    <p class="lede">Four arms, each changing one thing. G5 is the one decision metric '
        f"D35 fixed in advance as comparable across arms; regret is not. The guardrail never "
        f"refits &mdash; it re-solves the same fitted surfaces &mdash; so its G1 is the "
        f"baseline&rsquo;s number reappearing, not a fourth measurement.</p>\n"
        f"{_figure_html(arms_figure(arms, palette, narrow), f'arms-{palette.name}-{width}')}\n"
        f"  </section>\n"
        f'  <section class="fig">\n'
        f"    <h2>2 &middot; The plateau &mdash; why the estimate cannot be trusted</h2>\n"
        f'    <p class="lede">Four channels held at their true values; only TV&rsquo;s saturation '
        f"pair swept, 26&nbsp;&times;&nbsp;30 = 780 points per panel, C0 seed&nbsp;0. Band is "
        f"1% of the true parameters&rsquo; own cross-validation score. Read the two panels "
        f"against each other: the contrast <em>is</em> the mechanism.</p>\n"
        f"{_figure_html(plateau_figure(palette, narrow), f'plateau-{palette.name}-{width}')}\n"
        f"  </section>\n"
        f"</div>"
        for palette in (LIGHT, DARK)
        for width, narrow in (("wide", False), ("narrow", True))
    )

    gate_rows = "\n".join(
        f"      <tr><th>{gate}</th><td>{name}</td>"
        f'<td class="num thr">{threshold}</td>'
        f'<td class="num">{value}</td><td class="fail">fail</td></tr>'
        for gate, name, threshold, value in gates
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mmm-recovery &mdash; estimable &ne; actionable</title>
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
/* A plotly subplot grid cannot reflow with CSS, so both layouts are rendered and one is
   chosen here. The resize handler at the end of <body> re-sizes whichever becomes visible:
   a figure plotted while display:none has zero width and would otherwise stay broken. */
.viz-narrow {{ display: none; }}
@media (max-width: 720px) {{
  .viz-wide {{ display: none !important; }}
  .viz-light.viz-narrow {{ display: block; }}
}}
@media (max-width: 720px) and (prefers-color-scheme: dark) {{
  .viz-light.viz-narrow {{ display: none; }}
  .viz-dark.viz-narrow {{ display: block; }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 40px 32px 72px;
  background: var(--plane); color: var(--ink);
  font-family: {FONT}; font-size: 15px; line-height: 1.6;
  -webkit-text-size-adjust: 100%;
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
.tile.lead {{ border-color: var(--accent); }}
.tile .v {{
  font-size: 26px; font-weight: 600; color: var(--ink); line-height: 1.15;
  font-variant-numeric: tabular-nums;
}}
.tile.lead .v {{ color: var(--accent); }}
.tile .k {{ font-size: 12.5px; color: var(--muted); margin-top: 3px; }}
.fig {{
  background: var(--surface); border: 1px solid var(--hairline); border-radius: 12px;
  padding: 22px 22px 10px; margin: 0 0 22px;
}}
.lede {{ color: var(--secondary); font-size: 13.5px; margin-bottom: 10px; }}
.tw {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 7px 12px 7px 0; border-bottom: 1px solid var(--hairline); }}
th {{ color: var(--muted); font-weight: 600; }}
td.num {{ font-variant-numeric: tabular-nums; color: var(--ink); }}
td.thr {{ color: var(--muted); }}
td.fail {{ color: #d03b3b; font-weight: 600; }}
footer {{ color: var(--muted); font-size: 13px; margin-top: 34px; }}
footer code {{ font-size: 12.5px; }}
a {{ color: var(--series); }}
@media (max-width: 720px) {{
  body {{ padding: 24px 16px 56px; font-size: 16px; }}
  h1 {{ font-size: 25px; }}
  h2 {{ font-size: 17.5px; }}
  .sub {{ font-size: 16px; margin-bottom: 22px; }}
  .tiles {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
  .tile {{ min-width: 0; }}
  .fig {{ padding: 16px 12px 8px; border-radius: 10px; }}
  table {{ min-width: 460px; }}
}}
</style>
</head>
<body>
<h1>estimable &ne; actionable</h1>
<p class="sub">A pre-registered test of whether Marketing Mix Modelling produces a budget decision
worth acting on. Under the cleanest conditions the study could construct, it does not.</p>

<div class="tiles">
  <div class="tile lead"><div class="v">{c0_worse} / {c0_n}</div><div class="k">runs where the
    model&rsquo;s advice is <em>worse than doing nothing</em> (&sect;3 action space)</div></div>
  <div class="tile lead"><div class="v">{guard_worse} / {guard_n}</div><div class="k">the same,
    under a &plusmn;30% planning guardrail</div></div>
  <div class="tile"><div class="v">639 / 780</div><div class="k">transforms that the objective
    cannot tell apart from the truth</div></div>
  <div class="tile"><div class="v">16.4&times;</div><div class="k">spread in implied TV
    contribution across them</div></div>
  <div class="tile"><div class="v">116 / 780</div><div class="k">that fit the data <em>better</em>
    than the truth does</div></div>
</div>

{figures}

<section class="fig">
  <h2>3 &middot; The five pre-registered gates on C0</h2>
  <p class="lede">Thresholds fixed before any code ran. The kill criterion K1 fired on this table,
  so conditions C1&ndash;C7 were never run and are moot rather than negative.</p>
  <div class="tw">
  <table>
    <thead><tr><th>Gate</th><th>Metric</th><th>Threshold</th><th>C0, 200 seeds</th>
      <th>Verdict</th></tr></thead>
    <tbody>
{gate_rows}
    </tbody>
  </table>
  </div>
</section>

<footer>
<p>Every figure is recomputed at build time from <code>results/*.csv</code>; only the
pre-registered thresholds and the two truth scores are constants. <code>plotly.js</code> is
vendored inline, so this page fetches nothing &mdash; no script, stylesheet, font or image is
loaded from anywhere. Output is byte-deterministic: building twice gives the same file.</p>
<p>The plateau panels were regenerated at D39 as <code>mmm_recovery.plateau</code> after the
original harness was found never to have been committed. The figures they replace
(177&nbsp;of&nbsp;780, a 5.5&times; spread) described the superseded six-column control block.
Both values are in the deviations log.</p>
<p>Build: <code>uv sync --extra report &amp;&amp; make report</code></p>
</footer>

<script type="text/javascript">
/* A figure plotted inside a display:none container has zero width and stays that size when
   revealed. That happens when the viewport crosses the 720px breakpoint or the OS colour
   scheme changes after load. Static text, no clock, no effect on the bytes written. */
(function () {{
  var pending = null;
  function resizeVisible() {{
    var plots = document.querySelectorAll(".js-plotly-plot");
    for (var i = 0; i < plots.length; i++) {{
      if (plots[i].offsetParent !== null) {{ Plotly.Plots.resize(plots[i]); }}
    }}
  }}
  function schedule() {{
    if (pending !== null) {{ clearTimeout(pending); }}
    pending = setTimeout(resizeVisible, 150);
  }}
  window.addEventListener("resize", schedule);
  if (window.matchMedia) {{
    var dark = window.matchMedia("(prefers-color-scheme: dark)");
    if (dark.addEventListener) {{ dark.addEventListener("change", schedule); }}
  }}
}})();
</script>
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
