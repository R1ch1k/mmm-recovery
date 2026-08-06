"""Tests for the dashboard (Step 8).

These pin the three properties the file is *for* — self-contained, deterministic, and actually
rendering — rather than its wording. The third is not obvious and is the one that already went
wrong: the first build produced a structurally perfect page with two blank charts, because the
vendored bundle sat at the end of `<body>` and every `Plotly.newPlot` call had already run.
Nothing in the HTML was malformed; only a browser would have told you.
"""

import csv
import json
import re

import pytest

from mmm_recovery.plateau import BAND, PLATEAU_CSV
from mmm_recovery.report import (
    DASHBOARD_HTML,
    G5_THRESHOLD,
    TRUE_TV_CONTRIBUTION,
    TRUTH_CV_NOISY,
    load_arms,
    render,
)


@pytest.fixture(scope="module")
def html() -> str:
    return render()


def test_the_output_is_byte_deterministic() -> None:
    """CLAUDE.md rule 4. Plotly seeds div ids from a uuid unless told otherwise; they are fixed."""
    assert render() == render()


def test_the_page_fetches_nothing(html: str) -> None:
    """Self-contained means no load target, which is a narrower claim than "no URL appears".

    The vendored bundle carries map-tile attribution URLs as string literals for trace types this
    page never instantiates. Asserting the absence of "http" would fail on those and would be a
    claim the file cannot honestly make, so the assertion is about what the browser would fetch.
    """
    assert "<script src=" not in html
    assert "<link " not in html
    assert not re.search(r"url\(\s*['\"]?https?:", html)
    assert not re.search(r"@import", html)
    assert "<img" not in html


CALL_SITE = re.compile(r'Plotly\.newPlot\(\s*"([a-z-]+)"')
"""Real call sites only. The vendored bundle contains the bare string ``Plotly.newPlot`` twice in
its own source, so counting the substring finds six where there are four."""


def test_plotly_is_defined_before_the_first_figure_script(html: str) -> None:
    """The regression that shipped two blank charts and no error in the HTML itself.

    `to_html` emits `Plotly.newPlot(...)` inline, which executes at parse time. If the bundle is
    below it the page throws "Plotly is not defined" four times and renders empty panels.
    """
    bundle = html.index('<script type="text/javascript">')
    first_call = CALL_SITE.search(html)

    assert first_call is not None
    assert bundle < html.index("</head>"), "the bundle must be in <head>"
    assert bundle < first_call.start(), "the bundle must load before any figure script"


def test_every_figure_is_present_with_a_fixed_id(html: str) -> None:
    """Four divs: two figures × two colour modes. Fixed ids are what makes the file reproducible."""
    ids = ["plateau-light", "arms-light", "plateau-dark", "arms-dark"]
    for div_id in ids:
        assert f'id="{div_id}"' in html
    assert CALL_SITE.findall(html) == ids


def test_the_headline_tiles_match_the_committed_plateau_csv(html: str) -> None:
    """The tiles are prose in the template; the figures come from the CSV. They must not drift."""
    with PLATEAU_CSV.open(encoding="utf-8", newline="") as handle:
        noisy = [float(r["cv_rmse"]) for r in csv.DictReader(handle) if r["series"] == "noisy"]

    tied = sum(score <= TRUTH_CV_NOISY * (1.0 + BAND) for score in noisy)
    better = sum(score < TRUTH_CV_NOISY for score in noisy)

    assert f"{tied} / {len(noisy)}" in html
    assert f"{better} / {len(noisy)}" in html
    assert f"{tied} of {len(noisy)} within the band" in html


def test_the_arms_are_the_four_the_readme_compares() -> None:
    """Recomputed from the CSVs, not typed in. Values pinned so the figure cannot drift silently."""
    arms = {arm.label: arm for arm in load_arms()}
    assert set(arms) == {
        "C0 baseline",
        "Flighted",
        "spend_log_sd = 1.00",
        "Guardrail m_c ∈ [0.7, 1.3]",
    }

    assert arms["C0 baseline"].g5 == pytest.approx(0.200, abs=5e-4)
    assert arms["C0 baseline"].n == 200
    assert arms["Flighted"].g5 == pytest.approx(0.207, abs=5e-4)
    assert arms["Flighted"].n == 198
    assert arms["spend_log_sd = 1.00"].g5 == pytest.approx(0.462, abs=5e-4)
    assert arms["spend_log_sd = 1.00"].n == 199
    assert arms["Guardrail m_c ∈ [0.7, 1.3]"].g5 == pytest.approx(0.760, abs=5e-4)
    assert arms["Guardrail m_c ∈ [0.7, 1.3]"].n == 200

    # The finding the figure exists to show: two arms cleared the baseline's interval, two did not.
    moved = {label for label, arm in arms.items() if arm.moved}
    assert moved == {"spend_log_sd = 1.00", "Guardrail m_c ∈ [0.7, 1.3]"}
    baseline = arms["C0 baseline"]
    for label in moved:
        assert arms[label].low > baseline.high
    assert arms["Flighted"].low < baseline.high


def test_no_arm_reaches_the_pre_registered_threshold() -> None:
    """G5 ≥ 0.90 is the gate. If an arm ever clears it, the dashboard's framing is wrong."""
    for arm in load_arms():
        assert arm.high < G5_THRESHOLD


def test_the_true_contribution_constant_matches_the_plateau_modules(html: str) -> None:
    """One number, two modules, one figure annotation. Drift here is a silent unit error."""
    assert pytest.approx(51_989, abs=1.0) == TRUE_TV_CONTRIBUTION

    # Plotly serialises figure text with ensure_ascii, so "£" reaches the file as "\u00a3".
    # Comparing against the literal would fail for a reason that has nothing to do with the number.
    def as_written(text: str) -> str:
        return json.dumps(text)[1:-1]

    assert as_written(f"truth, £{TRUE_TV_CONTRIBUTION:,.0f}k") in html
    assert as_written("Implied TV contribution, £k") in html


def test_the_committed_dashboard_matches_the_module_that_writes_it() -> None:
    """`results/dashboard.html` is committed, so it can go stale against the code.

    This catches that.
    """
    if not DASHBOARD_HTML.exists():  # pragma: no cover - only before the first build
        pytest.skip("run `make report` first")
    assert DASHBOARD_HTML.read_text(encoding="utf-8") == render()
