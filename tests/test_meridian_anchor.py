"""Tests for the Meridian anchor (CLAUDE.md build order, Step 7).

Every test that needs the package is skipped without it, because `[meridian]` is an optional
extra and CLAUDE.md requires the core grid to run without it. The tests that do *not* need it
— the ones about what the anchor is allowed to see — run everywhere, since those are the ones
protecting non-negotiable rule 3.
"""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from mmm_recovery import meridian_anchor
from mmm_recovery.dgp import condition_params, simulate
from mmm_recovery.meridian_anchor import (
    ANCHOR_CONDITIONS,
    ANCHOR_JSON,
    INTERVAL_LEVEL,
    MERIDIAN_AVAILABLE,
    N_ANCHOR_SEEDS,
    SamplingSpec,
    fit_anchor,
    to_input_data,
)

needs_meridian = pytest.mark.skipif(
    not MERIDIAN_AVAILABLE, reason="the optional [meridian] extra is not installed"
)


def test_the_anchor_runs_only_the_conditions_section_4_names() -> None:
    """§4 scopes the anchor to C0, C3, C6 and C7 — verify-not-trust, not a second grid."""
    assert ANCHOR_CONDITIONS == ("C0", "C3", "C6", "C7")
    assert INTERVAL_LEVEL == 0.90, "G2 must grade both estimators on the same nominal interval"


def test_priors_are_not_a_parameter_anywhere_in_this_module() -> None:
    """§4 pre-commits to *default* priors, so there must be no knob that could move them.

    The interpretation §4 fixes — a prior that rescues C0 is supplying information the data
    does not contain — only holds if the priors were left alone. Making them configurable
    would make that sentence unfalsifiable, so `SamplingSpec` carries a compute budget and
    nothing else.
    """
    assert "prior" not in {field.lower() for field in SamplingSpec.__dataclass_fields__}
    for function in (fit_anchor, to_input_data):
        for name in inspect.signature(function).parameters:
            assert "prior" not in name.lower()


@needs_meridian
def test_the_anchor_sees_spend_and_sales_and_no_control_columns() -> None:
    """Rule 3 for the anchor: the frame it builds carries no latent series and no controls.

    Meridian models trend and seasonality with its own knot spline, so handing it our Fourier
    terms would be tuning it toward the DGP. Handing it `d_t` would invalidate C3 and C7 the
    same way it would for `RidgeMMM`.
    """
    sim = simulate(condition_params("C0"), 0)
    data = to_input_data(sim)

    assert data.controls is None, "the anchor must not receive control columns"
    assert data.non_media_treatments is None
    assert data.organic_media is None

    kpi = np.asarray(data.kpi, dtype=np.float64).ravel()
    assert np.allclose(kpi, sim.sales), "the KPI must be observed sales, noise included"
    media = np.asarray(data.media, dtype=np.float64).reshape(sim.spend.shape)
    assert np.allclose(media, sim.spend)
    assert not np.allclose(kpi, sim.noiseless_sales), "the anchor must not see noiseless sales"


@needs_meridian
def test_the_anchor_reports_convergence_rather_than_assuming_it() -> None:
    """An unconverged chain's posterior mean is an artefact of where it started.

    Reporting one as a failure of the method would be exactly the strawman K3 exists to
    prevent, so R-hat travels with every fit and the caller can see it. Run at a deliberately
    small budget — this asserts the plumbing, not the science.
    """
    sim = simulate(condition_params("C0"), 0)
    fit = fit_anchor(
        sim, seed=0, sampling=SamplingSpec(n_chains=2, n_adapt=50, n_burnin=50, n_keep=100)
    )
    assert fit.channel_names == sim.channel_names
    assert fit.contribution.shape == (len(sim.channel_names),)
    assert fit.interval.shape == (len(sim.channel_names), 2)
    assert np.all(fit.interval[:, 0] <= fit.interval[:, 1])
    assert np.isfinite(fit.worst_r_hat)
    assert fit.converged == (fit.worst_r_hat <= SamplingSpec().r_hat_ceiling)


def test_the_module_is_runnable_as_a_command() -> None:
    """The README documents `python -m mmm_recovery.meridian_anchor` as a 31-minute run.

    Until D38 the module had neither a `main()` nor a `__main__` block, so that command
    imported it and exited silently: a documented reproduction step that could not fail
    loudly. This pins the entry point rather than the science, which the ~31-minute
    `fit_anchor` path above covers.
    """
    source = Path(meridian_anchor.__file__).read_text(encoding="utf-8")

    assert callable(meridian_anchor.main)
    assert 'if __name__ == "__main__":' in source

    signature = inspect.signature(meridian_anchor.main)
    assert list(signature.parameters) == ["argv"]

    with pytest.raises(SystemExit) as exit_info:
        meridian_anchor.main(["--help"])
    assert exit_info.value.code == 0


def test_the_committed_anchor_output_matches_the_writer_that_now_produces_it() -> None:
    """`results/meridian_c0.json` must be byte-reproducible by the entry point.

    The file was written by an uncommitted harness with platform-default line endings, so it
    carried CRLF and no trailing newline while every other artefact in `results/` is written
    with an explicit `\n`. Normalised at D38. This fails if either side drifts again.
    """
    # `newline=""` disables translation so CRLF drift is visible; `Path.read_text` only grew
    # that argument in 3.13 and this project targets 3.12.
    with ANCHOR_JSON.open(encoding="utf-8", newline="") as handle:
        committed = handle.read()
    rows = json.loads(committed)

    assert committed == json.dumps(rows, indent=1) + "\n"
    assert [row["seed"] for row in rows] == list(range(N_ANCHOR_SEEDS))
    assert set(rows[0]) == {"seed", "g1", "coverage", "rhat", "converged", "rel", "seconds"}
