"""Tests for the regenerated plateau sweep (D39).

The sweep itself is a ~3-minute run and is not repeated here. What is pinned is the part that
went wrong the first time: the published numbers drifted away from the configuration that
produced them, and nothing failed. These tests pin the configuration, the band definition, and
the committed CSV against the module that writes it.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from mmm_recovery import plateau
from mmm_recovery.dgp import condition_params, simulate
from mmm_recovery.estimator import SearchBounds
from mmm_recovery.plateau import (
    BAND,
    N_ALPHA,
    N_RATIO,
    PLATEAU_CSV,
    PLATEAU_SEED,
    SWEPT_CHANNEL,
    PlateauCell,
    near_tied,
    truth_hyperparameters,
)
from mmm_recovery.truth import incremental_contribution, response_surface


def test_the_grid_is_780_points_matching_the_original() -> None:
    """26 × 30. The original's grid size is matched so the counts are comparable at all."""
    assert N_ALPHA * N_RATIO == 780


def test_the_swept_ranges_are_the_estimators_own_search_bounds() -> None:
    """No grid point may sit anywhere the random search could not have gone.

    A sweep over a wider range than the search would manufacture near-ties the estimator would
    never actually consider; a narrower one would hide them.
    """
    bounds = SearchBounds()
    alphas = np.linspace(bounds.hill_shape[0], bounds.hill_shape[1], N_ALPHA)
    ratios = np.geomspace(bounds.half_saturation_ratio[0], bounds.half_saturation_ratio[1], N_RATIO)

    assert alphas[0] == bounds.hill_shape[0]
    assert alphas[-1] == bounds.hill_shape[1]
    assert ratios[0] == pytest.approx(bounds.half_saturation_ratio[0])
    assert ratios[-1] == pytest.approx(bounds.half_saturation_ratio[1])


def test_the_true_transform_is_interior_to_the_swept_range() -> None:
    """A truth on the boundary would make "the grid cannot distinguish it" uninteresting."""
    params = condition_params("C0")
    sim = simulate(params, PLATEAU_SEED)
    index = sim.channel_names.index(SWEPT_CHANNEL)
    truth = truth_hyperparameters(params)
    bounds = SearchBounds()

    from mmm_recovery.estimator import adstock_spend

    mean_adstocked = float(adstock_spend(sim.spend, truth.decay)[:, index].mean())
    true_ratio = float(truth.half_saturation[index]) / mean_adstocked

    assert bounds.hill_shape[0] < truth.hill_shape[index] < bounds.hill_shape[1]
    assert bounds.half_saturation_ratio[0] < true_ratio < bounds.half_saturation_ratio[1]


def test_the_seed_is_the_one_the_original_sweeps_bias_arithmetic_implies() -> None:
    """C0 seed 0 gives a true TV contribution of £51,989k.

    This is how D39 establishes that the regeneration and the lost original share a draw rather
    than assuming it: the original reported a maximum bias of 362.6% at a contribution of
    £240,522k, and 240522 / (1 + 3.626) = 51,993 — this number to four significant figures.
    """
    sim = simulate(condition_params("C0"), PLATEAU_SEED)
    index = sim.channel_names.index(SWEPT_CHANNEL)
    true_contribution = float(incremental_contribution(response_surface(sim))[index])

    implied_by_the_originals_arithmetic = 240_522 / (1.0 + 3.626)

    assert true_contribution == pytest.approx(51_989, abs=1.0)
    assert implied_by_the_originals_arithmetic == pytest.approx(true_contribution, rel=1e-3)


def test_the_band_is_multiplicative_on_the_truths_score_and_includes_the_truth() -> None:
    """The band definition is the thing most easily changed to produce a nicer number.

    Pinned as arithmetic on hand-built cells so it cannot drift with the data.
    """

    def cell(score: float) -> PlateauCell:
        return PlateauCell("noisy", 1.0, 1.0, 1.0, score, 1.0, 0.0)

    truth_score = 100.0
    cells = [cell(99.0), cell(100.0), cell(100.9), cell(101.1), cell(200.0)]

    assert BAND == 0.01
    assert [c.cv_rmse for c in near_tied(cells, truth_score)] == [99.0, 100.0, 100.9]


def test_the_committed_csv_matches_the_module_that_writes_it(tmp_path: Path) -> None:
    """Header, arms, row count and formatting, against the writer rather than against prose.

    The failure D39 exists to prevent is exactly this drifting apart unnoticed.
    """
    with PLATEAU_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2 * N_ALPHA * N_RATIO
    assert set(rows[0]) == {
        "series",
        "hill_shape",
        "half_saturation_ratio",
        "half_saturation",
        "cv_rmse",
        "tv_contribution",
        "relative_bias",
    }
    assert {row["series"] for row in rows} == {"noisy", "noiseless"}
    for series in ("noisy", "noiseless"):
        assert sum(row["series"] == series for row in rows) == N_ALPHA * N_RATIO

    cells = [
        PlateauCell(
            series=row["series"],
            hill_shape=float(row["hill_shape"]),
            half_saturation_ratio=float(row["half_saturation_ratio"]),
            half_saturation=float(row["half_saturation"]),
            cv_rmse=float(row["cv_rmse"]),
            tv_contribution=float(row["tv_contribution"]),
            relative_bias=float(row["relative_bias"]),
        )
        for row in rows
    ]
    rewritten = tmp_path / "plateau_sweep.csv"
    plateau.write_csv(cells, rewritten)

    with PLATEAU_CSV.open(encoding="utf-8", newline="") as handle:
        committed = handle.read()
    with rewritten.open(encoding="utf-8", newline="") as handle:
        assert handle.read() == committed


def test_the_published_plateau_numbers_are_the_regenerated_ones(tmp_path: Path) -> None:
    """D39's headline figures, recomputed from the committed CSV.

    The pre-D39 README quoted 177 of 780 and a 5.5× spread from a harness nobody could run. This
    asserts the replacements against the artefact, so the write-up and the data cannot drift
    apart again without a red test.
    """
    with PLATEAU_CSV.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["series"] == "noisy"]

    scores = np.array([float(row["cv_rmse"]) for row in rows])
    contributions = np.array([float(row["tv_contribution"]) for row in rows])

    truth_score = 30.94153
    tied = scores <= truth_score * (1.0 + BAND)

    assert int(tied.sum()) == 639
    assert int((scores < truth_score).sum()) == 116
    assert contributions[tied].min() == pytest.approx(15_138, abs=1.0)
    assert contributions[tied].max() == pytest.approx(248_075, abs=1.0)
    assert contributions[tied].max() / contributions[tied].min() == pytest.approx(16.4, abs=0.05)
