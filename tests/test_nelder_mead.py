"""Tests for the regenerated start-at-the-truth diagnostic (D43).

The optimiser run itself is ~40 seconds and is not repeated here. What is pinned is everything
that could make the run mean something other than it appears to: the coordinate transform, the
search space, the positive controls that show this harness scores on identical terms to
`plateau`, and the published numbers against the committed artefact.

The failure this exists to prevent is D39's: a figure whose harness was never committed drifted
past a control-block change that invalidated it, and nothing failed.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from mmm_recovery import nelder_mead
from mmm_recovery.dgp import condition_params, simulate
from mmm_recovery.estimator import Hyperparameters, SearchBounds, adstock_spend
from mmm_recovery.nelder_mead import (
    DIAGNOSTIC_CSV,
    DIAGNOSTIC_SEED,
    MAX_EVALUATIONS,
    Stage,
    _pack,
    _search_bounds,
    _unpack,
)
from mmm_recovery.plateau import _fit_at, truth_hyperparameters
from mmm_recovery.truth import incremental_contribution, response_surface


def test_the_coordinate_transform_round_trips() -> None:
    """`_pack` and `_unpack` must be inverses, or the optimiser is not where it reports being.

    This is the quietest way the diagnostic could be wrong: a κ rebuilt from the wrong λ would
    move the starting point off the truth, and the headline claim is *about* the starting point.
    """
    params = condition_params("C0")
    sim = simulate(params, DIAGNOSTIC_SEED)
    truth = truth_hyperparameters(params)

    recovered = _unpack(_pack(truth, sim.spend), sim.spend)

    assert recovered.decay == pytest.approx(truth.decay)
    assert recovered.hill_shape == pytest.approx(truth.hill_shape)
    assert recovered.half_saturation == pytest.approx(truth.half_saturation)
    assert recovered.ridge_penalty == pytest.approx(truth.ridge_penalty)


def test_kappa_is_rebuilt_from_the_current_decay_as_the_search_draws_it() -> None:
    """κ is a multiple of *observed* mean adstocked spend, which moves when λ moves.

    Holding κ in £k while λ varies would let the optimiser explore ratios the random search
    cannot draw, and the whole point of the diagnostic is that both search the same space.
    """
    params = condition_params("C0")
    sim = simulate(params, DIAGNOSTIC_SEED)
    truth = truth_hyperparameters(params)

    packed = _pack(truth, sim.spend)
    packed[: sim.spend.shape[1]] = 0.0  # every λ to zero; ratios untouched

    moved = _unpack(packed, sim.spend)
    ratio_at_truth = truth.half_saturation / adstock_spend(sim.spend, truth.decay).mean(axis=0)
    expected = ratio_at_truth * adstock_spend(sim.spend, moved.decay).mean(axis=0)

    assert moved.half_saturation == pytest.approx(expected)
    assert not np.allclose(moved.half_saturation, truth.half_saturation)


def test_the_optimiser_may_go_nowhere_the_random_search_could_not() -> None:
    """The search space is `SearchBounds` exactly, log-scaled where the search draws log-uniformly.

    A wider space would let the diagnostic walk somewhere the estimator would never consider,
    which would make "the optimiser prefers to leave the truth" a claim about the wrong space.
    """
    bounds = SearchBounds()
    packed = _search_bounds(5, bounds)

    assert len(packed) == 3 * 5 + 1
    assert packed[:5] == [bounds.decay] * 5
    assert packed[5:10] == [bounds.hill_shape] * 5
    for low, high in packed[10:15]:
        assert (np.exp(low), np.exp(high)) == pytest.approx(bounds.half_saturation_ratio)
    assert (np.exp(packed[15][0]), np.exp(packed[15][1])) == pytest.approx(bounds.ridge_penalty)


def test_the_starting_point_is_the_truth_and_scores_what_plateau_publishes() -> None:
    """The positive control, and the reason to believe this harness at all.

    `plateau` publishes the truth's CV RMSE as 30.94153 on the noisy series and 0.00002 on the
    noiseless one. Reproducing both from this module's own start point proves the two are
    scoring on identical terms — same controls, same folds, same ridge solve. Without this the
    diagnostic could be measuring a different configuration and read exactly the same.
    """
    params = condition_params("C0")
    sim = simulate(params, DIAGNOSTIC_SEED)
    truth = truth_hyperparameters(params)

    noisy_score, _ = _fit_at(sim, sim.sales, truth)
    noiseless_score, noiseless_contribution = _fit_at(sim, sim.noiseless_sales, truth)
    true_contribution = incremental_contribution(response_surface(sim))

    assert noisy_score == pytest.approx(30.94153, abs=1e-4)
    assert noiseless_score == pytest.approx(1.7959e-05, rel=1e-3)

    # "Exact at the true hyperparameters once D22's controls are in place" — the claim the
    # xfail reason makes, measured rather than asserted.
    bias = np.abs(noiseless_contribution / true_contribution - 1.0)
    assert float(np.max(bias)) < 1e-5


def test_the_evaluation_budget_exceeds_what_the_binding_arm_needs() -> None:
    """A result reported at the cap is "where it got to", not an optimum.

    The noisy arm converges at 32,407 evaluations; a budget below that would silently turn a
    converged finding into a truncated one with no visible difference in the numbers.
    """
    assert MAX_EVALUATIONS > 32_407


def test_the_committed_csv_matches_the_module_that_writes_it(tmp_path: Path) -> None:
    """Header, arms, stages, row count and formatting, against the writer rather than prose."""
    with DIAGNOSTIC_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    sim = simulate(condition_params("C0"), DIAGNOSTIC_SEED)
    assert len(rows) == 2 * 2 * len(sim.channel_names)
    assert {row["series"] for row in rows} == {"noisy", "noiseless"}
    assert {row["stage"] for row in rows} == {"start", "end"}

    stages: list[Stage] = []
    for series in ("noisy", "noiseless"):
        for stage in ("start", "end"):
            group = [r for r in rows if r["series"] == series and r["stage"] == stage]
            assert [r["channel"] for r in group] == list(sim.channel_names)
            stages.append(
                Stage(
                    series=series,
                    stage=stage,
                    cv_rmse=float(group[0]["cv_rmse"]),
                    hyper=Hyperparameters(
                        decay=np.array([float(r["decay"]) for r in group]),
                        hill_shape=np.array([float(r["hill_shape"]) for r in group]),
                        half_saturation=np.array([float(r["half_saturation"]) for r in group]),
                        ridge_penalty=float(group[0]["ridge_penalty"]),
                    ),
                    contribution=np.array([float(r["contribution"]) for r in group]),
                    relative_bias=np.array([float(r["relative_bias"]) for r in group]),
                    n_evaluations=int(group[0]["n_evaluations"]),
                    converged=group[0]["converged"] == "True",
                    message="",
                )
            )

    rewritten = tmp_path / "nelder_mead_diagnostic.csv"
    nelder_mead.write_csv(stages, sim.channel_names, rewritten)

    with DIAGNOSTIC_CSV.open(encoding="utf-8", newline="") as handle:
        committed = handle.read()
    with rewritten.open(encoding="utf-8", newline="") as handle:
        assert handle.read() == committed


def _stage(series: str, stage: str) -> dict[str, float]:
    with DIAGNOSTIC_CSV.open(encoding="utf-8", newline="") as handle:
        group = [
            row
            for row in csv.DictReader(handle)
            if row["series"] == series and row["stage"] == stage
        ]
    bias = np.abs(np.array([float(row["relative_bias"]) for row in group]))
    return {
        "cv_rmse": float(group[0]["cv_rmse"]),
        "median_bias": float(np.median(bias)),
        "max_bias": float(np.max(bias)),
        "converged": group[0]["converged"] == "True",
    }


def test_on_the_noisy_arm_the_optimiser_walks_away_from_the_truth() -> None:
    """D43's headline, pinned against the artefact.

    This is the result that separates non-identification from search failure: handed the right
    answer for free, the optimiser improves the objective and makes recovery six times worse.
    The pre-D43 figures — CV 3.63160 → 3.06272, bias 2.4% → 57.3% — came from the superseded
    six-column control block and are replaced by these.
    """
    start, end = _stage("noisy", "start"), _stage("noisy", "end")

    assert end["converged"]
    assert end["cv_rmse"] < start["cv_rmse"]
    assert start["cv_rmse"] == pytest.approx(30.94153, abs=1e-4)
    assert end["cv_rmse"] == pytest.approx(30.60899, abs=1e-4)
    assert start["median_bias"] == pytest.approx(0.115, abs=5e-4)
    assert end["median_bias"] == pytest.approx(0.709, abs=5e-4)
    assert end["max_bias"] == pytest.approx(0.866, abs=5e-4)


def test_on_the_noiseless_arm_the_optimiser_stays_at_the_truth() -> None:
    """The contrast that pins the mechanism, and the direction change D43 had to report.

    Under D22's ten-column controls the noiseless truth is uniquely identified, so there is
    nothing for the optimiser to walk toward: it shaves the score by a rounding error and leaves
    recovery exact. The pre-D43 claim was almost certainly this arm under six columns, where a
    structured residual gave it somewhere to go. "Better search makes recovery worse" is
    therefore a statement about the *noisy* series — the one every gate is computed on.
    """
    start, end = _stage("noiseless", "start"), _stage("noiseless", "end")

    assert end["converged"]
    assert start["max_bias"] < 1e-5
    assert end["max_bias"] < 1e-4
    assert end["cv_rmse"] < 1e-4
