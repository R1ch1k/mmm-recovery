"""Tests for D26's exploratory spend-variation sweep.

The load-bearing assertion of `sweep.py` — that `spend_log_sd = 0.30` reproduces D23's
five-gate table — is not tested here. It costs 200 fits, and it is already enforced by
`verify_control` on every run that writes results, which is the moment it matters. What is
tested here is everything that decides whether that assertion means anything: that the sweep
moves one knob and no others, that a cell is a pure function of its inputs, that the gates
aggregate the way §6 and §7 define them, and that a failed solve is recorded rather than
absorbed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from mmm_recovery.dgp import condition_params
from mmm_recovery.sweep import (
    CONTROL_LEVEL,
    D23_CONTROL,
    CellResult,
    SeedOutcome,
    SolveFailure,
    descriptives,
    failures,
    gates,
    run_cell,
    successes,
    sweep_params,
    verify_control,
    wilson_interval,
    write_csv,
)


def make_outcome(
    *,
    level: float = 0.30,
    seed: int = 0,
    relative_bias: tuple[float, ...] = (0.1, 0.2, 0.3),
    covered: tuple[bool, ...] = (True, True, False),
    spearman: float = 0.5,
    regret: float = 0.4,
    beats: bool = True,
) -> SeedOutcome:
    """A `SeedOutcome` with every field explicit, for gate arithmetic."""
    return SeedOutcome(
        level=level,
        seed=seed,
        relative_bias=np.array(relative_bias, dtype=np.float64),
        covered=np.array(covered, dtype=np.bool_),
        spearman=spearman,
        regret=regret,
        beats_status_quo=beats,
        achievable_lift_share=0.0116,
        shortfall_from_optimum_share=0.0273,
        loss_vs_status_quo_share=0.0158,
        n_agreeing=3,
    )


class TestSweepParams:
    def test_moves_spend_log_sd_and_nothing_else(self) -> None:
        baseline = condition_params("C0")
        swept = sweep_params(0.60)
        assert swept.spend_log_sd == 0.60
        changed = {
            field.name
            for field in dataclasses.fields(baseline)
            if getattr(baseline, field.name) != getattr(swept, field.name)
        }
        assert changed == {"spend_log_sd"}

    def test_control_level_is_c0_exactly(self) -> None:
        """0.30 must *be* C0, or the control column is not a control."""
        assert sweep_params(CONTROL_LEVEL) == condition_params("C0")

    def test_c0_has_no_placebo_so_g6_is_undefined(self) -> None:
        """§7 applies G6 to C5-C7 only; the module must not invent a placebo here."""
        assert "placebo" not in [channel.name for channel in sweep_params(0.30).channels]


class TestGates:
    def test_g1_is_median_across_channels_then_across_seeds(self) -> None:
        outcomes = [
            make_outcome(seed=0, relative_bias=(0.1, -0.9, 0.5)),  # median |.| = 0.5
            make_outcome(seed=1, relative_bias=(0.2, 0.3, -0.7)),  # median |.| = 0.3
        ]
        assert gates(outcomes)["G1_median_abs_relative_bias"] == pytest.approx(0.4)

    def test_g1_uses_absolute_value_so_signs_cannot_cancel(self) -> None:
        outcomes = [make_outcome(relative_bias=(-0.6, -0.6, -0.6))]
        assert gates(outcomes)["G1_median_abs_relative_bias"] == pytest.approx(0.6)

    def test_g2_pools_channel_pairs_not_seed_means(self) -> None:
        """Coverage is over every (seed, channel) pair, so ragged seeds weight correctly."""
        outcomes = [
            make_outcome(seed=0, covered=(True, True, True)),
            make_outcome(seed=1, covered=(False, False, False)),
        ]
        assert gates(outcomes)["G2_coverage"] == pytest.approx(0.5)

    def test_g4_is_a_median_so_one_catastrophe_cannot_carry_it(self) -> None:
        outcomes = [make_outcome(seed=i, regret=r) for i, r in enumerate([0.1, 0.2, 900.0])]
        assert gates(outcomes)["G4_median_regret"] == pytest.approx(0.2)

    def test_g5_is_a_rate(self) -> None:
        outcomes = [
            make_outcome(seed=i, beats=b) for i, b in enumerate([True, False, False, False])
        ]
        assert gates(outcomes)["G5_beats_status_quo"] == pytest.approx(0.25)

    def test_empty_input_raises_rather_than_returning_nan(self) -> None:
        with pytest.raises(ValueError, match="no outcomes"):
            gates([])


class TestSolveFailureHandling:
    def test_failures_are_excluded_from_gates(self) -> None:
        good: list[CellResult] = [
            make_outcome(seed=0, regret=0.1),
            make_outcome(seed=1, regret=0.3),
        ]
        mixed: list[CellResult] = [
            *good,
            SolveFailure(0.30, 2, "recommendation-solve", "SLSQP failed from start 4"),
        ]
        assert gates(mixed) == gates(good)

    def test_split_helpers_partition_the_input(self) -> None:
        failure = SolveFailure(0.30, 2, "recommendation-solve", "boom")
        results: list[CellResult] = [make_outcome(seed=0), failure, make_outcome(seed=1)]
        assert len(successes(results)) == 2
        assert failures(results) == [failure]

    def test_worst_case_g5_counts_every_failure_as_a_loss(self) -> None:
        """The adverse bound is what keeps exclusion from being concealment."""
        results: list[CellResult] = [
            make_outcome(seed=0, beats=True),
            make_outcome(seed=1, beats=True),
            SolveFailure(0.30, 2, "recommendation-solve", "boom"),
        ]
        assert gates(results)["G5_beats_status_quo"] == pytest.approx(1.0)
        assert descriptives(results)["g5_worst_case"] == pytest.approx(2 / 3)
        assert descriptives(results)["n_solve_failures"] == pytest.approx(1.0)

    def test_a_non_slsqp_valueerror_is_not_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLAUDE.md: `except Exception` hiding real bugs as condition failures is banned."""
        from mmm_recovery import sweep

        def explode(*_args: object, **_kwargs: object) -> None:
            raise ValueError("something else entirely")

        monkeypatch.setattr(sweep, "optimal_allocation", explode)
        with pytest.raises(ValueError, match="something else entirely"):
            run_cell(0.30, 0)


class TestWilsonInterval:
    def test_brackets_the_point_estimate(self) -> None:
        low, high = wilson_interval(40, 200)
        assert low < 0.20 < high

    def test_g5_interval_excludes_its_threshold(self) -> None:
        """C0's 0.200 must be distinguishable from §7's 0.90, not merely below it."""
        _low, high = wilson_interval(40, 200)
        assert high < 0.90

    def test_stays_inside_the_unit_interval_at_the_extremes(self) -> None:
        """Where the normal approximation puts ends outside [0, 1], Wilson must not."""
        for n_successes in (0, 200):
            low, high = wilson_interval(n_successes, 200)
            assert 0.0 <= low <= high <= 1.0

    def test_narrows_as_the_sample_grows(self) -> None:
        small = wilson_interval(2, 10)
        large = wilson_interval(200, 1000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_zero_trials_raises(self) -> None:
        with pytest.raises(ValueError, match="n_trials must be positive"):
            wilson_interval(0, 0)


class TestVerifyControl:
    def _outcomes_hitting(self, targets: dict[str, float]) -> list[SeedOutcome]:
        """Two seeds whose gates land exactly on `targets`."""
        bias = targets["G1_median_abs_relative_bias"]
        return [
            make_outcome(
                seed=index,
                relative_bias=(bias, bias, bias),
                covered=(True, False, False),
                spearman=targets["G3_median_spearman"],
                regret=targets["G4_median_regret"],
                beats=index == 0,
            )
            for index in range(2)
        ]

    def test_raises_when_the_control_disagrees_with_d23(self) -> None:
        wrong = dict(D23_CONTROL, G4_median_regret=99.0)
        with pytest.raises(AssertionError, match="does not reproduce D23"):
            verify_control(self._outcomes_hitting(wrong))

    def test_names_every_disagreeing_gate(self) -> None:
        with pytest.raises(AssertionError) as caught:
            verify_control(self._outcomes_hitting(dict(D23_CONTROL, G4_median_regret=99.0)))
        assert "G4_median_regret" in str(caught.value)

    def test_raises_when_the_control_contains_a_solve_failure(self) -> None:
        results: list[CellResult] = [
            *self._outcomes_hitting(D23_CONTROL),
            SolveFailure(CONTROL_LEVEL, 5, "recommendation-solve", "SLSQP failed from start 4"),
        ]
        with pytest.raises(AssertionError, match="must solve cleanly"):
            verify_control(results)


class TestDeterminism:
    def test_a_cell_is_a_pure_function_of_level_and_seed(self) -> None:
        """CLAUDE.md rule 4. Worker count and completion order must not reach the numbers."""
        first, second = run_cell(0.30, 3), run_cell(0.30, 3)
        assert isinstance(first, SeedOutcome) and isinstance(second, SeedOutcome)
        np.testing.assert_array_equal(first.relative_bias, second.relative_bias)
        np.testing.assert_array_equal(first.covered, second.covered)
        assert first.regret == second.regret
        assert first.spearman == second.spearman
        assert first.beats_status_quo == second.beats_status_quo

    def test_different_levels_give_different_data(self) -> None:
        low, high = run_cell(0.15, 3), run_cell(0.60, 3)
        assert isinstance(low, SeedOutcome) and isinstance(high, SeedOutcome)
        assert low.regret != high.regret


class TestCsv:
    def test_failed_cells_appear_rather_than_vanishing(self, tmp_path: Path) -> None:
        """An absent seed must not be confusable with a seed that was never attempted."""
        path = tmp_path / "sweep.csv"
        results: list[CellResult] = [
            make_outcome(seed=0),
            SolveFailure(0.30, 1, "recommendation-solve", "SLSQP failed from start 4"),
        ]
        write_csv(results, ("tv", "video", "search"), path)
        text = path.read_text(encoding="utf-8")
        assert "solve_failed" in text
        assert "0.30,1,all,solve_failed,1" in text

    def test_header_and_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "sweep.csv"
        write_csv([make_outcome(seed=0)], ("tv", "video", "search"), path)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert lines[0] == "spend_log_sd,seed,channel,metric,value"
        # 3 channels x 2 metrics, plus 8 seed-level metrics
        assert len(lines) == 1 + 3 * 2 + 8
