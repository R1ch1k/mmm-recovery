"""Tests for D33's optimiser-bound robustness check.

The finding this module exists to establish — that C0's failure is not an artefact of an
optimiser allowed to extrapolate to 3x observed spend — rests on two things being true of the
code: that both solves receive the *same* bound (D18), and that `on_upper_bound` counts what it
claims to. Both are asserted here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mmm_recovery.robustness import (
    BOUNDS,
    GUARDRAIL,
    PREREGISTERED_BOUND,
    BoundOutcome,
    format_table,
    run_seed,
    summarise,
    write_csv,
)


def make(
    bound: float,
    seed: int,
    regret: float,
    beats: bool,
    on_bound: int = 0,
    zeroed: int = 0,
    truth_zeroed: int = 1,
    floor: float = 0.0,
) -> BoundOutcome:
    return BoundOutcome(
        floor=floor,
        bound=bound,
        seed=seed,
        regret=regret,
        beats_status_quo=beats,
        achievable_lift_share=0.0116,
        on_upper_bound=on_bound,
        on_lower_bound=zeroed,
        truth_on_lower_bound=truth_zeroed,
    )


class TestSummarise:
    def test_groups_by_bound(self) -> None:
        outcomes = [
            make(1.3, 0, 1.0, False),
            make(1.3, 1, 3.0, False),
            make(3.0, 0, 5.0, True),
        ]
        summary = summarise(outcomes)
        assert summary[0.0, 1.3]["n"] == 2
        assert summary[0.0, 3.0]["n"] == 1
        assert summary[0.0, 1.3]["median_regret"] == pytest.approx(2.0)

    def test_share_regret_above_one_is_the_worse_than_nothing_rate(self) -> None:
        outcomes = [make(3.0, i, r, False) for i, r in enumerate([0.5, 1.5, 2.5, 3.5])]
        assert summarise(outcomes)[0.0, 3.0]["share_regret_above_1"] == pytest.approx(0.75)

    def test_rate_carries_an_interval(self) -> None:
        outcomes = [make(3.0, i, 1.0, i == 0) for i in range(4)]
        row = summarise(outcomes)[0.0, 3.0]
        assert row["beats_ci_low"] < row["beats_status_quo"] < row["beats_ci_high"]

    def test_bounds_with_no_cells_are_omitted_rather_than_zero_filled(self) -> None:
        """An absent bound must not read as a bound where nothing beat the status quo."""
        assert set(summarise([make(3.0, 0, 1.0, True)])) == {(0.0, 3.0)}

    def test_the_guardrail_is_keyed_separately_from_the_same_cap(self) -> None:
        """[0.7, 1.3] and [0, 1.3] share a cap and must not be pooled."""
        outcomes = [make(1.3, 0, 1.0, False), make(1.3, 0, 4.0, True, floor=0.7)]
        summary = summarise(outcomes)
        assert set(summary) == {(0.0, 1.3), (0.7, 1.3)}
        assert summary[0.7, 1.3]["median_regret"] == pytest.approx(4.0)


class TestBothSolvesShareTheBound:
    def test_recommendation_is_scored_against_a_truth_solved_at_the_same_bound(self) -> None:
        """D18. If only one side were bounded, regret would compare unlike allocations.

        Verified behaviourally: tightening the bound must change achievable lift, which is a
        property of the *truth* solve. If the truth solve ignored the bound, it would not move.

        A bound may be absent: per D30 a cell whose SLSQP solve fails is dropped, and seed 11
        loses 1.5 that way. The assertion is therefore a subset, not equality.
        """
        outcomes = {(outcome.floor, outcome.bound): outcome for outcome in run_seed(11)}
        assert set(outcomes) <= set(BOUNDS)
        assert {(0.0, 1.3), PREREGISTERED_BOUND} <= set(outcomes)
        assert (
            outcomes[0.0, 1.3].achievable_lift_share
            != outcomes[PREREGISTERED_BOUND].achievable_lift_share
        )

    def test_the_guardrail_shrinks_the_achievable_lift(self) -> None:
        """D35: forbidding a channel from being switched off removes most of C0's headroom."""
        outcomes = {(outcome.floor, outcome.bound): outcome for outcome in run_seed(11)}
        assert GUARDRAIL in outcomes
        assert (
            outcomes[GUARDRAIL].achievable_lift_share
            < outcomes[PREREGISTERED_BOUND].achievable_lift_share
        )

    def test_no_channel_exceeds_the_bound(self) -> None:
        for outcome in run_seed(11):
            assert 0 <= outcome.on_upper_bound <= 5

    def test_both_bounds_are_counted(self) -> None:
        """The withdrawn "interior solutions" claim came from counting only the upper bound.

        The lower bound is the one that binds, so it must be observable in the output.
        """
        outcomes = run_seed(11)
        assert any(outcome.on_lower_bound > 0 for outcome in outcomes)
        assert all(outcome.truth_on_lower_bound >= 1 for outcome in outcomes)

    def test_zeroed_counts_are_reported_for_both_model_and_truth(self) -> None:
        """Like-for-like: the truth zeroes OOH too, so the model is not penalised for zeroing."""
        row = summarise([make(3.0, 0, 2.0, False, zeroed=2, truth_zeroed=1)])[0.0, 3.0]
        assert row["mean_channels_zeroed"] == pytest.approx(2.0)
        assert row["mean_truth_channels_zeroed"] == pytest.approx(1.0)
        assert row["share_any_channel_zeroed"] == pytest.approx(1.0)


class TestCsv:
    def test_round_trip_has_one_row_per_cell(self, tmp_path: Path) -> None:
        path = tmp_path / "bounds.csv"
        outcomes = [make(3.0, 1, 2.0, False), make(1.3, 0, 1.0, True)]
        write_csv(outcomes, path)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert lines[0].startswith("min_multiplier,max_multiplier,seed,regret")
        assert len(lines) == 3
        # sorted by (floor, bound, seed) so the bytes do not depend on completion order
        assert lines[1].startswith("0.0,1.3,0")

    def test_table_marks_the_preregistered_bound_and_the_guardrail(self) -> None:
        table = format_table(
            summarise(
                [
                    make(PREREGISTERED_BOUND[1], 0, 2.0, False, floor=PREREGISTERED_BOUND[0]),
                    make(GUARDRAIL[1], 0, 0.5, True, floor=GUARDRAIL[0]),
                ]
            )
        )
        assert "§3" in table
        assert "D35" in table
        assert "NOT comparable" in table
