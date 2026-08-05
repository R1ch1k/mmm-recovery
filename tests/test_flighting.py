"""Tests for D34's flighted-spend validity check.

The check is only worth anything if two things hold: the baseline arm is C0 *untouched*, and the
flighted arm differs from it in the spend process and nothing else. Both are asserted here, along
with the independence property that keeps this from silently becoming C1's collinearity test.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mmm_recovery.dgp import DGPParams, condition_params, duty_cycles, simulate
from mmm_recovery.flighting import (
    BURST_WEEKS,
    FLIGHTED_CHANNELS,
    baseline_params,
    flighted_params,
)


class TestArmsDifferOnlyInTheSpendProcess:
    def test_baseline_arm_is_c0_exactly(self) -> None:
        assert baseline_params() == condition_params("C0")

    def test_flighted_arm_moves_only_the_flighting_knobs(self) -> None:
        changed = {
            field.name
            for field in dataclasses.fields(DGPParams)
            if getattr(baseline_params(), field.name) != getattr(flighted_params(), field.name)
        }
        assert changed == {"flighted_channels"}

    def test_the_default_dgp_consumes_no_flighting_randomness(self) -> None:
        """C0 must be bit-identical to before the knob existed, or D23 stops reproducing."""
        without = simulate(condition_params("C0"), 3)
        explicit = simulate(dataclasses.replace(condition_params("C0"), flighted_channels=()), 3)
        np.testing.assert_array_equal(without.spend, explicit.spend)
        np.testing.assert_array_equal(without.sales, explicit.sales)


class TestFlightingBehaviour:
    def test_flighted_channels_go_dark_and_others_do_not(self) -> None:
        cycles = duty_cycles(simulate(flighted_params(), 0))
        for name, cycle in cycles.items():
            if name in FLIGHTED_CHANNELS:
                assert 0.2 < cycle < 0.9, f"{name} duty cycle {cycle}"
            else:
                assert cycle == 1.0, f"{name} should be always-on"

    def test_duty_cycle_lands_in_the_designed_band(self) -> None:
        """D34 committed to roughly 40-60%. Averaged over seeds, not promised per seed."""
        cycles = [
            value
            for seed in range(12)
            for name, value in duty_cycles(simulate(flighted_params(), seed)).items()
            if name in FLIGHTED_CHANNELS
        ]
        assert 0.40 <= float(np.mean(cycles)) <= 0.60

    def test_each_channel_keeps_its_total_budget(self) -> None:
        """Same money, bought in bursts — otherwise this measures "spent less", not "flighted"."""
        baseline = simulate(baseline_params(), 5)
        flighted = simulate(flighted_params(), 5)
        np.testing.assert_allclose(
            baseline.spend.sum(axis=0), flighted.spend.sum(axis=0), rtol=1e-9
        )

    def test_always_on_channels_are_untouched(self) -> None:
        baseline = simulate(baseline_params(), 5)
        flighted = simulate(flighted_params(), 5)
        for index, name in enumerate(baseline.channel_names):
            if name not in FLIGHTED_CHANNELS:
                np.testing.assert_array_equal(baseline.spend[:, index], flighted.spend[:, index])

    def test_channels_flight_independently(self) -> None:
        """Synchronised flighting would smuggle in C1's collinearity through the wrong knob.

        The live/dark indicators of the three flighted channels must not move together. Perfect
        independence is not asserted — these are finite samples — but a correlation near 1 would
        mean the schedules are shared.
        """
        spend = simulate(flighted_params(), 0).spend
        names = simulate(flighted_params(), 0).channel_names
        columns = [index for index, name in enumerate(names) if name in FLIGHTED_CHANNELS]
        live = (spend[:, columns] > 0).astype(np.float64)
        correlations = np.corrcoef(live, rowvar=False)
        off_diagonal = correlations[~np.eye(len(columns), dtype=bool)]
        assert np.abs(off_diagonal).max() < 0.5

    def test_flighting_is_deterministic_in_the_seed(self) -> None:
        np.testing.assert_array_equal(
            simulate(flighted_params(), 7).spend, simulate(flighted_params(), 7).spend
        )

    def test_different_seeds_give_different_schedules(self) -> None:
        assert not np.array_equal(
            simulate(flighted_params(), 7).spend > 0, simulate(flighted_params(), 8).spend > 0
        )

    def test_burst_lengths_stay_within_the_declared_range(self) -> None:
        """Blocks are 2-6 weeks apart from the truncated first block and the final partial one."""
        spend = simulate(flighted_params(), 0).spend
        names = simulate(flighted_params(), 0).channel_names
        index = names.index(FLIGHTED_CHANNELS[0])
        live = spend[:, index] > 0
        edges = np.flatnonzero(np.diff(live.astype(np.int8)) != 0)
        interior = np.diff(edges)
        assert interior.size > 0
        assert interior.min() >= BURST_WEEKS[0]
        assert interior.max() <= BURST_WEEKS[1]


class TestValidation:
    def test_an_unknown_channel_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no such channel"):
            dataclasses.replace(condition_params("C0"), flighted_channels=("radio",))

    def test_a_reversed_burst_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="flight_burst_weeks"):
            dataclasses.replace(
                condition_params("C0"), flighted_channels=("tv",), flight_burst_weeks=(6, 2)
            )
