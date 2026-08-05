"""Tests for interventional ground truth (CLAUDE.md build order, Step 3).

One test here was an `xfail(strict=True)`: CLAUDE.md's "multi-start solutions agree within
0.1%" is empirically false for this response surface, and the reason is a property of the DGP
rather than a defect in the solver. **D17 resolved it** — the requirement was wrong, because
requiring agreement on a non-concave surface is requiring convexity. The standard is now that
the returned optimum matches a 64-start reference, and the disagreement between starts is
kept as a reported diagnostic rather than dropped along with the test.

Seeds are fixed throughout and every figure quoted in a docstring was measured.
"""

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, minimize

from mmm_recovery.dgp import (
    CONDITION_LEVELS,
    PLACEBO,
    REAL_CHANNELS,
    DGPParams,
    condition_params,
    evaluate,
    simulate,
)
from mmm_recovery.truth import (
    AGREEMENT_TOLERANCE,
    MAX_MULTIPLIER,
    ResponseSurface,
    Vector,
    incremental_contribution,
    marginal_roas,
    optimal_allocation,
    response_surface,
)

ALL_CELLS = [(cond, level) for cond, levels in CONDITION_LEVELS.items() for level in levels]
PLACEBO_CELLS = [("C5", None), ("C6", 0.3), ("C6", 0.45), ("C6", 0.6), ("C7", None)]


def misspecified_with_placebo() -> DGPParams:
    """C4's misspecified forms plus a placebo — the combination C4 alone cannot test.

    C4 has no placebo channel and C7 has four other knobs moved at once, so neither isolates
    "does a zero-β channel stay exactly zero under Weibull adstock and logistic saturation".
    This does.
    """
    return DGPParams(channels=(*REAL_CHANNELS, PLACEBO), placebo_coupling=0.45, misspecified=True)


# --------------------------------------------------------------------------------------
# The response surface reproduces the generating process
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("condition", "level"), ALL_CELLS, ids=lambda v: str(v))
def test_surface_at_the_status_quo_is_bit_identical_to_the_dgp(
    condition: str, level: float | int | None
) -> None:
    """Adstocking once and rescaling must not change the answer at m = 1.

    The per-channel contributions are bit-identical. The *total* is equal to within one ulp
    rather than exactly, because the two sum in different orders: the DGP adds baseline and
    media week by week and then totals, while `total_sales` totals each separately and adds
    once. Measured gap at C0 seed 3 is 7e-11 on 5.1e5, or 1.4e-16 relative — floating-point
    associativity, not a modelling difference. Asserting `==` here would be asserting a
    property of numpy's summation order.
    """
    sim = simulate(condition_params(condition, level), 3)
    surface = response_surface(sim)
    assert np.array_equal(surface.contributions(surface.status_quo()), sim.contributions)
    assert surface.total_sales(surface.status_quo()) == pytest.approx(
        float(sim.noiseless_sales.sum()), rel=1e-14
    )


@pytest.mark.parametrize("condition", ["C0", "C4", "C7"])
def test_rescaling_adstocked_spend_equals_adstocking_rescaled_spend(condition: str) -> None:
    """The linearity the whole `ResponseSurface` shortcut rests on, asserted not assumed.

    Both adstock forms are linear filters, so ``adstock(m·x) == m·adstock(x)``. If that ever
    stopped holding, every interventional quantity in this module would be subtly wrong while
    still looking plausible. Measured relative gap on these cells: exactly 0.
    """
    sim = simulate(condition_params(condition), 3)
    surface = response_surface(sim)
    rng = np.random.default_rng(99)
    for _ in range(5):
        multipliers = rng.uniform(0.0, MAX_MULTIPLIER, size=surface.n_channels)
        direct = evaluate(sim.params, sim.spend * multipliers[None, :], sim.demand, sim.season)
        assert surface.total_sales(multipliers) == pytest.approx(
            float(direct.noiseless_sales.sum()), rel=1e-12
        )


def test_media_gradient_matches_a_central_difference() -> None:
    """A wrong analytic gradient converges confidently to the wrong point, so check it."""
    for condition in ("C0", "C4"):
        sim = simulate(condition_params(condition), 3)
        surface = response_surface(sim)
        rng = np.random.default_rng(7)
        for _ in range(3):
            point = rng.uniform(0.2, 2.5, size=surface.n_channels)
            analytic = surface.media_gradient(point)
            for index in range(surface.n_channels):
                step = 1e-6
                high, low = point.copy(), point.copy()
                high[index] += step
                low[index] -= step
                numeric = (surface.media_total(high) - surface.media_total(low)) / (2 * step)
                assert analytic[index] == pytest.approx(numeric, rel=1e-5, abs=1e-6)


# --------------------------------------------------------------------------------------
# The placebo contributes exactly nothing
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("condition", "level"), PLACEBO_CELLS, ids=lambda v: str(v))
def test_true_placebo_contribution_is_exactly_zero(
    condition: str, level: float | int | None
) -> None:
    """β = 0 means exactly 0.0, checked with ==, not approx.

    Its contribution column is exactly zero at every multiplier, so the totals either side of
    ``do(spend_placebo := 0)`` are summed from identical arrays and cancel bit for bit. mROAS
    inherits the same cancellation.
    """
    sim = simulate(condition_params(condition, level), 4)
    surface = response_surface(sim)
    placebo = sim.channel_names.index("placebo")

    assert np.array_equal(
        surface.contributions(surface.status_quo())[:, placebo], np.zeros(sim.params.n_weeks)
    )
    assert incremental_contribution(surface)[placebo] == 0.0
    assert marginal_roas(surface)[placebo] == 0.0


def test_placebo_is_exactly_zero_under_the_misspecified_forms_too() -> None:
    """Weibull adstock and zero-anchored logistic saturation, isolated from C7's other knobs.

    This is what D3 bought. With the literal logistic of §2 the placebo would have emitted
    about 1.8% of β at zero spend — except β is 0, so the giveaway would instead have been
    every *real* channel emitting sales while spending nothing.
    """
    sim = simulate(misspecified_with_placebo(), 4)
    surface = response_surface(sim)
    placebo = sim.channel_names.index("placebo")
    assert np.array_equal(
        surface.contributions(surface.status_quo())[:, placebo], np.zeros(sim.params.n_weeks)
    )
    assert incremental_contribution(surface)[placebo] == 0.0
    assert marginal_roas(surface)[placebo] == 0.0


@pytest.mark.parametrize(("condition", "level"), PLACEBO_CELLS, ids=lambda v: str(v))
def test_zeroing_the_placebo_leaves_noiseless_sales_bit_identical(
    condition: str, level: float | int | None
) -> None:
    """CLAUDE.md's single most important test, now through the intervention API.

    `dgp` asserts the same property about the generating process; this asserts it about the
    `do()` path that ground truth is actually read from, so a failure localises to one or the
    other rather than to "somewhere in the truth pipeline".
    """
    sim = simulate(condition_params(condition, level), 4)
    surface = response_surface(sim)
    placebo = sim.channel_names.index("placebo")
    intervened = surface.status_quo()
    intervened[placebo] = 0.0
    assert np.array_equal(
        surface.contributions(intervened), surface.contributions(surface.status_quo())
    )
    assert surface.total_sales(intervened) == surface.total_sales(surface.status_quo())


# --------------------------------------------------------------------------------------
# Contributions and mROAS
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("condition", "level"), ALL_CELLS, ids=lambda v: str(v))
def test_contributions_are_non_negative_and_sum_below_total_sales(
    condition: str, level: float | int | None
) -> None:
    sim = simulate(condition_params(condition, level), 5)
    surface = response_surface(sim)
    contributions = incremental_contribution(surface)
    assert np.all(contributions >= 0.0)
    assert float(contributions.sum()) < surface.total_sales(surface.status_quo())
    assert np.all(np.isfinite(contributions))


@pytest.mark.parametrize(("condition", "level"), ALL_CELLS, ids=lambda v: str(v))
def test_marginal_roas_is_non_negative_and_finite(
    condition: str, level: float | int | None
) -> None:
    sim = simulate(condition_params(condition, level), 5)
    roas = marginal_roas(response_surface(sim))
    assert np.all(roas >= 0.0)
    assert np.all(np.isfinite(roas))


def test_incremental_contribution_matches_a_direct_dgp_intervention() -> None:
    """Cross-check the fast path against `dgp.evaluate` with the column literally zeroed."""
    for condition in ("C0", "C4"):
        sim = simulate(condition_params(condition), 5)
        surface = response_surface(sim)
        fast = incremental_contribution(surface)
        reference = float(sim.noiseless_sales.sum())
        for index in range(surface.n_channels):
            zeroed = sim.spend.copy()
            zeroed[:, index] = 0.0
            direct = evaluate(sim.params, zeroed, sim.demand, sim.season)
            expected = reference - float(direct.noiseless_sales.sum())
            assert fast[index] == pytest.approx(expected, rel=1e-12)


def test_marginal_roas_is_the_prereg_difference_quotient() -> None:
    """§3's definition, recomputed the long way round through the DGP."""
    sim = simulate(condition_params("C0"), 5)
    surface = response_surface(sim)
    roas = marginal_roas(surface, bump=0.1)
    base = float(sim.noiseless_sales.sum())
    for index in range(surface.n_channels):
        bumped = sim.spend.copy()
        bumped[:, index] *= 1.1
        direct = evaluate(sim.params, bumped, sim.demand, sim.season)
        expected = (float(direct.noiseless_sales.sum()) - base) / (
            0.1 * float(sim.spend[:, index].sum())
        )
        assert roas[index] == pytest.approx(expected, rel=1e-10)


def test_marginal_roas_rejects_a_non_positive_bump() -> None:
    surface = response_surface(simulate(condition_params("C0"), 0))
    for bump in (0.0, -0.1):
        with pytest.raises(ValueError, match="bump"):
            marginal_roas(surface, bump=bump)


# --------------------------------------------------------------------------------------
# D2 — the optimiser's range stays differentiable
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("condition", "level"), ALL_CELLS, ids=lambda v: str(v))
def test_d2_no_channel_reaches_the_flat_region_at_the_upper_bound(
    condition: str, level: float | int | None
) -> None:
    """D2, asserted rather than assumed, for every cell of the grid.

    `response_surface` raises if any channel's saturation is exactly 1.0 or has a zero
    gradient at m_c = 3. Measured headroom between the peak adstocked spend at m = 3 and the
    start of the float64-flat region is between 1.93x (search, C7) and 5.02x (OOH, C7); at
    D2's scale of s = κ/4 the flat region begins at 11.175·κ.
    """
    sim = simulate(condition_params(condition, level), 6)
    surface = response_surface(sim)
    for index, channel in enumerate(sim.params.channels):
        peak = float(surface.adstocked[:, index].max()) * MAX_MULTIPLIER
        assert peak < channel.half_saturation * 11.175


def test_d2_check_fires_when_the_logistic_scale_is_too_small() -> None:
    """The positive control: a scale small enough to reach the plateau must be rejected.

    Without this the D2 assertion could be passing because it never fires. s = κ/400 puts the
    flat region at about 1.09·κ, which every channel clears immediately.
    """
    params = DGPParams(misspecified=True, logistic_scale_ratio=1.0 / 400.0)
    sim = simulate(params, 6)
    with pytest.raises(ValueError, match=r"exactly 1\.0|zero gradient"):
        response_surface(sim)


# --------------------------------------------------------------------------------------
# The optimiser
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("condition", "level"), ALL_CELLS, ids=lambda v: str(v))
def test_optimum_beats_the_status_quo_and_conserves_the_budget(
    condition: str, level: float | int | None
) -> None:
    """§3 requires the solution to be verified against the status quo, not assumed better."""
    sim = simulate(condition_params(condition, level), 6)
    surface = response_surface(sim)
    optimum = optimal_allocation(surface, seed=1234)

    assert optimum.total_sales > optimum.status_quo_sales
    assert optimum.achievable_lift > 0.0
    spent = float(np.dot(optimum.multipliers, surface.spend_totals))
    assert spent == pytest.approx(surface.budget, rel=1e-9)
    assert np.all(optimum.multipliers >= 0.0)
    assert np.all(optimum.multipliers <= MAX_MULTIPLIER + 1e-12)


@pytest.mark.parametrize(("condition", "level"), PLACEBO_CELLS, ids=lambda v: str(v))
def test_the_true_optimum_gives_the_placebo_exactly_no_budget(
    condition: str, level: float | int | None
) -> None:
    """A result in its own right: the true optimum spends nothing on the zero-effect channel.

    It is also the reference G6 is graded against — the correct placebo spend share is 0%, not
    merely small, so any share the model recommends is pure fabrication rather than rounding.

    Note the distinction from the exactly-zero assertions above. Contribution and mROAS are
    exactly 0.0 because they are algebraic cancellations. A multiplier is the output of a
    numerical optimiser, which lands on the bound to within solver precision — measured
    between 6.7e-17 and 2.7e-16 — so the honest assertion is on the budget share it implies,
    which is below one part in 10^18. The value is deliberately not clipped to zero in
    `optimal_allocation`: clipping would be cosmetic and would hide a genuinely non-zero one.
    """
    sim = simulate(condition_params(condition, level), 6)
    surface = response_surface(sim)
    optimum = optimal_allocation(surface, seed=1234)
    placebo = sim.channel_names.index("placebo")
    share = optimum.multipliers[placebo] * surface.spend_totals[placebo] / surface.budget
    assert share < 1e-15
    assert optimum.multipliers[placebo] < 1e-12


@pytest.mark.parametrize(("condition", "level"), [("C0", None), ("C4", None), ("C7", None)])
def test_the_returned_optimum_matches_a_large_reference_solve(
    condition: str, level: float | int | None
) -> None:
    """The replacement for "multi-start solutions agree": the *best* solution is what is stable.

    The default eight starts are checked against a 64-start reference from independent random
    draws. In a wider sweep — five representative cells at twenty seeds each, against a
    256-start reference — the structured-plus-screened design matched on 100 of 100 trials,
    where plain Dirichlet starts missed on 8%.
    """
    params = condition_params(condition, level)
    for seed in (0, 1, 2):
        surface = response_surface(simulate(params, seed))
        optimum = optimal_allocation(surface, seed=1234 + seed)
        reference = optimal_allocation(surface, seed=90_000 + seed, n_starts=64)
        assert optimum.total_sales == pytest.approx(reference.total_sales, rel=1e-6)


def test_d17_the_starts_are_recorded_as_disagreeing_rather_than_required_to_agree() -> None:
    """D17 deleted the agreement requirement; this asserts the disagreement stays *visible*.

    Deleting a failing test can mean two different things, and only one of them is honest.
    The requirement was wrong — S-shaped saturation makes the objective non-concave, so
    requiring starts to agree was requiring convexity, and if it held then one start would
    do. But the disagreement it was detecting is real: on C0 seed 6 the spread across starts
    is 0.105-0.156 on media contribution and only 3 to 7 of 8 starts reach the best.

    So `spread` and `n_agreeing` are still computed and still reported, and this pins them as
    live diagnostics rather than vestigial fields. If a future change made them constant, or
    made the surface concave, that is a finding rather than a silent improvement.
    """
    surface = response_surface(simulate(condition_params("C0"), 6))
    optimum = optimal_allocation(surface, seed=1234)
    assert optimum.spread > AGREEMENT_TOLERANCE
    assert 0 < optimum.n_agreeing < optimum.n_starts


def test_a_failed_solve_raises_rather_than_returning_a_number() -> None:
    """CLAUDE.md names this failure mode: a stalled solve reads downstream as low regret.

    The guard is checked by construction rather than by hoping SLSQP fails: `optimal_allocation`
    only ever appends a solution after `outcome.success`, and independently re-verifies the
    budget residual and the bounds, because a solver reporting success is a claim rather than
    evidence. Here the unscaled formulation — the one that genuinely does fail, with status 8,
    "positive directional derivative for linesearch" — is run directly to show the guard has
    something real to catch.
    """
    surface = response_surface(simulate(condition_params("C0"), 0))
    totals, budget = surface.spend_totals, surface.budget

    def unscaled_objective(multipliers: Vector, /) -> float:
        return -surface.media_total(multipliers)

    unscaled = minimize(
        unscaled_objective,
        surface.status_quo(),
        method="SLSQP",
        bounds=[(0.0, MAX_MULTIPLIER)] * surface.n_channels,
        constraints=LinearConstraint(totals, lb=budget, ub=budget),
        options={"maxiter": 300, "ftol": 1e-10},
    )
    assert not unscaled.success
    assert unscaled.status == 8

    # The scaled formulation the module actually uses converges from the same start.
    assert optimal_allocation(surface, seed=1234).total_sales > 0.0


def test_optimal_allocation_is_deterministic() -> None:
    surface = response_surface(simulate(condition_params("C7"), 2))
    first = optimal_allocation(surface, seed=77)
    second = optimal_allocation(surface, seed=77)
    assert np.array_equal(first.multipliers, second.multipliers)
    assert first.total_sales == second.total_sales
    assert first.spread == second.spread


def test_optimal_allocation_requires_enough_starts_to_seed_every_basin() -> None:
    surface = response_surface(simulate(condition_params("C0"), 0))
    with pytest.raises(ValueError, match="n_starts must be at least"):
        optimal_allocation(surface, seed=1, n_starts=3)


def test_status_quo_is_always_among_the_starts_so_the_lift_cannot_be_negative() -> None:
    """`achievable_lift` is regret's denominator; a negative one would invert every regret."""
    for condition, level in ALL_CELLS:
        surface = response_surface(simulate(condition_params(condition, level), 8))
        assert optimal_allocation(surface, seed=5).achievable_lift >= 0.0


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "multipliers",
    [
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0, 1.0, 1.0, -0.5]),
        np.array([1.0, 1.0, 1.0, 1.0, np.nan]),
        np.array([[1.0] * 5]),
    ],
    ids=["wrong-length", "negative", "nan", "two-dimensional"],
)
def test_surface_rejects_invalid_multipliers(multipliers: np.ndarray) -> None:
    surface = response_surface(simulate(condition_params("C0"), 0))
    with pytest.raises(ValueError):
        surface.total_sales(multipliers)


def test_surface_exposes_the_budget_and_channel_count() -> None:
    sim = simulate(condition_params("C5"), 0)
    surface = response_surface(sim)
    assert isinstance(surface, ResponseSurface)
    assert surface.n_channels == 6
    assert surface.budget == pytest.approx(float(sim.spend.sum()), rel=1e-12)
