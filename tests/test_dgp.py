"""Tests for the generating process (CLAUDE.md build order, Step 2).

Two of these are `xfail(strict=True)` rather than assertions. That is deliberate: both
record a gap between PREREGISTRATION.md and what its own numbers produce, measured rather
than argued. A strict xfail keeps the gap visible in every test run and turns it into an
error the moment it silently closes, which a comment or a loosened tolerance would not.

Seeds 0-29 are used throughout. Every figure quoted in a docstring was measured on that
range, so a change in the generator moves the numbers and the tests notice.
"""

import numpy as np
import pytest

from mmm_recovery.dgp import (
    CONDITION_LEVELS,
    PLACEBO,
    REAL_CHANNELS,
    DGPParams,
    condition_params,
    evaluate,
    mean_pairwise_spend_correlation,
    media_share,
    simulate,
)

SEEDS = range(30)
ALL_CELLS = [(cond, level) for cond, levels in CONDITION_LEVELS.items() for level in levels]

# C7 asks for a placebo coupling of 0.6 that the draw cannot always deliver; see
# test_c7_placebo_coupling_is_infeasible_on_some_seeds for the measurement.
C7_INFEASIBLE_SEED = 71


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("condition", "level"), ALL_CELLS, ids=lambda v: str(v))
def test_simulate_is_deterministic(condition: str, level: float | int | None) -> None:
    """Same params, same seed, same bits — including the solved correlation weights."""
    params = condition_params(condition, level)
    first, second = simulate(params, 11), simulate(params, 11)
    for name in ("spend", "sales", "noiseless_sales", "baseline", "contributions", "demand"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.mix_weight == second.mix_weight
    assert first.placebo_weight == second.placebo_weight


def test_simulate_is_order_independent() -> None:
    """Interleaving draws changes nothing — the proxy for worker-count independence.

    CLAUDE.md rule 4 requires the grid to be byte-identical regardless of how work is
    scheduled. Each cell owns its seed and its own Generator, so simulating out of order,
    or between other simulations, must not perturb a result.
    """
    params = condition_params("C0")
    sequential = [simulate(params, seed).sales for seed in (3, 4, 5)]
    interleaved = []
    for seed in (5, 3, 4):
        simulate(condition_params("C4"), 99)
        interleaved.append((seed, simulate(params, seed).sales))
    by_seed = dict(interleaved)
    for index, seed in enumerate((3, 4, 5)):
        np.testing.assert_array_equal(sequential[index], by_seed[seed])


def test_different_seeds_give_different_data() -> None:
    params = condition_params("C0")
    assert not np.array_equal(simulate(params, 1).sales, simulate(params, 2).sales)


def test_evaluate_consumes_no_randomness() -> None:
    """`evaluate` is the interventional workhorse; it must be a pure function of its inputs."""
    params = condition_params("C0")
    result = simulate(params, 6)
    first = evaluate(params, result.spend, result.demand, result.season)
    second = evaluate(params, result.spend, result.demand, result.season)
    np.testing.assert_array_equal(first.noiseless_sales, second.noiseless_sales)
    np.testing.assert_array_equal(first.noiseless_sales, result.noiseless_sales)


# --------------------------------------------------------------------------------------
# The D3 invariant — zero spend means exactly zero contribution
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("condition", "level"), ALL_CELLS, ids=lambda v: str(v))
def test_zeroing_all_spend_reproduces_the_baseline_exactly(
    condition: str, level: float | int | None
) -> None:
    """The D3 invariant, end to end and bit for bit — not to a tolerance.

    Under the misspecified conditions (C4, C7) this is the whole reason D3 zero-anchored the
    logistic. As literally specified in §2 the logistic returned about 1.8% of β at zero
    spend, so this test would have failed by roughly £10k/week of phantom sales while every
    approximate comparison passed.
    """
    params = condition_params(condition, level)
    seed = 12 if condition != "C7" else 13
    result = simulate(params, seed)
    response = evaluate(params, np.zeros_like(result.spend), result.demand, result.season)
    assert np.array_equal(response.noiseless_sales, response.baseline)
    assert np.array_equal(response.contributions, np.zeros_like(response.contributions))


@pytest.mark.parametrize("condition", ["C5", "C6", "C7"])
def test_zeroing_the_placebo_leaves_noiseless_sales_bit_identical(condition: str) -> None:
    """β = 0 means exactly zero, not nearly zero.

    CLAUDE.md calls this the single most important test in the repo. Step 3 will assert it
    again through the intervention API; here it is asserted about the generating process
    itself, so a later failure can be localised to `truth.py` rather than to the DGP.
    """
    level = CONDITION_LEVELS[condition][0]
    params = condition_params(condition, level)
    result = simulate(params, 14)
    placebo_column = result.channel_names.index("placebo")

    assert np.array_equal(
        result.contributions[:, placebo_column],
        np.zeros(params.n_weeks),
    )
    zeroed = result.spend.copy()
    zeroed[:, placebo_column] = 0.0
    response = evaluate(params, zeroed, result.demand, result.season)
    assert np.array_equal(response.noiseless_sales, result.noiseless_sales)


# --------------------------------------------------------------------------------------
# Media share — 25% is a C0 property (D4), and it is not met
# --------------------------------------------------------------------------------------


def test_c0_media_share_is_the_value_the_prereg_numbers_actually_produce() -> None:
    """Records the realised C0 media share: 18.57% (seeds 0-29, sd 0.09%).

    Not 25%. B0 = 1000, τ = 0.15 and the §2 channel table are jointly over-determined with
    the "≈25% of total sales" calibration claim, and they disagree. See the companion xfail
    below, and note that at mean spend the five channels sit at 46%, 48%, 52%, 45% and 17%
    of their β, so no combination of adstock or Jensen effects closes the gap.
    """
    shares = [media_share(simulate(condition_params("C0"), seed)) for seed in SEEDS]
    assert float(np.mean(shares)) == pytest.approx(0.1857, abs=0.002)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PREREGISTRATION.md §2 claims media contribution is calibrated to ~25% of total "
        "sales, but its own B0, tau and channel table give 18.57%. D4 fixes beta, so the "
        "only lever that does not touch ground truth is B0, which would have to be 680.8 "
        "rather than 1000. Pending a decision; recorded here so the gap cannot be forgotten."
    ),
)
def test_c0_media_share_meets_the_prereg_25_percent_target() -> None:
    shares = [media_share(simulate(condition_params("C0"), seed)) for seed in SEEDS]
    assert float(np.mean(shares)) == pytest.approx(0.25, abs=0.02)


@pytest.mark.parametrize(("condition", "level"), ALL_CELLS, ids=lambda v: str(v))
def test_media_share_is_measured_not_asserted_away_from_c0(
    condition: str, level: float | int | None
) -> None:
    """D4: the share drifts everywhere except C0, and the drift is described, not corrected.

    Measured means over seeds 0-29: C0 18.57%, C1 18.57-18.58%, C2 18.35-18.53%,
    C3 18.43-18.47%, C4 17.17%, C5/C6 18.56%, C7 17.02%. The two misspecified conditions
    are the movers; the confounded ones barely budge, which is the opposite of the direction
    D4 anticipated for C3 and is reported rather than adjusted.

    Only a wide sanity band is asserted here. A tight assertion would be a second, unlogged
    calibration target smuggled in under a test.
    """
    params = condition_params(condition, level)
    shares = [
        media_share(simulate(params, seed))
        for seed in SEEDS
        if not (condition == "C7" and seed == C7_INFEASIBLE_SEED)
    ]
    assert 0.10 < float(np.mean(shares)) < 0.30
    assert all(0.0 < share < 1.0 for share in shares)


# --------------------------------------------------------------------------------------
# D5 — collinearity on spend levels
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("condition", "level"), [("C1", 0.5), ("C1", 0.8), ("C1", 0.95)])
def test_d5_mean_pairwise_correlation_is_within_tolerance(condition: str, level: float) -> None:
    """D5: mean pairwise Pearson ρ on levels within ±0.02 of target, every C1 level."""
    params = condition_params(condition, level)
    for seed in SEEDS:
        realised = mean_pairwise_spend_correlation(simulate(params, seed))
        assert abs(realised - level) <= 0.02


def test_d5_mean_pairwise_correlation_holds_for_the_c7_composite() -> None:
    """The C7 case of D5, where ρ = 0.7 sits alongside φ = 0.6 and T = 104."""
    params = condition_params("C7")
    for seed in SEEDS:
        realised = mean_pairwise_spend_correlation(simulate(params, seed))
        assert abs(realised - 0.7) <= 0.02


@pytest.mark.parametrize(("condition", "level"), [("C1", 0.5), ("C1", 0.8), ("C1", 0.95)])
def test_d5_no_individual_pair_strays_beyond_ten_points(condition: str, level: float) -> None:
    """The per-pair half of D5. Holds at every C1 level, but only just at ρ=0.5.

    Worst deviations on seeds 0-29: 0.0856 at ρ=0.5, 0.0421 at ρ=0.8, 0.0268 at ρ=0.95.
    The ρ=0.5 figure has the least headroom; over 200 seeds it reaches 0.1018, so 0.1% of
    pairs already breach at T=520. The tolerance is comfortable here, but not by much.
    """
    params = condition_params(condition, level)
    for seed in SEEDS:
        result = simulate(params, seed)
        columns = [i for i, n in enumerate(result.channel_names) if n != "placebo"]
        matrix = np.corrcoef(result.spend[:, columns], rowvar=False)
        pairs = matrix[np.triu_indices(len(columns), k=1)]
        assert float(np.abs(pairs - level).max()) <= 0.10


@pytest.mark.xfail(
    strict=True,
    reason=(
        "D5's per-pair tolerance of +/-0.10 is tighter than sampling noise permits at "
        "C7's T=104. Worst pair deviation is 0.1251 on seeds 0-29 and 0.1648 over 200, "
        "with 2.08% of pairs breaching. A matched control with rho=0.7 and T=104 but "
        "none of C7's other knobs reaches 0.1495 and breaches on 1.70%, so this is the "
        "short series, not the composite. Theoretical SE of a single correlation at "
        "T=104 is 0.0507, making +/-0.10 a two-sigma bound that roughly 2% of pairs must "
        "exceed. Threshold, not bug — pending a decision."
    ),
)
def test_d5_per_pair_tolerance_holds_for_the_c7_composite() -> None:
    params = condition_params("C7")
    for seed in SEEDS:
        result = simulate(params, seed)
        columns = [i for i, n in enumerate(result.channel_names) if n != "placebo"]
        matrix = np.corrcoef(result.spend[:, columns], rowvar=False)
        pairs = matrix[np.triu_indices(len(columns), k=1)]
        assert float(np.abs(pairs - 0.7).max()) <= 0.10


def test_c0_spend_is_effectively_uncorrelated() -> None:
    """§5 describes C0 as ρ≈0, and it is — but not to D5's ±0.02.

    Each channel carries its own quarterly phase, so the shared budget factor is the only
    route to correlation and C0 switches it off. What remains is sampling noise: over 200
    seeds the mean pairwise ρ has mean −0.005 and sd 0.013, reaching 0.038 at worst.

    D5's ±0.02 is a tolerance on a *solved* target. C0 solves for nothing, so the right
    check here is against sampling noise, and ±0.06 is roughly 4.5 sd of the measured
    distribution. Asserting ±0.02 here would be borrowing a tolerance from a different
    question and would fail on seed 8.
    """
    realised = [
        mean_pairwise_spend_correlation(simulate(condition_params("C0"), seed)) for seed in SEEDS
    ]
    assert max(abs(value) for value in realised) <= 0.06
    assert abs(float(np.mean(realised))) <= 0.02


def test_collinearity_target_above_the_achievable_range_raises() -> None:
    """No silent clipping: an impossible ρ is an error, not a quietly different dataset."""
    with pytest.raises(ValueError, match="collinearity"):
        simulate(DGPParams(collinearity=0.999), 0)


# --------------------------------------------------------------------------------------
# D1 — placebo coupling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", [0.3, 0.45, 0.6])
def test_d1_placebo_couples_equally_to_search_and_season(level: float) -> None:
    """D1's targets, achieved at all three C6 levels.

    The solver pins the *mean* of the two correlations to the target exactly; the two then
    straddle it. Worst deviation of either individually on seeds 0-29: 0.062 at 0.3, 0.054
    at 0.45, 0.046 at 0.6. D1 fixes the target but no tolerance, so the ±0.08 asserted here
    is a choice, recorded in the Step 2 assumptions.
    """
    params = condition_params("C6", level)
    for seed in SEEDS:
        result = simulate(params, seed)
        placebo = result.channel_names.index("placebo")
        search = result.channel_names.index("search")
        to_search = float(np.corrcoef(result.spend[:, placebo], result.spend[:, search])[0, 1])
        to_season = float(np.corrcoef(result.spend[:, placebo], result.season)[0, 1])
        assert abs(0.5 * (to_search + to_season) - level) <= 1e-6
        assert abs(to_search - level) <= 0.08
        assert abs(to_season - level) <= 0.08


@pytest.mark.parametrize("level", [0.3, 0.45, 0.6])
def test_d1_targets_are_jointly_feasible_at_every_level(level: float) -> None:
    """The check that killed the original 0.8 specification, now run at the D1 levels.

    A *realised* sample correlation matrix is positive semi-definite by construction, so
    testing that would prove nothing. What is tested instead is the matrix of the D1
    *targets* combined with the search-to-season correlation actually present in the data —
    which is exactly the quantity that made two 0.8 targets impossible.
    """
    params = condition_params("C6", level)
    for seed in SEEDS:
        result = simulate(params, seed)
        search = result.channel_names.index("search")
        search_to_season = float(np.corrcoef(result.spend[:, search], result.season)[0, 1])
        target_matrix = np.array(
            [
                [1.0, level, level],
                [level, 1.0, search_to_season],
                [level, search_to_season, 1.0],
            ]
        )
        assert float(np.linalg.eigvalsh(target_matrix).min()) >= -1e-12


def test_c5_placebo_spend_is_independent() -> None:
    """C5 is 'spend independent', which is a weight of zero rather than a solved target."""
    for seed in SEEDS:
        result = simulate(condition_params("C5"), seed)
        placebo = result.channel_names.index("placebo")
        search = result.channel_names.index("search")
        assert result.placebo_weight == 0.0
        correlation = float(np.corrcoef(result.spend[:, placebo], result.spend[:, search])[0, 1])
        assert abs(correlation) < 0.2


def test_c7_placebo_coupling_is_infeasible_on_some_seeds() -> None:
    """C7 cannot always deliver its D1 coupling of 0.6, and says so instead of pretending.

    Measured: `simulate` raises on 9 of the first 500 seeds — [71, 136, 156, 221, 225, 367,
    418, 438, 450], a rate of 1.8%. C7 is specified at 500 seeds, so the grid would stop
    roughly nine times. The achievable ceiling across 300 seeds averages 0.704 with sd 0.047
    and a minimum of 0.567; every target up to 0.55 is reachable on all of them. C6 is
    unaffected — its ceiling never drops below 0.629 — so this is specific to C7, where
    φ=0.6 dilutes the placebo's seasonal correlation and T=104 makes the ceiling volatile.

    This test pins the diagnosis. It is expected to be deleted once the C7 coupling level is
    decided, and it will fail loudly if the generator starts silently coping instead.
    """
    with pytest.raises(ValueError, match="placebo coupling"):
        simulate(condition_params("C7"), C7_INFEASIBLE_SEED)


# --------------------------------------------------------------------------------------
# Endogeneity, latent demand, seasonality
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", [0.3, 0.6])
def test_positive_endogeneity_correlates_spend_with_latent_demand(level: float) -> None:
    """φ > 0 is the confounding channel: the marketer spends more when demand is high.

    Measured mean correlations across channels: about +0.25 at φ=0.3 and +0.45 at φ=0.6.
    """
    params = condition_params("C3", level)
    for seed in SEEDS:
        result = simulate(params, seed)
        for column in range(result.spend.shape[1]):
            assert float(np.corrcoef(result.spend[:, column], result.demand)[0, 1]) > 0.1


def test_zero_endogeneity_leaves_spend_uncorrelated_with_demand() -> None:
    """The C0 control for the test above: without φ, spend knows nothing about demand."""
    for seed in SEEDS:
        result = simulate(condition_params("C0"), seed)
        for column in range(result.spend.shape[1]):
            correlation = float(np.corrcoef(result.spend[:, column], result.demand)[0, 1])
            assert abs(correlation) < 0.2


def test_latent_demand_has_the_requested_scale_and_persistence() -> None:
    """d_t is standardised in sample, so γ and φ have an interpretable magnitude."""
    params = condition_params("C0")
    for seed in SEEDS:
        demand = simulate(params, seed).demand
        assert float(demand.std()) == pytest.approx(params.demand_sd, rel=1e-9)
        assert float(demand.mean()) == pytest.approx(0.0, abs=1e-12)
        lag_one = float(np.corrcoef(demand[1:], demand[:-1])[0, 1])
        assert 0.5 < lag_one < 0.98


def test_gamma_moves_the_baseline_and_nothing_else_does() -> None:
    """γ=0 leaves latent demand entirely inert, which is what makes C0 a clean control."""
    result = simulate(condition_params("C0"), 8)
    params = result.params
    weeks = np.arange(params.n_weeks, dtype=np.float64)
    expected = params.baseline_level * (1.0 + params.trend * weeks / params.n_weeks) * result.season
    np.testing.assert_allclose(result.baseline, expected, rtol=1e-12)

    confounded = simulate(condition_params("C3", 0.6), 8)
    assert float(np.corrcoef(confounded.baseline, confounded.demand)[0, 1]) > 0.5


def test_seasonality_is_positive_and_annual() -> None:
    result = simulate(condition_params("C0"), 0)
    assert np.all(result.season > 0.0)
    np.testing.assert_allclose(result.season[:52], result.season[52:104], rtol=1e-12)


# --------------------------------------------------------------------------------------
# Structure, units and validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("condition", "level"), ALL_CELLS, ids=lambda v: str(v))
def test_shapes_units_and_positivity(condition: str, level: float | int | None) -> None:
    params = condition_params(condition, level)
    result = simulate(params, 15)
    n_channels = len(params.channels)
    assert result.spend.shape == (params.n_weeks, n_channels)
    assert result.contributions.shape == (params.n_weeks, n_channels)
    for series in (result.sales, result.noiseless_sales, result.baseline, result.demand):
        assert series.shape == (params.n_weeks,)
    assert np.all(result.spend > 0.0)
    assert np.all(result.baseline > 0.0)
    assert np.all(result.contributions >= 0.0)
    assert np.all(result.sales > 0.0)


def test_mean_spend_lands_near_the_prereg_targets() -> None:
    """The §2 table's mean weekly spends, recovered from the log-normal draw."""
    result = simulate(condition_params("C0"), 0)
    for index, channel in enumerate(REAL_CHANNELS):
        assert float(result.spend[:, index].mean()) == pytest.approx(channel.mean_spend, rel=0.06)


def test_noise_is_three_percent_of_mean_sales() -> None:
    params = condition_params("C0")
    result = simulate(params, 0)
    residual = result.sales - result.noiseless_sales
    expected = params.noise_fraction * float(result.noiseless_sales.mean())
    assert float(residual.std()) == pytest.approx(expected, rel=0.1)


def test_conditions_move_one_knob_each_relative_to_c0() -> None:
    """§5: every condition is C0 with exactly one knob moved, except C7."""
    baseline = condition_params("C0")
    single_knob = {
        ("C1", 0.5): {"collinearity"},
        ("C2", 104): {"n_weeks"},
        ("C3", 0.6): {"demand_coefficient", "endogeneity"},  # §5 moves these as one knob
        ("C4", None): {"misspecified"},
        ("C5", None): {"channels", "placebo_coupling"},  # adding a channel needs both
        ("C6", 0.6): {"channels", "placebo_coupling"},
    }
    for (condition, level), allowed in single_knob.items():
        params = condition_params(condition, level)
        changed = {
            name
            for name in vars(baseline)
            if not np.array_equal(getattr(baseline, name), getattr(params, name))
        }
        assert changed == allowed, f"{condition} changed {changed}, expected {allowed}"


def test_placebo_beta_is_exactly_zero() -> None:
    assert PLACEBO.beta == 0.0
    assert all(channel.beta > 0.0 for channel in REAL_CHANNELS)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_weeks": 12}, "seasonal cycle"),
        ({"demand_ar": 1.0}, "demand_ar"),
        ({"demand_sd": 0.0}, "demand_sd"),
        ({"baseline_level": 0.0}, "baseline_level"),
        ({"spend_log_sd": -1.0}, "spend_log_sd"),
        ({"collinearity": 1.0}, "collinearity"),
        ({"placebo_coupling": 0.8}, "feasibility ceiling"),
        ({"season_coefficients": (2.0, 0.0, 0.0, 0.0)}, "non-positive"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DGPParams(**kwargs)  # type: ignore[arg-type]


def test_placebo_coupling_without_a_placebo_channel_is_rejected() -> None:
    with pytest.raises(ValueError, match="no channel named"):
        DGPParams(placebo_coupling=0.3)


@pytest.mark.parametrize(
    ("condition", "level"),
    [("C9", None), ("C1", 0.6), ("C0", 0.5), ("C6", 0.8)],
)
def test_unknown_conditions_and_levels_are_rejected(condition: str, level: float | None) -> None:
    with pytest.raises(ValueError):
        condition_params(condition, level)
