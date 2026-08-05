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
    BASELINE_LEVEL,
    C0_MEDIA_SHARE_TARGET,
    CONDITION_LEVELS,
    PLACEBO,
    REAL_CHANNELS,
    DGPParams,
    condition_params,
    evaluate,
    mean_pairwise_spend_correlation,
    media_share,
    per_pair_correlation_bound,
    simulate,
    solve_baseline_level,
)

SEEDS = range(30)
ALL_CELLS = [(cond, level) for cond, levels in CONDITION_LEVELS.items() for level in levels]


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


def test_c0_media_share_meets_the_prereg_25_percent_target() -> None:
    """D9: C0's realised media share is 25.0% within ±0.05pp, in and out of sample.

    Asserted over seeds 0-199 — the count C0 actually runs, and the set B0 was solved on —
    and again over seeds 200-399, which the solve never saw. The out-of-sample figure lands
    at 24.9951%, so the calibration is a property of the generator rather than of the
    particular draws it was fitted to.
    """
    params = condition_params("C0")
    assert params.baseline_level == BASELINE_LEVEL

    in_sample = float(np.mean([media_share(simulate(params, s)) for s in range(200)]))
    out_of_sample = float(np.mean([media_share(simulate(params, s)) for s in range(200, 400)]))
    assert abs(in_sample - C0_MEDIA_SHARE_TARGET) <= 0.0005
    assert abs(out_of_sample - C0_MEDIA_SHARE_TARGET) <= 0.0005


def test_the_committed_baseline_level_reproduces_the_solve() -> None:
    """D9 asked for a numerical solve, not a hard-coded constant. This is the difference.

    `BASELINE_LEVEL` is committed so `simulate` stays fast and deterministic, but it has to be
    the number the solver actually returns. Re-running the solve here is what makes it derived
    rather than asserted — and it fails if any change to the DGP moves the calibration.
    """
    assert solve_baseline_level() == pytest.approx(BASELINE_LEVEL, abs=1e-3)


def test_baseline_is_exactly_linear_in_b0_and_contributions_are_invariant() -> None:
    """Why B0 is the safe lever, and the resolution of the 680.8-versus-684.0 discrepancy.

    Baseline scales bit-exactly with B0 while spend and contributions do not move at all, so
    B0 changes the media *ratio* without touching any ground truth. Nothing in the calibration
    is non-linear in B0; the two closed-form answers differed only in which draws they used —
    seed 0 alone has a share of 18.496% and gives 680.8, the 30-seed mean is 18.568% and gives
    684.1. Since a mean of ratios is not a ratio of means, neither is the mean's root, which
    is why D9 asked for the solve.
    """
    full = simulate(DGPParams(baseline_level=1000.0), 0)
    half = simulate(DGPParams(baseline_level=500.0), 0)
    assert np.array_equal(full.spend, half.spend)
    assert np.array_equal(full.contributions, half.contributions)
    assert np.array_equal(half.baseline, full.baseline * 0.5)


@pytest.mark.parametrize(("condition", "level"), ALL_CELLS, ids=lambda v: str(v))
def test_media_share_is_measured_not_asserted_away_from_c0(
    condition: str, level: float | int | None
) -> None:
    """D4: the share drifts everywhere except C0, and the drift is described, not corrected.

    Measured means over seeds 0-29 after D9: C0 25.04%, C1 25.04-25.06%, C2 24.77-24.99%,
    C3 24.87-24.92%, C4 23.29%, C5/C6 25.03%, C7 23.10%. The two misspecified conditions are
    the only real movers, and C3 still drifts down rather than up (D13).

    B0 changes none of this underneath. Contributions do not depend on B0 at all, so the
    ratio of a condition's media contribution to C0's is invariant: C4/C0 was 0.9089 before
    D9 and is 0.9090 after; C7/C0 was 0.8992 and is 0.8992. Raising the share from 18.6% to
    25% simply expresses the same shortfall in more percentage points.

    Only a wide sanity band is asserted here. A tight assertion would be a second, unlogged
    calibration target smuggled in under a test.
    """
    params = condition_params(condition, level)
    shares = [media_share(simulate(params, seed)) for seed in SEEDS]
    assert 0.15 < float(np.mean(shares)) < 0.35
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


def worst_pair_deviation(params: DGPParams, target: float, seeds: range) -> float:
    """Largest absolute deviation of any pairwise spend correlation from its target."""
    worst = 0.0
    for seed in seeds:
        result = simulate(params, seed)
        columns = [i for i, n in enumerate(result.channel_names) if n != "placebo"]
        matrix = np.corrcoef(result.spend[:, columns], rowvar=False)
        pairs = matrix[np.triu_indices(len(columns), k=1)]
        worst = max(worst, float(np.abs(pairs - target).max()))
    return worst


@pytest.mark.parametrize(
    ("condition", "level", "target"),
    [("C1", 0.5, 0.5), ("C1", 0.8, 0.8), ("C7", None, 0.7)],
)
def test_d11_per_pair_bound_holds(condition: str, level: float | None, target: float) -> None:
    """D11's bound, ``4·(1-ρ²)/√(T-3)``. Holds at ρ=0.5, ρ=0.8 and the C7 composite.

    Worst deviation against bound on seeds 0-29: 0.0856 vs 0.1319 at ρ=0.5, 0.0421 vs 0.0633
    at ρ=0.8, 0.1251 vs 0.2030 at C7. The C7 case is what D11 was written to fix, and it now
    passes with margin where the old fixed ±0.10 breached on 2.08% of pairs.
    """
    params = condition_params(condition, level)
    bound = per_pair_correlation_bound(target, params.n_weeks)
    assert worst_pair_deviation(params, target, SEEDS) <= bound


@pytest.mark.xfail(
    strict=True,
    reason=(
        "D11's bound is a pure SAMPLING bound, but the realised per-pair spread also has a "
        "SYSTEMATIC component of 0.017-0.021 that does not shrink with more seeds. At "
        "rho=0.95 the sampling term collapses to 0.0172 while the systematic term does not, "
        "so the bound is breached (worst 0.0281) at this level and nowhere else. Cause is "
        "identified, not suspected: the pair means are spread 0.0327 against a seed-to-seed "
        "sd of 0.0030, and their ordering is exactly monotone in the per-channel quarterly "
        "phase gap (2wk -> 0.965, 4wk -> 0.946, 6wk -> 0.933). Setting quarterly_amplitude "
        "to 0.03 makes it pass. Recommended fix is a bound of systematic + 4*SE, with the "
        "systematic allowance measured at 0.025 -- pending a decision."
    ),
)
def test_d11_per_pair_bound_holds_at_the_highest_collinearity() -> None:
    params = condition_params("C1", 0.95)
    bound = per_pair_correlation_bound(0.95, params.n_weeks)
    assert worst_pair_deviation(params, 0.95, SEEDS) <= bound


def test_d11_bound_scales_with_both_series_length_and_target() -> None:
    """The bound is one formula with no magic numbers; these are the values it produces."""
    assert per_pair_correlation_bound(0.7, 104) == pytest.approx(0.2030, abs=1e-4)
    assert per_pair_correlation_bound(0.5, 520) == pytest.approx(0.1319, abs=1e-4)
    assert per_pair_correlation_bound(0.95, 520) == pytest.approx(0.0172, abs=1e-4)
    # Shorter series and weaker targets both loosen it; the ordering must never invert.
    assert per_pair_correlation_bound(0.5, 104) > per_pair_correlation_bound(0.5, 520)
    assert per_pair_correlation_bound(0.5, 520) > per_pair_correlation_bound(0.95, 520)
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="rho_target"):
            per_pair_correlation_bound(bad, 520)
    with pytest.raises(ValueError, match="n_weeks"):
        per_pair_correlation_bound(0.5, 3)


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


def test_c7_constructs_on_every_seed_it_will_actually_use() -> None:
    """D10: C7's coupling of 0.45 is reachable on all 500 seeds C7 runs at.

    The old level of 0.6 raised on 9 of those 500 — seeds 71, 136, 156, 221, 225, 367, 418,
    438 and 450 — because φ=0.6 dilutes the placebo's seasonal correlation and T=104 makes
    the achievable ceiling volatile (mean 0.704, sd 0.047, min 0.567 across 300 seeds).
    Measured failure counts at the four candidate levels: 0.45 → 0, 0.5 → 0, 0.55 → 1,
    0.6 → 9. This runs the full 500, not a sample, because a 1.8% failure rate is exactly the
    kind that a 30-seed test misses.
    """
    params = condition_params("C7")
    assert params.placebo_coupling == 0.45
    for seed in range(500):
        simulate(params, seed)


def test_c7_placebo_coupling_matches_a_c6_level_so_it_decomposes() -> None:
    """D10's second reason: an interaction is only isolable when each knob has a solo level."""
    assert condition_params("C7").placebo_coupling in CONDITION_LEVELS["C6"]


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
