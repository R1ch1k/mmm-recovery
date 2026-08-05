"""Tests for `RidgeMMM` (CLAUDE.md build order, Step 4).

One test here is `xfail(strict=True)`: CLAUDE.md asks that a noiseless C0 draw at T=520
recover contributions within 5%, and it does not. The measurement, the mechanism and the
refuted remedies are in the `reason` string. Following the repo's convention, the gap is
recorded rather than argued away, and it is a **K1 question** — C0 is the harness-validity
gate, so this one is not a threshold to be adjusted after the fact.

Seeds are fixed throughout and every figure quoted in a docstring was measured.
"""

import inspect

import numpy as np
import pytest
from scipy.optimize import minimize

from mmm_recovery.dgp import condition_params, simulate
from mmm_recovery.estimator import (
    BOOTSTRAP_BLOCK_WEEKS,
    CONTROL_COLUMN_NAMES,
    MAX_OBSERVED_CONDITION_NUMBER,
    N_CONTROL_COLUMNS,
    Hyperparameters,
    MMMFit,
    RidgeMMM,
    SearchBounds,
    _moving_block_indices,
    _normal_equations,
    _solve_bounded_ridge,
    adstock_spend,
    allocation_regret,
    bootstrap_contributions,
    control_matrix,
    design_column_names,
    design_matrix,
    estimated_marginal_roas,
    expanding_window_folds,
    recommended_allocation,
)
from mmm_recovery.truth import (
    incremental_contribution,
    optimal_allocation,
    response_surface,
)

FAST = RidgeMMM(n_draws=25, n_bootstrap=25)
"""A cheap estimator for the structural tests, which do not depend on the search budget."""


def fit_condition(condition: str, level: float | int | None = None, seed: int = 0) -> MMMFit:
    sim = simulate(condition_params(condition, level), seed)
    return FAST.fit(sim.spend, sim.sales, seed=seed)


# --------------------------------------------------------------------------------------
# The estimator never sees d_t — CLAUDE.md non-negotiable rule 3
# --------------------------------------------------------------------------------------


def test_the_control_block_is_a_function_of_the_week_index_and_nothing_else() -> None:
    """First structural guarantee: `control_matrix` takes only T, so no series can enter it.

    Asserted on the signature rather than by reading the body, because the body is what a
    future edit changes. A control block that took a dataset could absorb `d_t` legitimately-
    looking, and C3 and C7 would then measure nothing.
    """
    parameters = inspect.signature(control_matrix).parameters
    assert list(parameters) == ["n_weeks"]
    assert parameters["n_weeks"].annotation is int

    matrix = control_matrix(520)
    assert matrix.shape == (520, N_CONTROL_COLUMNS)
    assert np.array_equal(matrix, control_matrix(520))


def test_the_fit_entry_point_cannot_receive_a_latent_series() -> None:
    """Second structural guarantee: `fit` takes two arrays and a seed.

    `SimResult` carries `demand`, so an estimator that accepted one would have `d_t` in scope
    and rule 3 would rest on nobody ever writing the expression. It accepts arrays instead, so
    there is no expression to write.
    """
    parameters = inspect.signature(RidgeMMM.fit).parameters
    assert list(parameters) == ["self", "spend", "sales", "seed"]
    for name in ("demand", "params", "sim", "truth", "beta"):
        assert name not in parameters


def test_the_design_matrix_is_media_columns_plus_the_six_named_controls() -> None:
    """Third structural guarantee, and it runs on every fit rather than only under pytest."""
    sim = simulate(condition_params("C7"), 0)
    n_channels = sim.spend.shape[1]
    hyper = Hyperparameters(
        decay=np.full(n_channels, 0.5),
        hill_shape=np.full(n_channels, 1.0),
        half_saturation=np.full(n_channels, 20.0),
        ridge_penalty=1e-4,
    )
    matrix = design_matrix(sim.spend, hyper)
    assert matrix.shape == (sim.spend.shape[0], n_channels + N_CONTROL_COLUMNS)
    names = design_column_names(sim.channel_names)
    assert names == (*sim.channel_names, *CONTROL_COLUMN_NAMES)
    assert len(names) == matrix.shape[1]
    assert "demand" not in names


@pytest.mark.parametrize(("condition", "level"), [("C0", None), ("C3", 0.6), ("C7", None)])
def test_replacing_latent_demand_leaves_the_fit_bit_identical(
    condition: str, level: float | int | None
) -> None:
    """The empirical attack on rule 3, and the one that would catch a leak the others miss.

    The signature tests prove `d_t` is not in scope at the entry point. This proves it is not
    reaching the numbers by some other route: `d_t` is replaced with noise of a different
    scale, and every coefficient must come back bit-identical. C3 at φ=0.6 is included
    deliberately — it is the condition where spend is *correlated* with demand, so a leak
    would show up there most strongly and be most damaging.
    """
    sim = simulate(condition_params(condition, level), 4)
    before = FAST.fit(sim.spend, sim.sales, seed=4)

    corrupted = np.random.default_rng(999).normal(50.0, 10.0, size=sim.demand.shape)
    object.__setattr__(sim, "demand", corrupted)
    after = FAST.fit(sim.spend, sim.sales, seed=4)

    assert np.array_equal(before.coefficients, after.coefficients)
    assert np.array_equal(before.contribution, after.contribution)
    assert before.cv_rmse == after.cv_rmse


# --------------------------------------------------------------------------------------
# Determinism — CLAUDE.md non-negotiable rule 4
# --------------------------------------------------------------------------------------


def test_fitting_twice_gives_bit_identical_results() -> None:
    sim = simulate(condition_params("C0"), 2)
    first = FAST.fit(sim.spend, sim.sales, seed=7)
    second = FAST.fit(sim.spend, sim.sales, seed=7)
    assert np.array_equal(first.coefficients, second.coefficients)
    assert np.array_equal(first.contribution_interval, second.contribution_interval)
    assert first.hyperparameters.ridge_penalty == second.hyperparameters.ridge_penalty


def test_the_global_rng_cannot_move_a_fit() -> None:
    """Every stochastic step takes an explicit seed, so seeding numpy globally changes nothing.

    The legacy `np.random.seed` is used deliberately and NPY002 suppressed with it: the point
    is that the global state ruff is warning about has no effect here. Reaching for
    `default_rng` instead would test something the estimator never touches. This is what makes
    `make reproduce` byte-identical regardless of what else ran first in the process.
    """
    sim = simulate(condition_params("C0"), 2)
    np.random.seed(1)  # noqa: NPY002
    first = FAST.fit(sim.spend, sim.sales, seed=7)
    np.random.seed(12345)  # noqa: NPY002
    second = FAST.fit(sim.spend, sim.sales, seed=7)
    assert np.array_equal(first.coefficients, second.coefficients)


def test_changing_the_bootstrap_size_cannot_move_the_search_result() -> None:
    """The two streams are spawned independently, so B is free to change without re-selecting.

    If they shared a stream, raising B would silently change which hyperparameters were
    chosen, and a change to an uncertainty setting would move the point estimate.
    """
    sim = simulate(condition_params("C0"), 3)
    small = RidgeMMM(n_draws=25, n_bootstrap=10).fit(sim.spend, sim.sales, seed=3)
    large = RidgeMMM(n_draws=25, n_bootstrap=60).fit(sim.spend, sim.sales, seed=3)
    assert np.array_equal(small.coefficients, large.coefficients)
    assert small.cv_rmse == large.cv_rmse


# --------------------------------------------------------------------------------------
# The pieces
# --------------------------------------------------------------------------------------


def test_the_bounded_ridge_solve_matches_a_brute_force_optimiser() -> None:
    """The Cholesky-and-BVLS reduction must solve the objective it claims to.

    It is verified against a generic constrained optimiser on the *explicit* objective rather
    than by re-reading the algebra, because the reduction ``‖Rb − R'⁻¹g‖²`` is exactly the kind
    of step that stays plausible while being wrong by a transpose.
    """
    sim = simulate(condition_params("C0"), 1)
    n_media = sim.spend.shape[1]
    hyper = Hyperparameters(
        decay=np.full(n_media, 0.4),
        hill_shape=np.full(n_media, 1.3),
        half_saturation=np.full(n_media, 25.0),
        ridge_penalty=1e-3,
    )
    design = design_matrix(sim.spend, hyper)
    gram, moment = _normal_equations(design, sim.sales)
    solved = _solve_bounded_ridge(gram, moment, hyper.ridge_penalty, n_media)

    scale = float(np.trace(gram)) / gram.shape[0]
    tau = hyper.ridge_penalty * scale

    def objective(coefficients: np.ndarray) -> float:
        residual = sim.sales - design @ coefficients
        return float(residual @ residual + tau * coefficients[:n_media] @ coefficients[:n_media])

    bounds = [(0.0, None)] * n_media + [(None, None)] * N_CONTROL_COLUMNS
    reference = minimize(objective, solved, method="L-BFGS-B", bounds=bounds, tol=1e-14)
    assert objective(solved) <= objective(reference.x) * (1.0 + 1e-9)
    assert np.all(solved[:n_media] >= 0.0)


def test_the_normal_equations_stay_solvable_without_a_conditioning_jitter() -> None:
    """Why there is no jitter constant: the system is nowhere near singular, even at ρ=0.95.

    The worst case in the grid is the smallest drawable penalty against the most collinear
    spend. Measured across 4,800 solves spanning C0, C1[0.95], C2[104] and C7 — every draw, at
    both fold sizes — the Cholesky never failed and the condition number peaked at 3.1e5, some
    eleven orders of magnitude clear of float64. A jitter added "for safety" would have been
    doing statistical work on the unpenalised controls under a numerical name.
    """
    sim = simulate(condition_params("C1", 0.95), 0)
    n_media = sim.spend.shape[1]
    hyper = Hyperparameters(
        decay=np.full(n_media, 0.6),
        hill_shape=np.full(n_media, 1.5),
        half_saturation=np.full(n_media, 30.0),
        ridge_penalty=SearchBounds().ridge_penalty[0],
    )
    design = design_matrix(sim.spend, hyper)
    gram, moment = _normal_equations(design, sim.sales)
    size = gram.shape[0]
    penalties = np.zeros(size)
    penalties[:n_media] = hyper.ridge_penalty * float(np.trace(gram)) / size

    assert np.linalg.cond(gram + np.diag(penalties)) < MAX_OBSERVED_CONDITION_NUMBER
    solved = _solve_bounded_ridge(gram, moment, hyper.ridge_penalty, n_media)
    assert np.all(np.isfinite(solved))


def test_the_folds_expand_and_never_test_on_training_weeks() -> None:
    """Expanding window: training always precedes testing, and the window only ever grows."""
    for n_weeks in (104, 156, 260, 520):
        folds = expanding_window_folds(n_weeks)
        assert len(folds) == 3
        previous_end = 0
        for train_end, test_end in folds:
            assert train_end >= n_weeks // 2, "the initial window is half the series"
            # Each fold trains on everything before its split, so the window absorbs the
            # previous fold's test block exactly — equal, never less.
            assert train_end >= previous_end
            assert test_end > train_end, "a fold must test on weeks it did not train on"
            assert test_end <= n_weeks
            previous_end = test_end
        assert folds[0][0] < folds[-1][0], "the training window must expand"


def test_the_moving_block_bootstrap_resamples_weeks_not_channels() -> None:
    """CLAUDE.md names resampling the wrong axis as a specific failure mode.

    The indices must address weeks — length T, every value a valid week — and they must arrive
    in runs of `BOOTSTRAP_BLOCK_WEEKS` consecutive values, which is the whole point of a block
    bootstrap. An IID resample would satisfy the first check and fail the second.
    """
    rng = np.random.default_rng(0)
    indices = _moving_block_indices(520, BOOTSTRAP_BLOCK_WEEKS, rng)
    assert indices.shape == (520,)
    assert indices.min() >= 0 and indices.max() < 520

    steps = np.diff(indices)
    consecutive = int((steps == 1).sum())
    expected = 520 - (520 // BOOTSTRAP_BLOCK_WEEKS)
    assert consecutive >= expected - 1, "indices are not arriving in consecutive blocks"


def test_the_search_bounds_contain_every_true_value() -> None:
    """A search told where not to look would fail for a reason the study is not measuring.

    True λ spans 0.10-0.70 against a [0, 0.9] range; α spans 0.9-2.2 against [0.5, 3.0]; and
    κ/mean-spend spans 0.90 (search) to 2.08 (OOH) against [0.3, 3.0]. If a future channel
    table moved a value outside its range, every downstream failure would be an artefact.
    """
    bounds = SearchBounds()
    params = condition_params("C7")
    sim = simulate(params, 0)
    for index, channel in enumerate(params.channels):
        assert bounds.decay[0] <= channel.decay <= bounds.decay[1]
        assert bounds.hill_shape[0] <= channel.hill_shape <= bounds.hill_shape[1]
        mean_adstocked = float(
            adstock_spend(sim.spend, np.full(sim.spend.shape[1], channel.decay))[:, index].mean()
        )
        ratio = channel.half_saturation / mean_adstocked
        assert bounds.half_saturation_ratio[0] <= ratio <= bounds.half_saturation_ratio[1], (
            f"{channel.name}: κ ratio {ratio:.2f} is outside the search range"
        )


def test_bad_shapes_and_bad_bounds_raise() -> None:
    sim = simulate(condition_params("C0"), 0)
    with pytest.raises(ValueError, match="sales must be"):
        FAST.fit(sim.spend, sim.sales[:-1], seed=0)
    with pytest.raises(ValueError, match="spend must be"):
        FAST.fit(sim.spend[:, 0], sim.sales, seed=0)
    with pytest.raises(ValueError, match="bounds must be increasing"):
        SearchBounds(decay=(0.9, 0.1))
    with pytest.raises(ValueError, match="decay bounds"):
        SearchBounds(decay=(0.0, 1.0))
    with pytest.raises(ValueError, match="n_weeks"):
        control_matrix(20)


# --------------------------------------------------------------------------------------
# Estimates
# --------------------------------------------------------------------------------------


def test_media_coefficients_are_non_negative_and_contributions_follow() -> None:
    """§4 constrains media coefficients to be non-negative, so no channel can destroy sales."""
    for condition, level in (("C0", None), ("C1", 0.95), ("C7", None)):
        fit = fit_condition(condition, level)
        assert np.all(fit.surface.coefficients >= 0.0)
        assert np.all(fit.contribution >= 0.0)
        assert fit.contribution.sum() > 0.0


def test_the_estimated_contribution_is_an_intervention_not_a_coefficient() -> None:
    """Estimate and estimand must be the same quantity, or §6's bias measures nothing.

    Zeroing a channel on the fitted surface must remove exactly that channel's contribution
    and leave the others untouched — the same `do()` operation `truth.py` performs on the true
    surface, so the two are comparable term by term.
    """
    fit = fit_condition("C0")
    surface = fit.surface
    total = surface.media_total(surface.status_quo())
    for index in range(surface.n_channels):
        multipliers = surface.status_quo()
        multipliers[index] = 0.0
        removed = total - surface.media_total(multipliers)
        assert removed == pytest.approx(fit.contribution[index], rel=1e-12)
        others = surface.contributions(multipliers)
        assert others[:, index].max() == 0.0


def test_the_fitted_gradient_matches_a_central_difference() -> None:
    """A wrong analytic gradient converges to the wrong point while reporting success."""
    surface = fit_condition("C0").surface
    point = np.full(surface.n_channels, 1.2)
    analytic = surface.media_gradient(point)
    step = 1e-6
    for index in range(surface.n_channels):
        up, down = point.copy(), point.copy()
        up[index] += step
        down[index] -= step
        numeric = (surface.media_total(up) - surface.media_total(down)) / (2.0 * step)
        assert analytic[index] == pytest.approx(numeric, rel=1e-5)


def test_marginal_roas_is_positive_where_the_coefficient_is_positive() -> None:
    surface = fit_condition("C0").surface
    roas = estimated_marginal_roas(surface)
    assert roas.shape == (surface.n_channels,)
    assert np.all(roas >= 0.0)
    assert np.any(roas > 0.0)
    with pytest.raises(ValueError, match="bump must be positive"):
        estimated_marginal_roas(surface, bump=0.0)


# --------------------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------------------


def test_bootstrap_intervals_have_positive_width_and_bracket_the_estimate() -> None:
    fit = fit_condition("C0")
    assert np.all(fit.interval_width > 0.0)
    assert np.all(fit.contribution_interval[:, 0] <= fit.contribution_interval[:, 1])


def test_bootstrap_intervals_are_not_constant_across_conditions() -> None:
    """CLAUDE.md names motionless intervals as the symptom of a bootstrap on the wrong axis.

    C2 at T=104 has a fifth of C0's data, so its intervals must be materially wider relative
    to the estimate. An interval that did not move between those two would mean the resampling
    was not seeing the data at all.
    """
    clean = fit_condition("C0")
    short = fit_condition("C2", 104)

    def relative_width(fit: MMMFit) -> float:
        total = fit.contribution.sum()
        return float(fit.interval_width.sum() / total)

    assert relative_width(short) > relative_width(clean)
    assert not np.allclose(clean.interval_width, short.interval_width)


def test_the_bootstrap_docstring_states_the_fixed_hyperparameter_limitation() -> None:
    """§4 requires the limitation to be stated wherever coverage is computed.

    Pinned by a test rather than trusted to review: the caveat is the difference between an
    interval that is honestly too narrow and one that is silently too narrow, and it has to
    survive into the write-up. `metrics.py` computes coverage from these intervals.
    """
    doc = " ".join((bootstrap_contributions.__doc__ or "").split())
    assert "understates total uncertainty" in doc
    assert "hyperparameters are held fixed" in doc


# --------------------------------------------------------------------------------------
# D18 — one optimiser configuration, and the oracle-surface control
# --------------------------------------------------------------------------------------


def test_the_model_allocation_exposes_no_optimiser_knob_of_its_own() -> None:
    """D18, asserted structurally: the configuration cannot diverge because there is only one.

    `recommended_allocation` takes a surface and a seed. It has no `n_starts`, no
    `max_multiplier`, no tolerance — so the truth solve and the model solve cannot be given
    different settings without deleting this test first.
    """
    parameters = inspect.signature(recommended_allocation).parameters
    assert list(parameters) == ["surface", "seed"]
    for knob in ("n_starts", "max_multiplier", "tolerance", "screening_pool", "options"):
        assert knob not in parameters


@pytest.mark.parametrize(("condition", "level"), [("C0", None), ("C4", None), ("C7", None)])
def test_d18_the_oracle_surface_control_returns_zero_regret(
    condition: str, level: float | int | None
) -> None:
    """D18's control: run the *model's* allocation path on the **true** surface.

    Because regret compares two solves of the same non-concave problem, a difference in
    optimiser configuration would show up as regret that has nothing to do with estimation.
    Running the model's path on the true surface isolates exactly that: the result must be
    bit-identical to the truth solve, so the residual optimisation error is exactly zero and
    none of it can be hiding inside the estimator's regret.

    Measured: 0.0 at C0, C4 and C7, on the nose rather than to a tolerance.
    """
    sim = simulate(condition_params(condition, level), 3)
    surface = response_surface(sim)
    truth_optimum = optimal_allocation(surface, seed=1234)
    oracle = recommended_allocation(surface, seed=1234)

    assert oracle.total_sales == truth_optimum.total_sales
    assert np.array_equal(oracle.multipliers, truth_optimum.multipliers)
    assert allocation_regret(surface, oracle.multipliers, truth_optimum) == 0.0


def test_regret_is_not_clipped_when_the_advice_is_worse_than_doing_nothing() -> None:
    """CLAUDE.md: regret above 100% must survive. It is the most informative outcome there is.

    The status quo scores exactly 1.0 by construction — it is the denominator's own reference
    point — and an allocation worse than the status quo must therefore score above 1.0 rather
    than being quietly floored.
    """
    sim = simulate(condition_params("C0"), 0)
    surface = response_surface(sim)
    optimum = optimal_allocation(surface, seed=1234)

    assert allocation_regret(surface, surface.status_quo(), optimum) == pytest.approx(1.0)
    assert allocation_regret(surface, optimum.multipliers, optimum) == pytest.approx(0.0)

    starved = surface.status_quo()
    starved[0] = 0.0
    starved[1:] += float(surface.spend_totals[0]) / float(surface.spend_totals[1:].sum())
    assert allocation_regret(surface, starved, optimum) > 1.0


def test_the_model_allocation_conserves_the_budget_it_was_given() -> None:
    """The model reallocates; it does not get to spend more than the client has."""
    fit = fit_condition("C0")
    allocation = recommended_allocation(fit.surface, seed=99)
    spent = float(np.dot(allocation.multipliers, fit.surface.spend_totals))
    assert spent == pytest.approx(fit.surface.budget, rel=1e-9)
    assert np.all(allocation.multipliers >= 0.0)


# --------------------------------------------------------------------------------------
# Recovery — the K1 question
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_cv_score_prefers_the_true_hyperparameters_to_every_random_draw(seed: int) -> None:
    """The estimator's *selector* is sound: offered the truth, CV picks it. Measured, all seeds.

    This is what stops the xfail below being read as "the estimator is broken", and it is also
    what rules out the first explanation anyone reaches for. The true hyperparameters are
    injected into a pool of 200 random draws and rank **first** on out-of-sample RMSE on every
    seed tested. So the search bounds contain the truth, the transform is identified, the CV
    is not inverted, and the fit machinery works.

    An earlier version of this test asserted instead that the best of 200 draws *by actual
    contribution error* lands within 5%. That was selection on noise and is not asserted here:
    those low-error draws rank 57th to 157th of 200 on CV, so they are accidentally close
    rather than identified, and the 5% they reach is a property of taking a minimum over 200
    tries rather than of the estimator.

    What the xfail records is the harder fact that survives all of this: 0 of 7,500 random
    draws fit as well as the truth, and driving the fit past the truth with a real optimiser
    makes recovery worse rather than better.
    """
    from mmm_recovery.estimator import _cross_validated_rmse, _draw_hyperparameters

    params = condition_params("C0")
    sim = simulate(params, seed)
    n_media = sim.spend.shape[1]
    folds = expanding_window_folds(sim.spend.shape[0])
    floor = SearchBounds().ridge_penalty[0]

    def score(hyper: Hyperparameters) -> float:
        design = design_matrix(sim.spend, hyper)
        return _cross_validated_rmse(
            design, sim.noiseless_sales, hyper.ridge_penalty, n_media, folds
        )

    truth_score = score(
        Hyperparameters(
            decay=np.array([channel.decay for channel in params.channels]),
            hill_shape=np.array([channel.hill_shape for channel in params.channels]),
            half_saturation=np.array([channel.half_saturation for channel in params.channels]),
            ridge_penalty=floor,
        )
    )
    rng = np.random.default_rng(seed)
    drawn = [score(_draw_hyperparameters(sim.spend, SearchBounds(), rng)) for _ in range(200)]
    assert truth_score < min(drawn), (
        f"CV ranked a random draw above the truth (truth {truth_score:.4f}, "
        f"best draw {min(drawn):.4f}) — the selector, not the search, would then be at fault"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CLAUDE.md Step 4 asks that a noiseless C0 draw at T=520 recover contributions within "
        "5%. It does not, and this is a K1 harness-validity question rather than a tolerance "
        "to adjust. MEASURED on seeds 0-5, G1's own metric (median |relative bias| across "
        "channels, threshold 20%): 58.0% noiseless and 64.1% noisy. At the exactly true "
        "hyperparameters it is 3.7%, so the estimand IS recoverable and the failure is in "
        "choosing the transform. "
        "It is NOT the search, and this is the test that settles it: Nelder-Mead started "
        "EXACTLY AT THE TRUE HYPERPARAMETERS WALKS AWAY FROM THEM. CV improves 3.63160 -> "
        "3.06272 while median |bias| goes 2.4% -> 57.3% and max 3.7% -> 7133%. An optimiser "
        "handed the right answer for free prefers to leave it, so no amount of search power "
        "reaches the truth -- the truth is not what the objective wants. Differential "
        "evolution from scratch agrees: CV 3.07 against the truth's 3.63, recovery 57.1%, "
        "39.8%, 72.4% on seeds 0-2 against 13.8%, 40.8%, 54.4% for the 200-draw search. "
        "Nor is it the selector's FORM: in-sample RMSE (Spearman -0.312 with bias), an "
        "interpolating 5-fold CV (-0.278) and the implemented expanding-window CV (-0.294) "
        "are equally uninformative. "
        "MECHANISM, measured: the estimator trades the media LEVEL against the free "
        "unpenalised intercept, and no sum of squares can see the trade. True media averages "
        "243.9 GBPk/wk with a standard deviation of only 16.8, so a fit can match every "
        "wiggle and still be wrong about the level. At the selected draw the media error "
        "(-4.6 GBPk/wk) and the baseline error (+4.6) cancel to -0.00 in predicted sales -- "
        "exactly, not approximately. Held at truth on four channels and swept over TV's "
        "(alpha, kappa), 177 grid points sit within 1% of the truth's CV RMSE while TV's "
        "contribution ranges 43,938 to 240,522 GBPk (0.5% to 362.6% bias), all "
        "indistinguishable to the objective. That is 'attributable != incremental' stated as "
        "a measurement. "
        "This also settles K3's anti-strawman question in the estimator's favour: the "
        "200-draw random search is not a weakened MMM, it is an ACCIDENTALLY FLATTERING one, "
        "and replacing it with a competent optimiser (as Robyn does with Nevergrad) makes "
        "the measured failure larger. "
        "The failure is NOT confined to this noiseless build-order test. On the noisy C0 "
        "cell the grid actually runs, over 10 seeds, G1 fails at 64.0% against 20% and G2 "
        "coverage fails at 32.0% against 80%. Under noise CV RMSE is 29.1-32.3 against a "
        "noise sd of 29.32 GBPk/wk, so the selection criterion is over 99% noise variance. "
        "Other K3 remedies attempted and refuted: (1) tighter regularisation is monotonically "
        "WORSE -- median bias 13.8% at the CV-chosen penalty, 44.5% at 1e-2, 90.9% at 1e-1, "
        "99.0% at 1.0, because ridge shrinks the media block toward zero; non-negativity is "
        "already enforced. (3) a demand proxy is inapplicable at C0, where gamma=0 and phi=0 "
        "and there is no confounding to fix -- confirmed by the noiseless case failing just "
        "as hard. (2) shrinkage to an informative prior ROAS is untested and is a Step 7 "
        "item; rescuing the HARNESS gate with a prior would mean C0 only validates when the "
        "estimator is told the answer. "
        "SECONDARY, separate and much smaller: the DGP baseline is B0*(1+tau*t/T)*season, a "
        "product, while the controls are additive {1, t, cos, sin}, so the control block "
        "cannot represent it -- residual sd 2.45 GBPk, 0.25% of sales, driven to 1e-13 by "
        "adding trend*Fourier interactions, which also makes recovery at the true "
        "hyperparameters EXACT. It is not the cause: adding those interactions moves the "
        "measured bias only 58.0% -> 50.3% noiseless and 64.1% -> 64.0% noisy. "
        "Both are decisions for the pre-registration's author, not for this test."
    ),
)
def test_a_noiseless_c0_draw_recovers_contributions_within_five_percent() -> None:
    sim = simulate(condition_params("C0"), 0)
    true = incremental_contribution(response_surface(sim))
    fit = RidgeMMM().fit(sim.spend, sim.noiseless_sales, seed=0)
    relative = np.abs((fit.contribution - true) / true)
    assert float(np.median(relative)) <= 0.05
