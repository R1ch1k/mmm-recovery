"""Ground truth by intervention on the generating process — never by reading a parameter.

PREREGISTRATION.md §3 is the load-bearing design decision of the whole study, and everything
here implements it literally:

* **incremental contribution** of channel c is ``sales(spend) - sales(spend | spend_c := 0)``,
  evaluated noiselessly with every other input and the same seed held fixed. It is `do()`,
  written as a function call.
* **mROAS** of channel c is ``[sales(1.1·spend_c) - sales(spend_c)] / (0.1 · Σ_t spend_c,t)``,
  noiseless.
* **optimal allocation** is the per-channel multipliers maximising noiseless sales subject to
  the budget being conserved, with ``m_c ∈ [0, 3]``.

Nothing in this module compares an estimate to a true parameter. If a future change makes
``abs(beta_hat - beta_true)`` appear anywhere, the estimand has been swapped for a coefficient
and the study no longer measures what it claims to.

Why a `ResponseSurface`
-----------------------
Both adstock forms are linear filters, so ``adstock(m·x) == m·adstock(x)``. Adstocking the
status-quo spend once and rescaling it lets the optimiser explore allocations without
refiltering, which is the difference between a grid that finishes and one that does not — the
optimiser evaluates the response thousands of times per dataset. The equivalence is asserted
against `dgp.evaluate` in the tests rather than assumed.

Units: spend and sales are £k. Contributions and mROAS are per-channel arrays of length C,
in £k over the whole horizon and £k of sales per £k of spend respectively.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit

from mmm_recovery.dgp import Channel, DGPParams, SimResult
from mmm_recovery.transforms import (
    geometric_adstock,
    hill_saturation,
    hill_saturation_derivative,
    logistic_saturation,
    weibull_adstock,
)

Vector = np.ndarray[tuple[int], np.dtype[np.float64]]
"""A 1-D float64 array. Spelled out because scipy's stubs require the shape in the type."""

MAX_MULTIPLIER = 3.0
"""§3's upper bound on a channel's spend multiplier."""

MROAS_BUMP = 0.1
"""§3's 10% spend increment for the marginal-ROAS difference."""

AGREEMENT_TOLERANCE = 1e-3
"""Relative gap counted as two starts agreeing, for the `n_agreeing` diagnostic only.

CLAUDE.md Step 3 originally made this a *requirement*. **D17 removed that**: the surface is
non-concave, so requiring starts to agree is requiring convexity, and if it held then one
start would suffice. The threshold survives as a way of reporting how much the starts scatter.
"""

_BUDGET_TOLERANCE = 1e-8
"""Relative slack allowed on the budget equality before a solve is rejected as infeasible."""


class AllocatableSurface(Protocol):
    """What `optimal_allocation` needs from a response surface — and nothing more.

    **This protocol is how D18 is enforced rather than promised.** D18 requires the model's
    allocation solve to use an *identical* optimiser configuration to the truth solve, so that
    residual optimisation error is common-mode and cancels out of regret instead of masquerading
    as estimation error. The way to guarantee that is not to write the same configuration twice
    carefully — it is to have only one. `optimal_allocation` is that one solve, and it runs
    against anything shaped like this: the true surface built from `DGPParams`, or the fitted
    surface `estimator.py` builds from λ̂, α̂, κ̂ and β̂.

    Deliberately structural (no `runtime_checkable`, no base class). A fitted surface is not a
    kind of true surface and must not be able to inherit from one — that is precisely the
    direction a true parameter could leak.
    """

    @property
    def n_channels(self) -> int: ...

    @property
    def budget(self) -> float:
        """Total status-quo spend across all channels, £k over the horizon."""

    @property
    def spend_totals(self) -> NDArray[np.float64]:
        """(C,) total status-quo spend per channel, £k over the horizon."""

    def media_total(self, multipliers: NDArray[np.float64]) -> float:
        """Total media contribution over the horizon, £k."""

    def media_gradient(self, multipliers: NDArray[np.float64]) -> NDArray[np.float64]:
        """(C,) d(media_total)/dm_c, in closed form."""

    def total_sales(self, multipliers: NDArray[np.float64]) -> float:
        """Total sales over the horizon, £k. The objective §3 maximises."""

    def status_quo(self) -> NDArray[np.float64]:
        """The all-ones multiplier vector — the allocation the client already has."""


def _saturate(
    params: DGPParams, channel: Channel, adstocked: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply whichever saturation this condition's truth uses."""
    if params.misspecified:
        return logistic_saturation(
            adstocked,
            channel.half_saturation,
            channel.half_saturation * params.logistic_scale_ratio,
        )
    return hill_saturation(adstocked, channel.half_saturation, channel.hill_shape)


def _saturation_derivative(
    params: DGPParams, channel: Channel, adstocked: NDArray[np.float64]
) -> NDArray[np.float64]:
    """d(saturation)/d(adstocked spend), in closed form.

    Supplying this to SLSQP rather than letting it difference the objective is not a
    micro-optimisation: the optimiser runs once per dataset across a 4,500-dataset grid, and
    a finite-difference gradient costs C+1 objective evaluations per iteration.

    The Hill branch delegates to `transforms.hill_saturation_derivative`, which carries the
    `HILL_GRADIENT_FLOOR` that keeps the α < 1 slope at zero spend finite. Search (α = 0.9) is
    the channel that needs it, and only exactly at m_c = 0, a single boundary point.
    """
    values = np.asarray(adstocked, dtype=np.float64)
    if params.misspecified:
        scale = channel.half_saturation * params.logistic_scale_ratio
        response = expit((values - channel.half_saturation) / scale)
        at_zero = expit((0.0 - channel.half_saturation) / scale)
        return np.asarray(response * (1.0 - response) / (scale * (1.0 - at_zero)))
    return hill_saturation_derivative(values, channel.half_saturation, channel.hill_shape)


def _adstock(
    params: DGPParams, channel: Channel, spend: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply whichever adstock this condition's truth uses."""
    if params.misspecified:
        return weibull_adstock(
            spend, params.weibull_peak_lag, params.weibull_shape, params.weibull_max_lag
        )
    return geometric_adstock(spend, channel.decay)


@dataclass(frozen=True)
class ResponseSurface:
    """The noiseless response as a function of per-channel spend multipliers.

    Attributes:
        adstocked: (T, C) adstocked status-quo spend, £k per week.
        baseline: (T,) £k per week. Invariant to the multipliers, which is why the optimiser
            can maximise media contribution and total sales interchangeably.
        spend_totals: (C,) total status-quo spend per channel, £k over the horizon.
    """

    adstocked: NDArray[np.float64]
    baseline: NDArray[np.float64]
    spend_totals: NDArray[np.float64]
    params: DGPParams = field(repr=False)

    @property
    def n_channels(self) -> int:
        return len(self.params.channels)

    @property
    def budget(self) -> float:
        """Total status-quo spend across all channels, £k over the horizon."""
        return float(self.spend_totals.sum())

    def _checked(self, multipliers: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(multipliers, dtype=np.float64)
        if values.shape != (self.n_channels,):
            raise ValueError(
                f"multipliers must have shape ({self.n_channels},); got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("multipliers must be finite")
        if np.any(values < 0.0):
            raise ValueError(f"multipliers must be non-negative; got a minimum of {values.min()}")
        return values

    def contributions(self, multipliers: NDArray[np.float64]) -> NDArray[np.float64]:
        """(T, C) noiseless media contribution, £k per week, under these multipliers."""
        values = self._checked(multipliers)
        result = np.empty_like(self.adstocked)
        for index, channel in enumerate(self.params.channels):
            saturated = _saturate(self.params, channel, values[index] * self.adstocked[:, index])
            result[:, index] = channel.beta * saturated
        return result

    def media_total(self, multipliers: NDArray[np.float64]) -> float:
        """Total noiseless media contribution over the horizon, £k."""
        return float(self.contributions(multipliers).sum())

    def media_gradient(self, multipliers: NDArray[np.float64]) -> NDArray[np.float64]:
        """(C,) d(media_total)/dm_c, in closed form.

        Chain rule on ``β_c · Σ_t sat(m_c · a_t)``, so the derivative is
        ``β_c · Σ_t sat'(m_c · a_t) · a_t``. Verified against a central difference in the
        tests, because a wrong analytic gradient is exactly the kind of bug that produces a
        converged-looking solve at the wrong point.
        """
        values = self._checked(multipliers)
        gradient = np.empty(self.n_channels, dtype=np.float64)
        for index, channel in enumerate(self.params.channels):
            column = self.adstocked[:, index]
            derivative = _saturation_derivative(self.params, channel, values[index] * column)
            gradient[index] = channel.beta * float(np.dot(derivative, column))
        return gradient

    def total_sales(self, multipliers: NDArray[np.float64]) -> float:
        """Total noiseless sales over the horizon, £k. The objective §3 maximises."""
        return float(self.baseline.sum()) + self.media_total(multipliers)

    def status_quo(self) -> NDArray[np.float64]:
        """The all-ones multiplier vector — the allocation the client already has."""
        return np.ones(self.n_channels, dtype=np.float64)


def _assert_saturation_stays_differentiable(
    surface: ResponseSurface, max_multiplier: float
) -> None:
    """D2: no channel reaches the float64-flat region of the saturation at m_c = max.

    ``expit`` returns exactly 1.0 once its argument passes about 36.7, so a logistic with too
    small an s contains a genuinely flat region where the gradient is exactly zero — SLSQP
    would see no direction to move and stop, reporting success. D2 requires this to be
    asserted rather than assumed, so it is checked on every surface, for both saturation forms
    (Hill can overflow to a nan at extreme arguments by a different route).
    """
    for index, channel in enumerate(surface.params.channels):
        peak = float(surface.adstocked[:, index].max()) * max_multiplier
        probe = np.array([peak, peak * (1.0 + 1e-6) + 1e-9], dtype=np.float64)
        values = _saturate(surface.params, channel, probe)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{channel.name}: saturation is not finite at the optimiser's upper bound "
                f"(adstocked spend {peak:.1f} £k/week)"
            )
        if not values[0] < 1.0:
            raise ValueError(
                f"{channel.name}: saturation reaches exactly 1.0 at the optimiser's upper "
                f"bound (adstocked spend {peak:.1f} £k/week, kappa {channel.half_saturation}). "
                f"The gradient there is exactly zero and SLSQP will stall. This is the D2 "
                f"condition; the logistic scale ratio is too small."
            )
        if not values[1] > values[0]:
            raise ValueError(
                f"{channel.name}: saturation has a zero gradient at the optimiser's upper "
                f"bound (adstocked spend {peak:.1f} £k/week). See D2."
            )


def response_surface(sim: SimResult, max_multiplier: float = MAX_MULTIPLIER) -> ResponseSurface:
    """Build the interventional response surface for one simulated dataset.

    Args:
        sim: the dataset, carrying its own params, spend and baseline.
        max_multiplier: the optimiser's upper bound, used for the D2 differentiability check.

    Returns:
        A `ResponseSurface` whose `total_sales(ones)` reproduces `sim.noiseless_sales.sum()`
        bit for bit.
    """
    params = sim.params
    adstocked = np.empty_like(sim.spend)
    for index, channel in enumerate(params.channels):
        adstocked[:, index] = _adstock(params, channel, sim.spend[:, index])
    surface = ResponseSurface(
        adstocked=adstocked,
        baseline=sim.baseline,
        spend_totals=np.asarray(sim.spend.sum(axis=0), dtype=np.float64),
        params=params,
    )
    _assert_saturation_stays_differentiable(surface, max_multiplier)
    return surface


def incremental_contribution(surface: ResponseSurface) -> NDArray[np.float64]:
    """(C,) true incremental contribution per channel, £k over the horizon.

    ``sales(spend) - sales(spend | spend_c := 0)`` — §3's definition, computed by intervening
    on the generating process. For a channel with β = 0 this is **exactly** 0.0: its
    contribution column is exactly zero at every multiplier, so the two totals are summed from
    identical arrays and cancel bit for bit.
    """
    reference = surface.total_sales(surface.status_quo())
    result = np.empty(surface.n_channels, dtype=np.float64)
    for index in range(surface.n_channels):
        multipliers = surface.status_quo()
        multipliers[index] = 0.0
        result[index] = reference - surface.total_sales(multipliers)
    return result


def marginal_roas(surface: ResponseSurface, bump: float = MROAS_BUMP) -> NDArray[np.float64]:
    """(C,) true marginal ROAS per channel — £k of sales per additional £k of spend.

    ``[sales(1.1·spend_c) - sales(spend_c)] / (0.1 · Σ_t spend_c,t)``, noiseless. Exactly 0.0
    for a zero-β channel, by the same cancellation as `incremental_contribution`.
    """
    if bump <= 0.0:
        raise ValueError(f"bump must be positive; got {bump}")
    reference = surface.total_sales(surface.status_quo())
    result = np.empty(surface.n_channels, dtype=np.float64)
    for index in range(surface.n_channels):
        multipliers = surface.status_quo()
        multipliers[index] = 1.0 + bump
        gained = surface.total_sales(multipliers) - reference
        result[index] = gained / (bump * float(surface.spend_totals[index]))
    return result


@dataclass(frozen=True)
class OptimalAllocation:
    """The best allocation the true response surface admits, and how sure we are of it.

    **`total_sales` is the best value found, not a proven global maximum** (D17). The surface
    is non-concave; the claim rests on matching a 256-start reference on 100 of 100 trials,
    which is empirical support rather than proof. `n_agreeing` and `spread` below are what
    make the non-concavity visible in the output rather than only in this docstring.

    Attributes:
        multipliers: (C,) the optimal per-channel spend multipliers.
        total_sales: noiseless total sales under them, £k over the horizon.
        status_quo_sales: noiseless total sales under the current allocation, £k.
        n_starts: how many starting points were tried.
        n_agreeing: how many landed within `AGREEMENT_TOLERANCE` of the best, on media
            contribution. Reported as a diagnostic, not required — see `optimal_allocation`.
        spread: relative gap between the best and worst start, on media contribution.
            Measured on media rather than total sales because the baseline is a large constant
            the multipliers cannot move; including it divides every gap by roughly four and
            flatters the number.
    """

    multipliers: NDArray[np.float64]
    total_sales: float
    status_quo_sales: float
    n_starts: int
    n_agreeing: int
    spread: float

    @property
    def achievable_lift(self) -> float:
        """`S_opt - S_status_quo`, the denominator of allocation regret, £k."""
        return self.total_sales - self.status_quo_sales


def _structured_starts(
    surface: AllocatableSurface, max_multiplier: float, min_multiplier: float = 0.0
) -> list[NDArray[np.float64]]:
    """Status quo, equal budget share, and each channel pushed to its upper bound.

    These are not decoration. This response surface is **not concave** — Hill with α > 1 is
    S-shaped, and TV (1.8) and OOH (2.2) both sit there — so the budget-constrained problem has
    genuine local optima, most of them at points where one S-shaped channel is scaled up and
    the rest starve. A Dirichlet cloud finds those basins only by luck. Seeding each channel's
    basin deliberately takes the miss rate against a 256-start reference from 8% to 0%.

    The status quo is always first and is exactly feasible, which is what makes the optimum
    provably no worse than doing nothing rather than merely usually so.
    """
    n_channels = surface.n_channels
    totals, budget = surface.spend_totals, surface.budget
    starts = [
        surface.status_quo(),
        np.clip((budget / n_channels) / totals, min_multiplier, max_multiplier),
    ]
    for channel in range(n_channels):
        candidate = np.full(n_channels, min_multiplier, dtype=np.float64)
        candidate[channel] = max_multiplier
        committed = max_multiplier * float(totals[channel])
        others = [index for index in range(n_channels) if index != channel]
        if others and committed < budget:
            candidate[others] = (budget - committed) / float(totals[others].sum())
        starts.append(np.clip(candidate, min_multiplier, max_multiplier))
    return starts


def _screened_random_starts(
    surface: AllocatableSurface,
    n_wanted: int,
    pool: int,
    max_multiplier: float,
    seed: int,
    min_multiplier: float = 0.0,
) -> list[NDArray[np.float64]]:
    """Draw `pool` budget-feasible points and keep the `n_wanted` scoring highest.

    Evaluating the response is far cheaper than optimising from it, so screening a large pool
    buys global coverage for the price of a fraction of one SLSQP run.
    """
    if n_wanted <= 0:
        return []
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(surface.n_channels), size=pool)
    candidates = [
        np.clip(row * surface.budget / surface.spend_totals, min_multiplier, max_multiplier)
        for row in weights
    ]
    candidates.sort(key=surface.media_total, reverse=True)
    return candidates[:n_wanted]


def _starting_points(
    surface: AllocatableSurface,
    n_starts: int,
    max_multiplier: float,
    seed: int,
    pool: int,
    min_multiplier: float = 0.0,
) -> list[NDArray[np.float64]]:
    """All structured starts, topped up with screened random ones to reach `n_starts`."""
    structured = _structured_starts(surface, max_multiplier, min_multiplier)
    if n_starts < len(structured):
        raise ValueError(
            f"n_starts must be at least {len(structured)} so every channel's basin is seeded; "
            f"got {n_starts}"
        )
    return structured + _screened_random_starts(
        surface, n_starts - len(structured), pool, max_multiplier, seed, min_multiplier
    )


def optimal_allocation(
    surface: AllocatableSurface,
    seed: int,
    n_starts: int = 8,
    max_multiplier: float = MAX_MULTIPLIER,
    tolerance: float = AGREEMENT_TOLERANCE,
    screening_pool: int = 200,
    min_multiplier: float = 0.0,
) -> OptimalAllocation:
    """Maximise noiseless sales subject to the budget, from multiple starts, with SLSQP.

    §3: weekly spend patterns are preserved and only scaled, which mirrors how budgets are
    actually reallocated — a planner moves money between channels, not between weeks.

    Every solve's exit status is checked and **a failed solve raises**. CLAUDE.md names the
    alternative as a specific failure mode: an optimiser that quietly returns its starting
    point produces an `S_opt` barely above status quo, which then reads downstream as a small
    achievable lift and a flatteringly low regret. A crash is recoverable; a plausible number
    is not. Feasibility and the bounds are re-checked independently of `outcome.success`,
    because a solver reporting success is a claim, not evidence.

    **The starts are not required to agree** (D17). CLAUDE.md's "multi-start solutions agree
    within 0.1%" does not hold for this surface — only 12-31% of starts land within 0.1% of
    the best, because the problem is genuinely non-concave. Taking the best of many starts is
    what multi-start is *for*. What is required instead, and what the tests assert, is the
    stronger and more useful property: the returned optimum matches a 256-start reference on
    every trial. `n_agreeing` and `spread` are reported so the non-concavity stays visible.

    **Limitation, per D17, and it carries wherever this optimum is reported: global optimality
    here is empirically supported, not proven.** On a non-concave surface no finite set of
    starts can prove it. The evidence is that the structured-plus-screened design matched a
    256-start reference on 100 of 100 trials across five representative cells, where plain
    Dirichlet starts missed on 8%. That is a strong prior, not a guarantee, and any regret
    computed against `total_sales` inherits the same status.

    Args:
        surface: the true response surface.
        seed: seeds the screened random starting points. Explicit, per CLAUDE.md rule 4.
        n_starts: total starting points. Must cover the structured set, which is 2 + C.
        max_multiplier: §3's upper bound on m_c.
        tolerance: relative gap counted as agreement, for the diagnostic only.
        screening_pool: random points evaluated before the best are handed to SLSQP.

    Returns:
        The best allocation found, with the multi-start agreement recorded alongside it.

    Raises:
        ValueError: if any solve fails, if any solution is infeasible or out of bounds, or if
            the optimum fails to beat the status quo.
    """

    if not 0.0 <= min_multiplier < max_multiplier:
        raise ValueError(
            f"need 0 <= min_multiplier < max_multiplier; got [{min_multiplier}, {max_multiplier}]"
        )
    totals = surface.spend_totals
    budget = surface.budget
    bounds = [(min_multiplier, max_multiplier)] * surface.n_channels

    # Both the objective and the constraint are scaled to O(1). Without this SLSQP fails from
    # every start with "positive directional derivative for linesearch": media contribution is
    # of order 1.3e5 £k, so a finite-difference step of 1.5e-8 moves it by about 7e-4 — a
    # relative change of 6e-9, which is down in the summation noise of a T x C reduction. The
    # analytic gradient below removes the differencing entirely, and the scaling keeps the
    # line search and the ftol test in a sane numeric range.
    objective_scale = surface.media_total(surface.status_quo())
    if not objective_scale > 0.0:
        raise ValueError("every channel has zero contribution at the status quo")

    # The budget constraint is linear in the multipliers, so it is declared as such rather
    # than as an opaque callable: SLSQP gets the exact Jacobian for free and never differences
    # it. Scaled to `Σ m_c·S_c / budget == 1` for the same reason the objective is scaled.
    constraints = LinearConstraint(totals / budget, lb=1.0, ub=1.0)

    def objective(multipliers: Vector, /) -> float:
        return -surface.media_total(multipliers) / objective_scale

    def objective_jacobian(multipliers: Vector, /) -> Vector:
        return -surface.media_gradient(multipliers) / objective_scale

    solutions: list[tuple[float, NDArray[np.float64]]] = []
    starts = _starting_points(
        surface, n_starts, max_multiplier, seed, screening_pool, min_multiplier
    )
    for index, start in enumerate(starts):
        outcome = minimize(
            objective,
            start,
            method="SLSQP",
            jac=objective_jacobian,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-12},
        )
        if not outcome.success:
            raise ValueError(
                f"SLSQP failed from start {index}: status {outcome.status}, "
                f"{outcome.message!r}. Refusing to return a number that would read as a "
                f"small achievable lift downstream."
            )
        solution = np.asarray(outcome.x, dtype=np.float64)
        residual = abs(float(np.dot(solution, totals)) - budget)
        if residual > _BUDGET_TOLERANCE * budget:
            raise ValueError(
                f"SLSQP reported success from start {index} but the budget constraint is "
                f"violated by {residual:.6g} £k ({residual / budget:.3e} relative)"
            )
        if np.any(solution < min_multiplier - 1e-9) or np.any(solution > max_multiplier + 1e-9):
            raise ValueError(
                f"SLSQP returned a solution outside [{min_multiplier}, {max_multiplier}]"
            )
        bounded = np.clip(solution, min_multiplier, max_multiplier)
        solutions.append((surface.media_total(bounded), bounded))

    best_media, best_multipliers = max(solutions, key=lambda pair: pair[0])
    worst_media = min(media for media, _ in solutions)
    spread = (best_media - worst_media) / abs(best_media)
    n_agreeing = sum(
        1 for media, _ in solutions if (best_media - media) / abs(best_media) <= tolerance
    )

    best_sales = surface.total_sales(best_multipliers)
    status_quo_sales = surface.total_sales(surface.status_quo())
    if best_sales < status_quo_sales:
        raise ValueError(
            f"the optimum ({best_sales:.2f} £k) is worse than the status quo "
            f"({status_quo_sales:.2f} £k), which is itself a feasible starting point — the "
            f"solve is wrong, not the allocation"
        )

    return OptimalAllocation(
        multipliers=best_multipliers,
        total_sales=best_sales,
        status_quo_sales=status_quo_sales,
        n_starts=len(solutions),
        n_agreeing=n_agreeing,
        spread=spread,
    )
