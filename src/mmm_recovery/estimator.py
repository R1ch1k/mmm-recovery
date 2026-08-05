"""`RidgeMMM` — the estimator under test, seeing exactly what a real analyst sees.

PREREGISTRATION.md §4: geometric adstock → Hill saturation → ridge regression with
non-negative media coefficients, plus intercept, linear trend and 2 Fourier pairs at period 52
as controls. Hyperparameters by random search over bounded ranges, scored by expanding-window
time-series CV on RMSE. Uncertainty by moving-block bootstrap of residuals at fixed
hyperparameters.

What the estimator may see
--------------------------
**Spend and sales. Nothing else.** CLAUDE.md's non-negotiable rule 3, and the reason C3 and C7
measure anything at all: latent demand ``d_t`` is the confounder those conditions introduce, so
an estimator that could see it would be answering a different question. The guarantee here is
structural rather than a matter of care:

* `fit` takes two arrays. It never receives a `SimResult`, so ``d_t`` is not in scope — there
  is no expression that could reach it.
* `control_matrix` takes the number of weeks and nothing else. Every control column is a
  function of ``t`` alone.
* `design_matrix` checks its own width on every call and raises if a column has appeared from
  anywhere. That assertion is permanent.

`tests/test_estimator.py` attacks the guarantee from a fourth direction: it re-fits with
``d_t`` replaced by noise and requires the coefficients to be bit-identical.

Deliberately misspecified, in two conditions
--------------------------------------------
The estimator always assumes geometric adstock and Hill saturation. Under C4 and C7 the truth
is Weibull adstock and logistic saturation. That gap is the condition, not a bug — do not add a
Weibull option here.

Why the fitted surface is not a `ResponseSurface`
-------------------------------------------------
`FittedSurface` satisfies `truth.AllocatableSurface` structurally and inherits from nothing, so
the *same* `optimal_allocation` solves the model's allocation and the truth's. That is
**D18**: one optimiser configuration, so residual optimisation error is common-mode and drops
out of regret instead of being counted as estimation error. Making the fitted surface a
subclass of the true one would have been the shorter route and would have opened the exact
channel through which a true parameter could leak.

Units: spend and sales are £k per week; contributions are £k over the horizon; mROAS is £k of
sales per £k of spend.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear

from mmm_recovery.dgp import WEEKS_PER_YEAR
from mmm_recovery.transforms import (
    geometric_adstock,
    hill_saturation,
    hill_saturation_derivative,
)
from mmm_recovery.truth import (
    MROAS_BUMP,
    AllocatableSurface,
    OptimalAllocation,
    optimal_allocation,
)

N_SEARCH_DRAWS: Final = 200
"""§4's random-search budget. CLAUDE.md's performance note: cut this before cutting seeds."""

N_CV_FOLDS: Final = 3
"""§4's expanding-window fold count."""

BOOTSTRAP_BLOCK_WEEKS: Final = 13
"""§4's moving-block length — one quarter, long enough to carry the AR(1) demand structure."""

N_BOOTSTRAP_REPLICATES: Final = 200
"""§4's B."""

INTERVAL_LEVEL: Final = 0.90
"""§6 grades coverage of the nominal 90% interval."""

FOURIER_PAIRS: Final = 2
"""§4's control specification: 2 Fourier pairs at period 52."""

CONTROL_COLUMN_NAMES: Final = (
    "intercept",
    "trend",
    "cos52_1",
    "sin52_1",
    "cos52_2",
    "sin52_2",
    "trend_x_cos52_1",
    "trend_x_sin52_1",
    "trend_x_cos52_2",
    "trend_x_sin52_2",
)
"""Every non-media column, in order. Each is a function of the week index and nothing else.

The last four are **D22**. §2 builds the baseline as a *product*, ``B0·(1 + τ·t/T)·season_t``,
which expands to ``B0 + B0·τ·(t/T) + B0·season' + B0·τ·(t/T)·season'``. §4's control list names
the first three terms and not the fourth, so an additive control block cannot represent the
baseline of the very world it is being tested in — a structured residual of 2.45 £k/week,
0.25% of sales, in *every* condition including the clean one. Adding ``trend × Fourier``
completes the span and drives that residual to 1e-13.

This is not a concession to the estimator. It removes the last implementation-level
explanation for C0's failure, so what remains is structural. Every column is still a function
of the week index alone, so the leakage guarantee is untouched.
"""

N_CONTROL_COLUMNS: Final = len(CONTROL_COLUMN_NAMES)

MAX_OBSERVED_CONDITION_NUMBER: Final = 3.2e5
"""Worst ``cond(X'X + τD)`` measured across the grid's hardest cells, for reference.

There is deliberately **no conditioning jitter** on the normal equations. One was written and
then removed: measured over 4,800 solves spanning C0, C1 at ρ=0.95, C2 at T=104 and C7 — every
random-search draw, at both fold sizes — the Cholesky never failed, and the worst condition
number was 3.1e5, eleven orders of magnitude clear of float64's limit. A jitter would have been
inert at best; at worst it shrinks the *unpenalised* control block, which is statistical work
done under a numerical name. If a future condition does make the system singular, the Cholesky
raises, which is the behaviour CLAUDE.md asks for.
"""


@dataclass(frozen=True)
class SearchBounds:
    """The bounded ranges §4 requires for the random search, without saying what they are.

    Every bound is deliberately wider than the truth it has to find, and none of them is
    derived from `dgp.py` — a search told where to look would flatter the estimator. `κ` is the
    exception in form only: it is drawn as a multiple of each channel's *observed* mean
    adstocked spend, which is data the analyst has, because a fixed £k range would be
    meaningless across channels spending 12 and 55 £k per week.

    Attributes:
        decay: λ range. True values span 0.10-0.70.
        hill_shape: α range. True values span 0.9-2.2.
        half_saturation_ratio: κ as a multiple of mean adstocked spend. True ratios span
            0.90 (search) to 2.08 (OOH). Drawn log-uniformly.
        ridge_penalty: the ridge coefficient relative to ``trace(X'X)/p``, drawn log-uniformly
            so the search spends its draws evenly across orders of magnitude.
    """

    decay: tuple[float, float] = (0.0, 0.9)
    hill_shape: tuple[float, float] = (0.5, 3.0)
    half_saturation_ratio: tuple[float, float] = (0.3, 3.0)
    ridge_penalty: tuple[float, float] = (1e-8, 1.0)

    def __post_init__(self) -> None:
        for name in ("decay", "hill_shape", "half_saturation_ratio", "ridge_penalty"):
            low, high = getattr(self, name)
            if not low < high:
                raise ValueError(f"{name} bounds must be increasing; got ({low}, {high})")
        if self.decay[0] < 0.0 or self.decay[1] >= 1.0:
            raise ValueError(f"decay bounds must lie in [0, 1); got {self.decay}")
        if self.hill_shape[0] <= 0.0:
            raise ValueError(f"hill_shape must stay positive; got {self.hill_shape}")
        if self.half_saturation_ratio[0] <= 0.0 or self.ridge_penalty[0] <= 0.0:
            raise ValueError("log-uniform ranges must have positive lower bounds")


@dataclass(frozen=True)
class Hyperparameters:
    """One random-search draw: the transform per channel, plus the shared ridge penalty.

    Attributes:
        decay: (C,) λ̂ per channel.
        hill_shape: (C,) α̂ per channel.
        half_saturation: (C,) κ̂ per channel, £k per week.
        ridge_penalty: dimensionless, scaled by ``trace(X'X)/p`` before it is applied.
    """

    decay: NDArray[np.float64]
    hill_shape: NDArray[np.float64]
    half_saturation: NDArray[np.float64]
    ridge_penalty: float

    @property
    def n_channels(self) -> int:
        return int(self.decay.size)


def control_matrix(n_weeks: int) -> NDArray[np.float64]:
    """(T, 10) intercept, linear trend, 2 Fourier pairs at period 52, and trend × Fourier.

    **The whole argument list is `n_weeks`.** Every column is a function of the week index, so
    no observed or latent series can enter the control block, whatever a future caller passes
    around it. That is the first of the three structural guarantees in the module docstring,
    and D22's four extra columns do not weaken it — a product of two functions of ``t`` is
    still a function of ``t``.

    The trend runs 0 to 1 across the horizon rather than 0 to T, so its column is O(1) like
    every other and the ridge penalty's scale does not depend on the length of the series.
    That choice also keeps the interaction columns O(1), which matters more now that there
    are four of them.

    See `CONTROL_COLUMN_NAMES` for why the interactions are there.
    """
    if n_weeks < WEEKS_PER_YEAR:
        raise ValueError(f"n_weeks must cover at least one seasonal cycle; got {n_weeks}")
    weeks = np.arange(n_weeks, dtype=np.float64)
    trend = weeks / float(n_weeks - 1)
    fourier: list[NDArray[np.float64]] = []
    for harmonic in range(1, FOURIER_PAIRS + 1):
        angle = 2.0 * np.pi * harmonic * weeks / WEEKS_PER_YEAR
        fourier.extend((np.cos(angle), np.sin(angle)))
    columns = [np.ones(n_weeks, dtype=np.float64), trend, *fourier]
    columns.extend(trend * column for column in fourier)
    return np.column_stack(columns)


def adstock_spend(spend: NDArray[np.float64], decay: NDArray[np.float64]) -> NDArray[np.float64]:
    """(T, C) geometric adstock applied per channel at λ̂.

    Split out from `media_columns` because the optimiser needs the adstocked series on its own:
    adstock is a linear filter, so ``adstock(m·x) == m·adstock(x)`` and an allocation can be
    explored by rescaling this once rather than refiltering per evaluation. Same trick, and
    the same reason, as `truth.ResponseSurface`.
    """
    if spend.ndim != 2:
        raise ValueError(f"spend must be (T, C); got shape {spend.shape}")
    if decay.shape != (spend.shape[1],):
        raise ValueError(f"decay must have one entry per channel; got shape {decay.shape}")
    adstocked = np.empty_like(spend, dtype=np.float64)
    for index in range(spend.shape[1]):
        adstocked[:, index] = geometric_adstock(spend[:, index], float(decay[index]))
    return adstocked


def saturate_columns(
    adstocked: NDArray[np.float64],
    half_saturation: NDArray[np.float64],
    hill_shape: NDArray[np.float64],
) -> NDArray[np.float64]:
    """(T, C) Hill saturation applied per channel at (κ̂, α̂)."""
    saturated = np.empty_like(adstocked, dtype=np.float64)
    for index in range(adstocked.shape[1]):
        saturated[:, index] = hill_saturation(
            adstocked[:, index], float(half_saturation[index]), float(hill_shape[index])
        )
    return saturated


def design_matrix(spend: NDArray[np.float64], hyper: Hyperparameters) -> NDArray[np.float64]:
    """(T, C + 10) the full design: saturated adstocked spend, then the controls.

    Raises:
        ValueError: if the assembled matrix is not exactly C media columns plus the six
            controls. **This check is permanent** (CLAUDE.md rule 3). It is cheap, it runs on
            every fit, and it is the assertion that a column smuggled in from anywhere — a
            latent series, a true parameter, a helpfully added interaction — cannot survive.
    """
    if spend.ndim != 2:
        raise ValueError(f"spend must be (T, C); got shape {spend.shape}")
    n_weeks, n_channels = spend.shape
    if hyper.n_channels != n_channels:
        raise ValueError(
            f"hyperparameters cover {hyper.n_channels} channels but spend has {n_channels}"
        )
    adstocked = adstock_spend(spend, hyper.decay)
    media = saturate_columns(adstocked, hyper.half_saturation, hyper.hill_shape)
    matrix = np.hstack([media, control_matrix(n_weeks)])
    expected = (n_weeks, n_channels + N_CONTROL_COLUMNS)
    if matrix.shape != expected:
        raise ValueError(
            f"design matrix is {matrix.shape}, expected {expected}. The estimator sees "
            f"{n_channels} media columns and the {N_CONTROL_COLUMNS} controls "
            f"{CONTROL_COLUMN_NAMES} — nothing else. A column has come from somewhere it "
            f"should not have."
        )
    return matrix


def design_column_names(channel_names: tuple[str, ...]) -> tuple[str, ...]:
    """The design matrix's columns, in order. The estimator's entire view of the world."""
    return (*channel_names, *CONTROL_COLUMN_NAMES)


def _normal_equations(
    design: NDArray[np.float64], sales: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(X'X, X'y)``, cached once per transform and reused by every fold and replicate.

    The estimator-performance plan in one line: for `RidgeMMM` the solve is an 11×11 system and
    costs nothing, while the adstock-plus-saturation transform is the real expense. Reducing to
    the normal equations here means the 200 bootstrap replicates — which hold hyperparameters
    fixed, so ``X`` never changes — re-use this ``X'X`` and only recompute ``X'y``.
    """
    return design.T @ design, design.T @ sales


def _solve_bounded_ridge(
    gram: NDArray[np.float64],
    moment: NDArray[np.float64],
    ridge_penalty: float,
    n_media: int,
) -> NDArray[np.float64]:
    """Ridge with the media coefficients constrained non-negative, from the normal equations.

    The objective is ``‖Xb − y‖² + τ‖b_media‖²`` with ``b_media ≥ 0`` and the controls free.
    Written in terms of ``M = X'X + τD`` and ``g = X'y`` it is ``b'Mb − 2b'g``, and factoring
    ``M = R'R`` turns that into an ordinary bounded least-squares problem ``‖Rb − z‖²`` with
    ``z = R'⁻¹g``. That is an 11-row problem regardless of how long the series is, which is
    what makes 200 draws × 3 folds × 4,500 datasets affordable.

    Only the media block is penalised. Shrinking the controls would push variance the trend and
    seasonality should absorb into the media coefficients instead, which is the direction that
    inflates measured media contribution — the exact quantity the study is measuring.

    Raises:
        numpy.linalg.LinAlgError: if the penalised normal equations are not positive definite.
            Deliberately not caught: a fit that cannot be solved is not a fit with a small
            coefficient, and swallowing it here would surface downstream as a real result.
    """
    size = gram.shape[0]
    penalties = np.zeros(size, dtype=np.float64)
    penalties[:n_media] = ridge_penalty * float(np.trace(gram)) / size
    penalised = gram + np.diag(penalties)

    lower = np.linalg.cholesky(penalised)
    projected = np.linalg.solve(lower, moment)

    bounds_low = np.concatenate([np.zeros(n_media), np.full(size - n_media, -np.inf)])
    bounds_high = np.full(size, np.inf)
    outcome = lsq_linear(lower.T, projected, bounds=(bounds_low, bounds_high), method="bvls")
    if not outcome.success:
        raise ValueError(
            f"bounded ridge solve failed: status {outcome.status}, {outcome.message!r}"
        )
    return np.asarray(outcome.x, dtype=np.float64)


def expanding_window_folds(n_weeks: int, n_folds: int = N_CV_FOLDS) -> list[tuple[int, int]]:
    """`(train_end, test_end)` per fold — train on everything before the split, test after it.

    The first half of the series is training data for every fold and the second half is cut
    into `n_folds` consecutive test blocks. §4 fixes "expanding window, 3 folds" and leaves the
    split points open; anchoring the initial window at half the series is what keeps the design
    usable at T=104, where the alternative of equal (n_folds+1) blocks would score the first
    fold on 26 training weeks against 11 parameters.

    Returns:
        Fold boundaries in week indices, earliest first.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be at least 1; got {n_folds}")
    initial = n_weeks // 2
    block = (n_weeks - initial) // n_folds
    if block < 1:
        raise ValueError(f"n_weeks={n_weeks} is too short for {n_folds} expanding-window folds")
    return [(initial + fold * block, initial + (fold + 1) * block) for fold in range(n_folds)]


def _cross_validated_rmse(
    design: NDArray[np.float64],
    sales: NDArray[np.float64],
    ridge_penalty: float,
    n_media: int,
    folds: list[tuple[int, int]],
) -> float:
    """Pooled out-of-sample RMSE across the expanding-window folds, £k per week.

    Pooled rather than averaged per fold: the folds are equal-length by construction here, so
    the two agree, and pooling stays correct if that ever stops being true.
    """
    squared = 0.0
    counted = 0
    for train_end, test_end in folds:
        gram, moment = _normal_equations(design[:train_end], sales[:train_end])
        coefficients = _solve_bounded_ridge(gram, moment, ridge_penalty, n_media)
        residual = sales[train_end:test_end] - design[train_end:test_end] @ coefficients
        squared += float(residual @ residual)
        counted += test_end - train_end
    return float(np.sqrt(squared / counted))


def _log_uniform(
    rng: np.random.Generator, bounds: tuple[float, float], size: int
) -> NDArray[np.float64]:
    """(size,) draws uniform on the log scale, so each order of magnitude gets equal weight."""
    low, high = float(np.log(bounds[0])), float(np.log(bounds[1]))
    return np.asarray(np.exp(rng.uniform(low=low, high=high, size=size)), dtype=np.float64)


def _draw_hyperparameters(
    spend: NDArray[np.float64], bounds: SearchBounds, rng: np.random.Generator
) -> Hyperparameters:
    """One random-search draw. κ̂ is a multiple of *observed* mean adstocked spend."""
    n_channels = spend.shape[1]
    decay = np.asarray(
        rng.uniform(low=bounds.decay[0], high=bounds.decay[1], size=n_channels),
        dtype=np.float64,
    )
    hill_shape = np.asarray(
        rng.uniform(low=bounds.hill_shape[0], high=bounds.hill_shape[1], size=n_channels),
        dtype=np.float64,
    )
    ratio = _log_uniform(rng, bounds.half_saturation_ratio, n_channels)
    mean_adstocked = adstock_spend(spend, decay).mean(axis=0)
    return Hyperparameters(
        decay=decay,
        hill_shape=hill_shape,
        half_saturation=ratio * mean_adstocked,
        ridge_penalty=float(_log_uniform(rng, bounds.ridge_penalty, 1)[0]),
    )


@dataclass(frozen=True)
class FittedSurface:
    """The model's belief about the response, as a function of per-channel spend multipliers.

    Shaped to satisfy `truth.AllocatableSurface` and nothing else, so `optimal_allocation`
    solves it with the identical configuration it uses on the true surface — **D18**.

    It carries no true parameter and no reference to `DGPParams`. Everything in it came out of
    the fit: λ̂ through `adstocked`, (κ̂, α̂) through the saturation, β̂ through `coefficients`,
    and the control coefficients through `baseline`.

    Attributes:
        adstocked: (T, C) status-quo spend adstocked at λ̂, £k per week.
        coefficients: (C,) β̂, £k. Non-negative by construction.
        half_saturation: (C,) κ̂, £k per week.
        hill_shape: (C,) α̂.
        baseline: (T,) the fitted control contribution — intercept, trend and Fourier terms at
            their estimated coefficients, £k per week. Invariant to the multipliers, exactly as
            the true baseline is, which is why maximising media and maximising sales coincide.
        spend_totals: (C,) total status-quo spend per channel, £k over the horizon.
    """

    adstocked: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    half_saturation: NDArray[np.float64]
    hill_shape: NDArray[np.float64]
    baseline: NDArray[np.float64]
    spend_totals: NDArray[np.float64]

    @property
    def n_channels(self) -> int:
        return int(self.coefficients.size)

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
        """(T, C) estimated media contribution, £k per week, under these multipliers."""
        values = self._checked(multipliers)
        result = np.empty_like(self.adstocked)
        for index in range(self.n_channels):
            saturated = hill_saturation(
                values[index] * self.adstocked[:, index],
                float(self.half_saturation[index]),
                float(self.hill_shape[index]),
            )
            result[:, index] = self.coefficients[index] * saturated
        return result

    def media_total(self, multipliers: NDArray[np.float64]) -> float:
        """Total estimated media contribution over the horizon, £k."""
        return float(self.contributions(multipliers).sum())

    def media_gradient(self, multipliers: NDArray[np.float64]) -> NDArray[np.float64]:
        """(C,) d(media_total)/dm_c, in closed form — the same chain rule truth.py uses."""
        values = self._checked(multipliers)
        gradient = np.empty(self.n_channels, dtype=np.float64)
        for index in range(self.n_channels):
            column = self.adstocked[:, index]
            derivative = hill_saturation_derivative(
                values[index] * column,
                float(self.half_saturation[index]),
                float(self.hill_shape[index]),
            )
            gradient[index] = self.coefficients[index] * float(np.dot(derivative, column))
        return gradient

    def total_sales(self, multipliers: NDArray[np.float64]) -> float:
        """Total *predicted* sales over the horizon, £k — the model's own objective.

        Never a term in regret. Regret evaluates the multipliers this surface recommends on
        the **true** surface; what the model predicted its own advice would earn is a
        diagnostic, and treating it as an outcome is the error the whole study is about.
        """
        return float(self.baseline.sum()) + self.media_total(multipliers)

    def status_quo(self) -> NDArray[np.float64]:
        """The all-ones multiplier vector — the allocation the client already has."""
        return np.ones(self.n_channels, dtype=np.float64)


def estimated_contribution(surface: FittedSurface) -> NDArray[np.float64]:
    """(C,) estimated incremental contribution per channel, £k over the horizon.

    Computed by the same intervention §3 defines for the truth — total minus total with the
    channel zeroed — rather than read off β̂. The two happen to coincide here because Hill maps
    0 to 0 exactly, but writing it as the intervention is what keeps estimate and estimand the
    same quantity, so the bias in §6 is a bias in the thing the decision depends on.
    """
    reference = surface.media_total(surface.status_quo())
    result = np.empty(surface.n_channels, dtype=np.float64)
    for index in range(surface.n_channels):
        multipliers = surface.status_quo()
        multipliers[index] = 0.0
        result[index] = reference - surface.media_total(multipliers)
    return result


def estimated_marginal_roas(
    surface: FittedSurface, bump: float = MROAS_BUMP
) -> NDArray[np.float64]:
    """(C,) estimated mROAS per channel — £k of sales per additional £k of spend."""
    if bump <= 0.0:
        raise ValueError(f"bump must be positive; got {bump}")
    reference = surface.media_total(surface.status_quo())
    result = np.empty(surface.n_channels, dtype=np.float64)
    for index in range(surface.n_channels):
        multipliers = surface.status_quo()
        multipliers[index] = 1.0 + bump
        gained = surface.media_total(multipliers) - reference
        result[index] = gained / (bump * float(surface.spend_totals[index]))
    return result


@dataclass(frozen=True)
class MMMFit:
    """Everything one `RidgeMMM` fit produces. No true parameter appears anywhere in it.

    Attributes:
        hyperparameters: the winning random-search draw.
        coefficients: (C + 10,) the full coefficient vector, media block first.
        surface: the fitted response surface, ready for `recommended_allocation`.
        contribution: (C,) estimated incremental contribution, £k over the horizon.
        contribution_interval: (C, 2) the 5th and 95th bootstrap percentiles of the same,
            £k. **Understates total uncertainty — see `bootstrap_contributions`.**
        marginal_roas: (C,) estimated mROAS.
        cv_rmse: the winning draw's pooled out-of-sample RMSE, £k per week.
        n_draws_scored: how many random-search draws were actually scored.
    """

    hyperparameters: Hyperparameters
    coefficients: NDArray[np.float64]
    surface: FittedSurface
    contribution: NDArray[np.float64]
    contribution_interval: NDArray[np.float64]
    marginal_roas: NDArray[np.float64]
    cv_rmse: float
    n_draws_scored: int

    @property
    def n_channels(self) -> int:
        return self.surface.n_channels

    @property
    def interval_width(self) -> NDArray[np.float64]:
        """(C,) width of the nominal 90% contribution interval, £k."""
        return np.asarray(
            self.contribution_interval[:, 1] - self.contribution_interval[:, 0],
            dtype=np.float64,
        )


def _moving_block_indices(n_weeks: int, block: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """(T,) row indices drawn as overlapping blocks of `block` consecutive weeks.

    Blocks preserve the serial correlation an IID residual bootstrap would destroy. That
    matters here because the residual carries the AR(1) part of latent demand — the component
    the Fourier controls cannot absorb — so an IID resample would report intervals far tighter
    than the data supports, precisely on the conditions the study cares most about.
    """
    if not 1 <= block <= n_weeks:
        raise ValueError(f"block must lie in [1, {n_weeks}]; got {block}")
    n_blocks = -(-n_weeks // block)
    starts = rng.integers(0, n_weeks - block + 1, size=n_blocks)
    offsets = np.arange(block, dtype=np.intp)
    return np.asarray((starts[:, None] + offsets[None, :]).ravel()[:n_weeks], dtype=np.intp)


def bootstrap_contributions(
    design: NDArray[np.float64],
    sales: NDArray[np.float64],
    surface: FittedSurface,
    ridge_penalty: float,
    seed: int,
    n_replicates: int = N_BOOTSTRAP_REPLICATES,
    block: int = BOOTSTRAP_BLOCK_WEEKS,
) -> NDArray[np.float64]:
    """(B, C) estimated contribution per bootstrap replicate, £k over the horizon.

    **Known limitation, §4, and it must travel with every coverage number computed from these
    intervals: the hyperparameters are held fixed at the selected draw, so this understates
    total uncertainty.** The transform (λ̂, α̂, κ̂) is treated as known when it was in fact
    chosen from the same data, and the selection step contributes variance that never appears
    here.

    **The limitation is more severe than "understates uncertainty" suggests, and this is the
    load-bearing caveat on G2.** The omitted term is not a secondary correction — on this
    problem it is the *dominant* source of uncertainty. The hyperparameter surface is a
    plateau: 177 of 780 points on a sweep of TV's (α, κ) sit within 1% of the true
    parameters' CV RMSE while implying channel contributions spanning 43,938 to 240,522 £k, a
    5.5× range. Conditioning on one point of that plateau and resampling residuals around it
    prices the *smallest* component of the error and omits the largest.

    So the interval this produces answers "how much would this estimate move if the noise had
    been drawn differently", not "how much do we know about the contribution". Measured
    consequence: G2 coverage on C0 is 32.0% against an 80% threshold. That number is not
    evidence of a bootstrap defect — it follows directly from pricing the wrong term, and it
    is what §4's construction guarantees on a non-identified surface.

    Holding them fixed is what §4 specifies and it is also what makes the grid affordable —
    ``X`` never changes across replicates, so ``X'X`` is computed once and each replicate is a
    single 11×11 solve.

    Args:
        design: (T, C + 10) the design at the selected hyperparameters.
        sales: (T,) observed sales, £k per week.
        surface: the fitted surface, for its adstocked spend and saturation.
        ridge_penalty: the selected penalty, applied unchanged to every replicate.
        seed: explicit, per CLAUDE.md rule 4.
        n_replicates: B.
        block: moving-block length in weeks.

    Returns:
        (B, C) contributions. Percentiles of this are the nominal 90% interval.
    """
    n_weeks = design.shape[0]
    n_media = surface.n_channels
    gram, _ = _normal_equations(design, sales)
    fitted_coefficients = _solve_bounded_ridge(gram, design.T @ sales, ridge_penalty, n_media)
    fitted = design @ fitted_coefficients
    residual = sales - fitted

    saturated_totals = np.asarray(
        [
            hill_saturation(
                surface.adstocked[:, index],
                float(surface.half_saturation[index]),
                float(surface.hill_shape[index]),
            ).sum()
            for index in range(n_media)
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)
    replicates = np.empty((n_replicates, n_media), dtype=np.float64)
    for replicate in range(n_replicates):
        resampled = fitted + residual[_moving_block_indices(n_weeks, block, rng)]
        coefficients = _solve_bounded_ridge(gram, design.T @ resampled, ridge_penalty, n_media)
        replicates[replicate] = coefficients[:n_media] * saturated_totals
    return replicates


class RidgeMMM:
    """The estimator under test. Sees spend and sales; never sees the truth.

    Usage is one call::

        fit = RidgeMMM().fit(spend, sales, seed=0)

    `fit`'s signature is the leakage guarantee: two arrays and a seed. There is no parameter
    through which a `SimResult`, a `DGPParams` or a latent series could arrive, so no
    expression inside can reach one.
    """

    def __init__(
        self,
        bounds: SearchBounds | None = None,
        n_draws: int = N_SEARCH_DRAWS,
        n_folds: int = N_CV_FOLDS,
        n_bootstrap: int = N_BOOTSTRAP_REPLICATES,
        block: int = BOOTSTRAP_BLOCK_WEEKS,
    ) -> None:
        if n_draws < 1:
            raise ValueError(f"n_draws must be at least 1; got {n_draws}")
        self.bounds = bounds if bounds is not None else SearchBounds()
        self.n_draws = n_draws
        self.n_folds = n_folds
        self.n_bootstrap = n_bootstrap
        self.block = block

    def fit(self, spend: NDArray[np.float64], sales: NDArray[np.float64], seed: int) -> MMMFit:
        """Fit by random search over the transform, then bootstrap at the winning draw.

        Args:
            spend: (T, C) weekly spend, £k per week. Observed.
            sales: (T,) weekly sales, £k per week. Observed, noise included.
            seed: explicit, per CLAUDE.md rule 4. The search and the bootstrap take
                independent streams spawned from it, so changing B cannot move the search.

        Returns:
            The fit, its interventional contributions and their bootstrap intervals.

        Raises:
            ValueError: on shape mismatches, or if no draw could be scored.
        """
        spend = np.asarray(spend, dtype=np.float64)
        sales = np.asarray(sales, dtype=np.float64)
        if spend.ndim != 2:
            raise ValueError(f"spend must be (T, C); got shape {spend.shape}")
        if sales.shape != (spend.shape[0],):
            raise ValueError(
                f"sales must be (T,) matching spend's {spend.shape[0]} weeks; "
                f"got shape {sales.shape}"
            )
        n_weeks, n_media = spend.shape
        folds = expanding_window_folds(n_weeks, self.n_folds)
        search_seed, bootstrap_seed = np.random.SeedSequence(seed).spawn(2)
        rng = np.random.Generator(np.random.PCG64(search_seed))

        best: tuple[float, Hyperparameters, NDArray[np.float64]] | None = None
        for _ in range(self.n_draws):
            hyper = _draw_hyperparameters(spend, self.bounds, rng)
            design = design_matrix(spend, hyper)
            score = _cross_validated_rmse(design, sales, hyper.ridge_penalty, n_media, folds)
            if best is None or score < best[0]:
                best = (score, hyper, design)
        if best is None:
            raise ValueError("no random-search draw was scored")
        cv_rmse, hyper, design = best

        gram, moment = _normal_equations(design, sales)
        coefficients = _solve_bounded_ridge(gram, moment, hyper.ridge_penalty, n_media)
        controls = control_matrix(n_weeks)
        surface = FittedSurface(
            adstocked=adstock_spend(spend, hyper.decay),
            coefficients=coefficients[:n_media],
            half_saturation=hyper.half_saturation,
            hill_shape=hyper.hill_shape,
            baseline=controls @ coefficients[n_media:],
            spend_totals=np.asarray(spend.sum(axis=0), dtype=np.float64),
        )
        replicates = bootstrap_contributions(
            design,
            sales,
            surface,
            hyper.ridge_penalty,
            seed=int(bootstrap_seed.generate_state(1)[0]),
            n_replicates=self.n_bootstrap,
            block=self.block,
        )
        tail = 100.0 * (1.0 - INTERVAL_LEVEL) / 2.0
        interval = np.column_stack(
            [
                np.percentile(replicates, tail, axis=0),
                np.percentile(replicates, 100.0 - tail, axis=0),
            ]
        )
        return MMMFit(
            hyperparameters=hyper,
            coefficients=coefficients,
            surface=surface,
            contribution=estimated_contribution(surface),
            contribution_interval=np.asarray(interval, dtype=np.float64),
            marginal_roas=estimated_marginal_roas(surface),
            cv_rmse=cv_rmse,
            n_draws_scored=self.n_draws,
        )


def recommended_allocation(surface: AllocatableSurface, seed: int) -> OptimalAllocation:
    """The allocation the model advises, solved exactly as the truth's optimum is — **D18**.

    Two arguments, deliberately. There is **no optimiser parameter here at all**: not
    `n_starts`, not `max_multiplier`, not the SLSQP tolerances. The configuration is whatever
    `truth.optimal_allocation` defaults to, which is the same configuration the truth solve
    runs under, and it stays that way because there is nothing here that could diverge from it.
    D18's requirement that the two be identical is therefore structural rather than a
    convention someone has to remember.

    That also makes the oracle-surface control meaningful: pass the *true* surface to this and
    the result must match `optimal_allocation` on it bit for bit, so any regret it produces is
    pure optimisation error and gets reported separately from the estimator's.
    """
    return optimal_allocation(surface, seed=seed)


def allocation_regret(
    true_surface: AllocatableSurface,
    recommended: NDArray[np.float64],
    true_optimum: OptimalAllocation,
) -> float:
    """§6's rung-3 metric: ``(S_opt − S_model) / (S_opt − S_status_quo)``, as a fraction.

    Every term is **noiseless true sales**. `recommended` is the model's multiplier vector, but
    it is evaluated here on the true surface — what the model predicted it would earn plays no
    part. Values above 1 mean the advice is worse than doing nothing and are **not clipped**:
    CLAUDE.md names silent clipping as a failure mode, and worse-than-nothing is the most
    informative outcome the study can produce.

    D19 travels with this number: the denominator is achievable lift, which runs from 1.10% of
    sales at C0 to 11.18% at C7, so regret is "share of what was achievable there" and not
    "damage done". Absolute lift lost as a share of total sales must be reported alongside it.

    Raises:
        ValueError: if achievable lift is not positive, which would make the ratio meaningless
            rather than large.
    """
    lift = true_optimum.achievable_lift
    if not lift > 0.0:
        raise ValueError(
            f"achievable lift is {lift:.6g} £k, so regret has no denominator. The optimum is "
            f"not above the status quo and the solve, not the allocation, is what is wrong."
        )
    return float((true_optimum.total_sales - true_surface.total_sales(recommended)) / lift)
