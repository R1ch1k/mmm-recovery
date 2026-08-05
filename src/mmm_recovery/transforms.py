"""Adstock and saturation transforms for the media response curve.

Two variants of each, exactly as tabulated in PREREGISTRATION.md §2:

* the **correct** forms, which the estimator also assumes — geometric adstock, Hill
  saturation;
* the **misspecified truths** used to generate data under conditions C4 and C7 —
  Weibull-PDF adstock, logistic saturation.

Application order is always adstock then saturation, matching Robyn and Meridian
convention. Every function here is pure: same inputs, same outputs, no state, no RNG,
no mutation of its arguments.

Units and shapes
----------------
Spend is £k per week. Every series argument is a 1-D array of length T (weeks), and
every return value is a 1-D array of length T. Adstock kernels are normalised to sum
to 1, so adstocked spend carries the same units as raw spend and the half-saturation
point κ is directly comparable to a weekly spend level — which is what makes the
κ column of the prereg's channel table (e.g. TV: κ=60 against mean spend 55) meaningful.
Saturation returns a dimensionless fraction in [0, 1); the channel's β (£k) scales it.

Adstock is causal with a cold start: weeks before t=0 are treated as zero spend, so the
first few weeks of any simulated series carry less carryover than the steady state.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter
from scipy.special import expit


def _validated_series(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Coerce to a 1-D float64 array and reject anything a spend series cannot be."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D series of weekly values; got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite; found nan or inf")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative; got a minimum of {arr.min()}")
    return arr


def geometric_adstock(spend: NDArray[np.float64], decay: float) -> NDArray[np.float64]:
    """Geometric adstock with a unit-sum kernel — the *correct* carryover form.

    PREREGISTRATION.md §2 states the recursion ``x̃_t = x_t + λ·x̃_{t-1}``, whose implied
    kernel is ``λ^k``. That kernel sums to ``1/(1-λ)``, so normalising it to sum to 1 —
    which the prereg also requires — scales the whole thing by ``(1-λ)``, giving the
    equivalent recursion implemented here::

        x̃_t = (1-λ)·x_t + λ·x̃_{t-1},    x̃_{-1} = 0

    The impulse response is therefore ``(1-λ)·λ^k``: it decays at exactly λ per week and
    sums to 1 over an infinite horizon. Unlike the Weibull form below the kernel is not
    truncated; it is applied as an IIR recursion, so no lag budget has to be chosen.

    Args:
        spend: (T,) weekly spend, £k per week, finite and non-negative.
        decay: λ, the weekly carryover fraction, in [0, 1). λ = 1 is rejected because
            the un-normalised kernel diverges and the unit-sum scaling is undefined.

    Returns:
        (T,) adstocked spend, £k per week.
    """
    series = _validated_series(spend, "spend")
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay (λ) must lie in [0, 1); got {decay}")
    return np.asarray(lfilter([1.0 - decay], [1.0, -decay], series), dtype=np.float64)


def _weibull_kernel(peak_lag: float, shape: float, max_lag: int) -> NDArray[np.float64]:
    """Unit-sum Weibull-PDF lag kernel, evaluated at integer lags 0, 1, ..., max_lag.

    The Weibull density ``f(t) = (k/s)·(t/s)^(k-1)·exp(-(t/s)^k)`` has its mode at
    ``t* = s·((k-1)/k)^(1/k)`` for shape ``k > 1``, so pinning the peak at ``peak_lag``
    fixes the scale at ``s = peak_lag / ((k-1)/k)^(1/k)``. For the prereg's C4 settings
    (peak 2, shape 2.0) that is ``s = 2·√2 ≈ 2.828``.

    Returns:
        (max_lag + 1,) weights summing to 1, indexed by lag in weeks.
    """
    scale = peak_lag / ((shape - 1.0) / shape) ** (1.0 / shape)
    lags = np.arange(max_lag + 1, dtype=np.float64)
    scaled = lags / scale
    density = (shape / scale) * scaled ** (shape - 1.0) * np.exp(-(scaled**shape))
    total = float(density.sum())
    if not total > 0.0:
        raise ValueError(
            f"Weibull kernel has no mass over lags 0..{max_lag} "
            f"for peak_lag={peak_lag}, shape={shape}"
        )
    return np.asarray(density / total, dtype=np.float64)


def weibull_adstock(
    spend: NDArray[np.float64], peak_lag: float, shape: float, max_lag: int
) -> NDArray[np.float64]:
    """Weibull-PDF adstock — the *misspecified* carryover truth used by C4 and C7.

    A delayed-peak kernel: unlike geometric adstock, which puts its largest weight on the
    current week, this one builds to a maximum at ``peak_lag`` weeks. Under C4 the data are
    generated with this while the estimator still assumes the geometric form, which is the
    functional-form misspecification the condition is designed to test.

    Lag convention (see the Deviations Log discussion — this is an interpretation of a
    detail the prereg leaves open): the kernel is evaluated at integer lags 0..max_lag
    inclusive, so "truncated at lag 12" means a 13-tap kernel. Because the Weibull density
    is zero at t=0 for any shape > 1, lag 0 carries **exactly zero** weight — a channel
    transformed this way has no same-week effect at all.

    Args:
        spend: (T,) weekly spend, £k per week, finite and non-negative.
        peak_lag: the lag in weeks at which the continuous kernel is maximised. Must be
            positive and no greater than max_lag, else the peak falls outside the window.
        shape: Weibull shape k. Must be > 1; at k <= 1 the density has no interior mode
            and the peak-to-scale mapping is undefined.
        max_lag: last lag retained, in weeks. The prereg fixes this at 12.

    Returns:
        (T,) adstocked spend, £k per week.
    """
    series = _validated_series(spend, "spend")
    if shape <= 1.0:
        raise ValueError(f"shape (Weibull k) must be > 1 for an interior peak; got {shape}")
    if peak_lag <= 0.0:
        raise ValueError(f"peak_lag must be positive; got {peak_lag}")
    if max_lag < 1:
        raise ValueError(f"max_lag must be at least 1 week; got {max_lag}")
    if peak_lag > max_lag:
        raise ValueError(f"peak_lag {peak_lag} falls outside the retained window 0..{max_lag}")
    kernel = _weibull_kernel(peak_lag, shape, max_lag)
    return np.asarray(np.convolve(series, kernel)[: series.size], dtype=np.float64)


def hill_saturation(
    adstocked: NDArray[np.float64], half_saturation: float, shape: float
) -> NDArray[np.float64]:
    """Hill saturation — the *correct* diminishing-returns form.

    ``h(x) = x^α / (x^α + κ^α)``, so ``h(0) = 0``, ``h(κ) = 0.5`` and ``h → 1`` as spend
    grows. For α <= 1 the curve is concave everywhere (diminishing returns from the first
    pound); for α > 1 it is S-shaped — convex below κ, concave above — which is the shape
    the prereg assigns to TV (α=1.8) and OOH (α=2.2).

    Args:
        adstocked: (T,) adstocked spend, £k per week, finite and non-negative.
        half_saturation: κ in £k per week — the adstocked spend at which the channel
            reaches half its maximum response. Must be positive.
        shape: α, the Hill exponent. Must be positive.

    Returns:
        (T,) dimensionless response fraction in [0, 1).
    """
    series = _validated_series(adstocked, "adstocked")
    if half_saturation <= 0.0:
        raise ValueError(f"half_saturation (κ) must be positive; got {half_saturation}")
    if shape <= 0.0:
        raise ValueError(f"shape (α) must be positive; got {shape}")
    powered = series**shape
    return np.asarray(powered / (powered + half_saturation**shape), dtype=np.float64)


HILL_GRADIENT_FLOOR = 1e-12
"""Fraction of κ below which the Hill gradient is evaluated at a floor rather than at zero.

For α < 1 the Hill curve has an infinite slope at zero spend, which is mathematically correct
and useless to an optimiser: it returns an inf that no line search can work with. The floor
keeps the gradient finite and pointing the right way — upward. It bites only exactly at zero
adstocked spend, and only for α < 1.
"""


def hill_saturation_derivative(
    adstocked: NDArray[np.float64], half_saturation: float, shape: float
) -> NDArray[np.float64]:
    """d(`hill_saturation`)/d(adstocked spend), in closed form.

    ``h'(x) = α·κ^α·x^(α-1) / (x^α + κ^α)²``. Lives here, next to the function it
    differentiates, because both the true response surface and the estimator's fitted surface
    need it and a gradient formula written out twice is exactly the defect that produces a
    converged-looking solve at the wrong point.

    Args:
        adstocked: (T,) adstocked spend, £k per week.
        half_saturation: κ in £k per week. Must be positive.
        shape: α, the Hill exponent. Must be positive.

    Returns:
        (T,) derivative, per £k per week.
    """
    series = _validated_series(adstocked, "adstocked")
    if half_saturation <= 0.0:
        raise ValueError(f"half_saturation (κ) must be positive; got {half_saturation}")
    if shape <= 0.0:
        raise ValueError(f"shape (α) must be positive; got {shape}")
    floored = np.maximum(series, HILL_GRADIENT_FLOOR * half_saturation)
    powered = floored**shape
    half_powered = half_saturation**shape
    return np.asarray(
        shape * half_powered * floored ** (shape - 1.0) / (powered + half_powered) ** 2,
        dtype=np.float64,
    )


def logistic_saturation(
    adstocked: NDArray[np.float64], half_saturation: float, scale: float
) -> NDArray[np.float64]:
    """Logistic saturation, zero-anchored — the *misspecified* truth for C4 and C7.

    Deviation **D3**. The raw logistic ``g(x) = 1/(1 + exp(-(x - κ)/s))`` is strictly
    positive at zero spend, so as literally specified in PREREGISTRATION.md §2 every
    channel would emit sales while spending nothing — about £3.96k/week for TV at the D2
    scale of s = κ/4. What is implemented is therefore the zero-anchored form::

        sat(x) = (g(x) - g(0)) / (1 - g(0))

    which maps 0 to 0 exactly, still spans [0, 1], and keeps the S-shape. D3's reasoning:
    C4 is meant to test misspecification of curve *shape*, and an additive offset would
    have made it test two things at once.

    The anchoring is exact, not approximate. ``g(0)`` is subtracted using the identical
    expression that produces ``g(x)`` at x = 0, so the two cancel bit for bit and a zeroed
    channel contributes exactly nothing. `dgp.py` depends on that: with all spend zeroed,
    noiseless sales must equal the baseline exactly.

    Note that κ is no longer the half-response point — it is the inflection point of the
    underlying logistic. The anchored curve reaches ``(0.5 - g(0))/(1 - g(0))`` there,
    which is 0.482 at s = κ/4.

    Args:
        adstocked: (T,) adstocked spend, £k per week, finite and non-negative.
        half_saturation: κ in £k per week — the inflection point of the underlying curve.
        scale: s in £k per week, the width of the transition. Must be positive; D2 fixes
            it at κ/4. Smaller values push the curve toward a step function at κ, and past
            roughly κ + 37·s the float64 result is exactly 1.0 with an exactly zero
            gradient — which is why D2 requires `truth.py` to assert the optimiser's
            spend range stays clear of it.

    Returns:
        (T,) dimensionless response fraction in [0, 1], exactly 0 at zero spend.
    """
    series = _validated_series(adstocked, "adstocked")
    if scale <= 0.0:
        raise ValueError(f"scale (s) must be positive; got {scale}")
    response = expit((series - half_saturation) / scale)
    at_zero = expit((0.0 - half_saturation) / scale)
    return np.asarray((response - at_zero) / (1.0 - at_zero), dtype=np.float64)
