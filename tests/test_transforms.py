"""Tests for the adstock and saturation transforms (CLAUDE.md build order, Step 1).

Where a property can be checked two ways, it is: `geometric_adstock` is implemented as an
IIR filter and verified against a hand-written recursion, so a bug would have to be made
twice, in two different idioms, to survive.

The (λ, α, κ) triples below are the channel truths tabulated in PREREGISTRATION.md §2.
They live here for now; Step 2 moves them into a dataclass in `dgp.py` and this module
will import them instead of restating them.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from mmm_recovery.transforms import (
    geometric_adstock,
    hill_saturation,
    logistic_saturation,
    weibull_adstock,
)

# (channel, λ decay, α Hill shape, κ half-saturation in £k/week)
CHANNEL_TRUTHS = [
    ("tv", 0.70, 1.8, 60.0),
    ("video", 0.45, 1.2, 30.0),
    ("search", 0.10, 0.9, 18.0),
    ("social", 0.30, 1.0, 22.0),
    ("ooh", 0.60, 2.2, 25.0),
    ("placebo", 0.30, 1.0, 20.0),
]
DECAYS = [decay for _, decay, _, _ in CHANNEL_TRUTHS] + [0.0, 0.95]


def unit_impulse(length: int) -> NDArray[np.float64]:
    """One unit of spend in week 0 and nothing afterwards; the response is the kernel."""
    impulse = np.zeros(length, dtype=np.float64)
    impulse[0] = 1.0
    return impulse


def spend_series(seed: int = 20260805, length: int = 260) -> NDArray[np.float64]:
    """A stand-in weekly spend series, £k per week. Log-normal, strictly positive."""
    rng = np.random.default_rng(seed)
    return np.asarray(rng.lognormal(mean=3.0, sigma=0.4, size=length), dtype=np.float64)


# --------------------------------------------------------------------------------------
# Geometric adstock
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("decay", [d for d in DECAYS if d > 0.0])
def test_geometric_impulse_decays_at_exactly_lambda(decay: float) -> None:
    """Consecutive impulse-response weights differ by a factor of exactly λ."""
    response = geometric_adstock(unit_impulse(200), decay)
    ratios = response[1:100] / response[:99]
    np.testing.assert_allclose(ratios, decay, rtol=1e-12, atol=0.0)


def test_geometric_with_zero_decay_is_the_identity() -> None:
    spend = spend_series()
    np.testing.assert_allclose(geometric_adstock(spend, 0.0), spend, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("decay", DECAYS)
def test_geometric_kernel_sums_to_one(decay: float) -> None:
    """The unit-sum normalisation the prereg requires, measured on the impulse response.

    The horizon is long enough that the truncated tail is below the tolerance even at
    λ=0.95, where λ^2000 is of order 1e-45.
    """
    response = geometric_adstock(unit_impulse(2000), decay)
    assert float(response.sum()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("decay", DECAYS)
def test_geometric_adstock_is_the_prereg_recursion_rescaled(decay: float) -> None:
    """Independent check of the interpretive step taken in `geometric_adstock`.

    PREREGISTRATION.md §2 writes the recursion as ``x̃_t = x_t + λ·x̃_{t-1}`` and separately
    requires the kernel to sum to 1. Those two together mean the implemented filter must
    equal the literal recursion scaled by (1-λ). This test computes the literal recursion
    with an explicit loop — a different implementation route from the IIR filter used in
    the module — and asserts the identity.
    """
    spend = spend_series()
    literal = np.empty_like(spend)
    carry = 0.0
    for t, x in enumerate(spend):
        carry = x + decay * carry
        literal[t] = carry
    np.testing.assert_allclose(
        geometric_adstock(spend, decay), (1.0 - decay) * literal, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("decay", DECAYS)
def test_geometric_adstock_never_exceeds_total_spend(decay: float) -> None:
    """A unit-sum causal kernel can move spend later in time but cannot create any."""
    spend = spend_series()
    adstocked = geometric_adstock(spend, decay)
    assert float(adstocked.sum()) <= float(spend.sum()) + 1e-9


# --------------------------------------------------------------------------------------
# Weibull adstock
# --------------------------------------------------------------------------------------


def test_weibull_kernel_peaks_at_lag_two() -> None:
    """The C4 misspecification setting: peak lag 2, shape 2.0, truncated at lag 12."""
    response = weibull_adstock(unit_impulse(60), peak_lag=2.0, shape=2.0, max_lag=12)
    assert int(np.argmax(response)) == 2
    # Strictly the maximum, not tied with a neighbour.
    assert response[2] > response[1]
    assert response[2] > response[3]


@pytest.mark.parametrize("peak_lag", [1.0, 2.0, 3.0, 4.0, 6.0])
def test_weibull_kernel_peaks_where_asked(peak_lag: float) -> None:
    response = weibull_adstock(unit_impulse(60), peak_lag=peak_lag, shape=2.0, max_lag=12)
    assert int(np.argmax(response)) == int(peak_lag)


def test_weibull_kernel_sums_to_one_and_truncates_at_max_lag() -> None:
    max_lag = 12
    response = weibull_adstock(unit_impulse(60), peak_lag=2.0, shape=2.0, max_lag=max_lag)
    assert float(response.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float(response[: max_lag + 1].sum()) == pytest.approx(1.0, abs=1e-12)
    assert np.all(response[max_lag + 1 :] == 0.0)


def test_weibull_kernel_has_no_same_week_effect() -> None:
    """Lag 0 weight is exactly zero: the Weibull density vanishes at t=0 for shape > 1.

    This is a real modelling consequence of the delayed-peak form, not a rounding artefact,
    and it is one of the ways C4 data differ from what the geometric estimator assumes.
    """
    response = weibull_adstock(unit_impulse(60), peak_lag=2.0, shape=2.0, max_lag=12)
    assert response[0] == 0.0


def test_weibull_adstock_never_exceeds_total_spend() -> None:
    spend = spend_series()
    adstocked = weibull_adstock(spend, peak_lag=2.0, shape=2.0, max_lag=12)
    assert float(adstocked.sum()) <= float(spend.sum()) + 1e-9


# --------------------------------------------------------------------------------------
# Saturation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("channel", "half_saturation", "shape"),
    [(name, kappa, alpha) for name, _, alpha, kappa in CHANNEL_TRUTHS],
)
def test_hill_maps_zero_to_zero(channel: str, half_saturation: float, shape: float) -> None:
    result = hill_saturation(np.array([0.0]), half_saturation=half_saturation, shape=shape)
    assert result[0] == 0.0


@pytest.mark.parametrize(
    ("half_saturation", "shape"), [(kappa, alpha) for _, _, alpha, kappa in CHANNEL_TRUTHS]
)
def test_hill_is_strictly_monotone_increasing(half_saturation: float, shape: float) -> None:
    grid = np.linspace(0.0, 400.0, 801)
    response = hill_saturation(grid, half_saturation=half_saturation, shape=shape)
    assert np.all(np.diff(response) > 0.0)
    assert np.all(response >= 0.0)
    assert np.all(response < 1.0)


@pytest.mark.parametrize("shape", [0.5, 0.9, 1.0])
def test_hill_is_concave_when_shape_at_most_one(shape: float) -> None:
    """α <= 1 means diminishing returns from the first pound: the curve never bends up."""
    grid = np.linspace(0.0, 400.0, 801)
    response = hill_saturation(grid, half_saturation=25.0, shape=shape)
    second_difference = np.diff(response, n=2)
    assert np.all(second_difference < 0.0)


@pytest.mark.parametrize(("shape", "half_saturation"), [(1.8, 60.0), (2.2, 25.0)])
def test_hill_is_s_shaped_when_shape_above_one(shape: float, half_saturation: float) -> None:
    """α > 1 puts an inflection at κ·((α-1)/(α+1))^(1/α): convex below it, concave above.

    TV (α=1.8) and OOH (α=2.2) both sit here, so the DGP's response curve genuinely has a
    take-off region — worth pinning down, because a concave-everywhere curve would make the
    allocation problem easier than the prereg intends.
    """
    inflection = half_saturation * ((shape - 1.0) / (shape + 1.0)) ** (1.0 / shape)
    grid = np.linspace(0.0, 400.0, 4001)
    second_difference = np.diff(hill_saturation(grid, half_saturation, shape), n=2)
    midpoints = grid[1:-1]
    assert np.all(second_difference[midpoints < 0.9 * inflection] > 0.0)
    assert np.all(second_difference[midpoints > 1.1 * inflection] < 0.0)


def test_hill_reaches_one_half_at_the_half_saturation_point() -> None:
    for _, _, shape, half_saturation in CHANNEL_TRUTHS:
        value = hill_saturation(np.array([half_saturation]), half_saturation, shape)
        assert float(value[0]) == pytest.approx(0.5, abs=1e-12)


def test_logistic_does_not_map_zero_to_zero() -> None:
    """Documents a conflict between PREREGISTRATION.md §2 and the CLAUDE.md Step 1 test list.

    The prereg defines logistic saturation as ``1/(1 + exp(-(x-κ)/s))``, which is strictly
    positive at x=0, while CLAUDE.md asks every saturation to map 0 to 0. The prereg is the
    binding document, so the literal form is what is implemented, and this test pins the
    non-zero intercept in place so that any future shift to a zero-anchored variant has to
    be a deliberate, logged change rather than a silent one.
    """
    half_saturation, scale = 25.0, 8.0
    value = float(logistic_saturation(np.array([0.0]), half_saturation, scale)[0])
    assert value == pytest.approx(1.0 / (1.0 + np.exp(half_saturation / scale)), rel=1e-12)
    assert value > 0.0


@pytest.mark.parametrize("scale", [1.0, 5.0, 8.0, 20.0])
def test_logistic_is_monotone_and_hits_one_half_at_kappa(scale: float) -> None:
    grid = np.linspace(0.0, 400.0, 801)
    response = logistic_saturation(grid, half_saturation=25.0, scale=scale)
    assert np.all(response > 0.0)
    assert np.all(np.diff(response) >= 0.0)
    # Strictly increasing while the curve is still meaningfully below its asymptote. Closer
    # in than that, neighbouring grid points collapse onto the same float64 — the spacing
    # just below 1.0 is 2.2e-16 — so ties there are a representation limit, not a property
    # of the function. See test_logistic_gradient_dies_once_float64_saturates.
    assert np.all(np.diff(response[response < 1.0 - 1e-9]) > 0.0)
    assert float(logistic_saturation(np.array([25.0]), 25.0, scale)[0]) == pytest.approx(0.5)


@pytest.mark.parametrize("scale", [0.5, 1.0, 8.0])
def test_logistic_gradient_dies_once_float64_saturates(scale: float) -> None:
    """An exactly-flat region, which will matter to the Step 3 optimiser.

    In float64, ``expit(z)`` returns exactly 1.0 once z exceeds about 36.7, so logistic
    saturation is *exactly* constant beyond roughly ``κ + 37·s`` — not merely nearly
    constant. A channel pushed past that point has a true marginal return of exactly zero
    and SLSQP sees a zero gradient rather than a small one. With κ=25 that flat region
    begins around 43 £k/week when s=0.5, but not until around 320 £k/week when s=8. Since
    PREREGISTRATION.md fixes no value for s, the choice decides whether the flat region is
    reachable at all inside the optimiser's m_c ∈ [0, 3] range.
    """
    half_saturation = 25.0
    flat_from = half_saturation + 40.0 * scale
    response = logistic_saturation(
        np.array([flat_from, 2.0 * flat_from, 10.0 * flat_from]), half_saturation, scale
    )
    assert np.all(response == 1.0)
    assert np.all(np.diff(response) == 0.0)


def test_logistic_is_stable_far_into_both_tails() -> None:
    """expit rather than a naive exp: no overflow warnings, no nan, at extreme arguments."""
    extreme = np.array([0.0, 1e3, 1e6, 1e12])
    response = logistic_saturation(extreme, half_saturation=25.0, scale=0.5)
    assert np.all(np.isfinite(response))
    assert np.all(np.diff(response) >= 0.0)


# --------------------------------------------------------------------------------------
# Purity and input validation
# --------------------------------------------------------------------------------------


def test_transforms_are_pure_and_repeatable() -> None:
    """CLAUDE.md rule 4: same inputs, same outputs, bit for bit, and no argument mutation."""
    spend = spend_series()
    original = spend.copy()
    for first, second in [
        (geometric_adstock(spend, 0.7), geometric_adstock(spend, 0.7)),
        (
            weibull_adstock(spend, 2.0, 2.0, 12),
            weibull_adstock(spend, 2.0, 2.0, 12),
        ),
        (hill_saturation(spend, 60.0, 1.8), hill_saturation(spend, 60.0, 1.8)),
        (logistic_saturation(spend, 25.0, 8.0), logistic_saturation(spend, 25.0, 8.0)),
    ]:
        assert np.array_equal(first, second)
    np.testing.assert_array_equal(spend, original)


@pytest.mark.parametrize(
    "bad_series",
    [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([], dtype=np.float64),
        np.array([1.0, -2.0, 3.0]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
    ids=["two-dimensional", "empty", "negative", "nan", "inf"],
)
def test_every_transform_rejects_an_invalid_series(bad_series: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        geometric_adstock(bad_series, 0.7)
    with pytest.raises(ValueError):
        weibull_adstock(bad_series, 2.0, 2.0, 12)
    with pytest.raises(ValueError):
        hill_saturation(bad_series, 60.0, 1.8)
    with pytest.raises(ValueError):
        logistic_saturation(bad_series, 25.0, 8.0)


@pytest.mark.parametrize("decay", [-0.01, 1.0, 1.5])
def test_geometric_rejects_decay_outside_unit_interval(decay: float) -> None:
    with pytest.raises(ValueError, match="decay"):
        geometric_adstock(spend_series(length=10), decay)


@pytest.mark.parametrize(
    ("peak_lag", "shape", "max_lag"),
    [
        (2.0, 1.0, 12),
        (2.0, 0.5, 12),
        (0.0, 2.0, 12),
        (-1.0, 2.0, 12),
        (2.0, 2.0, 0),
        (13.0, 2.0, 12),
    ],
    ids=[
        "shape-one",
        "shape-below-one",
        "zero-peak",
        "negative-peak",
        "no-lags",
        "peak-past-window",
    ],
)
def test_weibull_rejects_impossible_kernels(peak_lag: float, shape: float, max_lag: int) -> None:
    with pytest.raises(ValueError):
        weibull_adstock(spend_series(length=30), peak_lag, shape, max_lag)


@pytest.mark.parametrize(
    ("half_saturation", "shape"), [(0.0, 1.8), (-1.0, 1.8), (60.0, 0.0), (60.0, -1.0)]
)
def test_hill_rejects_non_positive_parameters(half_saturation: float, shape: float) -> None:
    with pytest.raises(ValueError):
        hill_saturation(spend_series(length=10), half_saturation, shape)


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_logistic_rejects_non_positive_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        logistic_saturation(spend_series(length=10), 25.0, scale)
