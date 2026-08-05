"""The generating process — the world in which the truth is known.

`simulate(params, seed)` is a pure function: the same parameters and seed produce the same
arrays, bit for bit, every time and on any worker. Nothing here reads a clock, a global RNG,
or a dict ordering.

Structure, following PREREGISTRATION.md §2::

    sales_t       = baseline_t + Σ_c contrib_{c,t} + ε_t
    baseline_t    = B0 · (1 + τ·t/T) · season_t · exp(γ · d_t)
    contrib_{c,t} = β_c · sat_c( adstock_c( spend_{c,t} ) )

The separation that matters is between `simulate`, which draws random numbers, and
`evaluate`, which does not. `evaluate` maps a spend matrix to a noiseless response given a
fixed demand and seasonal path, so Step 3 can re-run it under `do(spend_c := 0)` and read
the causal truth straight off the generating process rather than off a fitted parameter.

Latent demand `d_t` is returned in `SimResult` because the *study* needs it for diagnostics.
It must never reach the estimator; `estimator.py` asserts that at Step 4.

Units: spend and sales are £k per week. Spend matrices are (T, C), series are (T,).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.signal import lfilter

from mmm_recovery.transforms import (
    geometric_adstock,
    hill_saturation,
    logistic_saturation,
    weibull_adstock,
)

WEEKS_PER_YEAR = 52
WEEKS_PER_QUARTER = 13
_AR_BURN_IN = 500
_PLACEBO_WEIGHT_CEILING = 1.0 / np.sqrt(2.0)

C0_MEDIA_SHARE_TARGET = 0.25
"""§2's calibration target, which D9 restores by correcting B0."""

CALIBRATION_SEEDS = 200
"""Seeds the B0 solve runs over — the same count C0 uses in the grid."""

BASELINE_LEVEL = 682.5179
"""B0 in £k per week (D9), the numerical solution of ``solve_baseline_level()``.

§2's stated B0 of 1000 gives a C0 media share of 18.57%, not the ≈25% the same section
claims; the two were over-determined together with the channel table. β is locked by D4, so
B0 is the only lever that moves the ratio without touching ground truth — baseline is exactly
linear in B0, and spend and contributions are bit-identical across it.

The constant is committed rather than solved at import so that `simulate` stays fast and
deterministic; `test_the_committed_baseline_level_reproduces_the_solve` re-runs the solver and
checks it lands here, so the value is derived rather than asserted.
"""


@dataclass(frozen=True)
class Channel:
    """One media channel's ground truth, from the PREREGISTRATION.md §2 table.

    Attributes:
        name: channel identifier.
        decay: λ, weekly geometric carryover in [0, 1).
        hill_shape: α, the Hill exponent.
        half_saturation: κ in £k per week.
        beta: maximum weekly contribution in £k. Exactly 0.0 for the placebo.
        mean_spend: target mean weekly spend in £k.
        cycle_phase: offset in weeks of this channel's quarterly budget cycle. Fixed per
            channel rather than derived from its position, so that adding the placebo cannot
            shift TV's buying cycle — a channel's spend process must not depend on how many
            other channels happen to exist.
    """

    name: str
    decay: float
    hill_shape: float
    half_saturation: float
    beta: float
    mean_spend: float
    cycle_phase: float


REAL_CHANNELS: tuple[Channel, ...] = (
    Channel("tv", 0.70, 1.8, 60.0, 220.0, 55.0, 0.0),
    Channel("video", 0.45, 1.2, 30.0, 120.0, 28.0, 2.0),
    Channel("search", 0.10, 0.9, 18.0, 90.0, 20.0, 4.0),
    Channel("social", 0.30, 1.0, 22.0, 70.0, 18.0, 6.0),
    Channel("ooh", 0.60, 2.2, 25.0, 60.0, 12.0, 8.0),
)
"""The five channels with a real effect. Present in every condition."""

PLACEBO = Channel("placebo", 0.30, 1.0, 20.0, 0.0, 15.0, 10.0)
"""β is exactly 0.0, not approximately. Present in C5, C6 and C7 only."""


@dataclass(frozen=True)
class DGPParams:
    """Every knob of the generating process. One condition moves one of these.

    Defaults are C0. Values the pre-registration fixes are named for the symbol it uses;
    values it left open carry a default chosen here and are listed in the Step 2 assumptions.
    """

    channels: tuple[Channel, ...] = REAL_CHANNELS
    n_weeks: int = 520
    """T, weeks of observation."""

    baseline_level: float = BASELINE_LEVEL
    """B0, £k per week. D9-corrected from §2's 1000; see `BASELINE_LEVEL`."""

    trend: float = 0.15
    """τ, fractional baseline growth across the full span."""

    noise_fraction: float = 0.03
    """σ as a fraction of mean noiseless sales."""

    season_coefficients: tuple[float, float, float, float] = (0.10, 0.05, 0.03, -0.02)
    """(a1, b1, a2, b2) for two Fourier pairs at period 52, added to a base of 1."""

    demand_ar: float = 0.8
    """φ_AR of the latent demand process."""

    demand_sd: float = 0.25
    """In-sample standard deviation of d_t. Not fixed by the prereg — see assumptions."""

    demand_seasonal_share: float = 0.3
    """Share of d_t's variance carried by its annual sinusoid rather than the AR(1) part."""

    demand_coefficient: float = 0.0
    """γ, the elasticity of baseline to latent demand. 0.5 under C3 and C7."""

    endogeneity: float = 0.0
    """φ, the elasticity of spend to latent demand. The confounding channel."""

    collinearity: float | None = None
    """ρ target for mean pairwise spend correlation on levels (D5). None leaves it free."""

    spend_log_sd: float = 0.30
    """Log-scale volatility of weekly spend."""

    quarterly_amplitude: float = 0.06
    """Amplitude of each channel's own quarterly budget cycle."""

    placebo_coupling: float | None = None
    """D1 target: equal correlation of placebo spend with search spend and with season."""

    misspecified: bool = False
    """C4 and C7: the truth uses Weibull adstock and logistic saturation instead."""

    weibull_peak_lag: float = 2.0
    weibull_shape: float = 2.0
    weibull_max_lag: int = 12

    logistic_scale_ratio: float = 0.25
    """D2: s_c = κ_c / 4."""

    def __post_init__(self) -> None:
        if self.n_weeks < WEEKS_PER_YEAR:
            raise ValueError(f"n_weeks must cover at least one seasonal cycle; got {self.n_weeks}")
        if not self.channels:
            raise ValueError("at least one channel is required")
        names = [channel.name for channel in self.channels]
        if len(set(names)) != len(names):
            raise ValueError(f"channel names must be unique; got {names}")
        if self.baseline_level <= 0.0:
            raise ValueError(f"baseline_level (B0) must be positive; got {self.baseline_level}")
        if not 0.0 <= self.demand_ar < 1.0:
            raise ValueError(f"demand_ar must lie in [0, 1); got {self.demand_ar}")
        if self.demand_sd <= 0.0:
            raise ValueError(f"demand_sd must be positive; got {self.demand_sd}")
        if not 0.0 <= self.demand_seasonal_share <= 1.0:
            raise ValueError(
                f"demand_seasonal_share must lie in [0, 1]; got {self.demand_seasonal_share}"
            )
        if self.spend_log_sd <= 0.0:
            raise ValueError(f"spend_log_sd must be positive; got {self.spend_log_sd}")
        if not 0.0 <= self.quarterly_amplitude < 1.0:
            raise ValueError(
                f"quarterly_amplitude must lie in [0, 1); got {self.quarterly_amplitude}"
            )
        if self.noise_fraction < 0.0:
            raise ValueError(f"noise_fraction must be non-negative; got {self.noise_fraction}")
        if self.collinearity is not None and not 0.0 <= self.collinearity < 1.0:
            raise ValueError(f"collinearity (ρ) must lie in [0, 1); got {self.collinearity}")
        if self.placebo_coupling is not None:
            if not 0.0 <= self.placebo_coupling < _PLACEBO_WEIGHT_CEILING:
                raise ValueError(
                    f"placebo_coupling must lie in [0, {_PLACEBO_WEIGHT_CEILING:.4f}); "
                    f"got {self.placebo_coupling}. Equal correlation with two near-orthogonal "
                    f"series cannot exceed 1/sqrt(2) — this is the D1 feasibility ceiling."
                )
            if "placebo" not in names:
                raise ValueError("placebo_coupling set but no channel named 'placebo'")
        if float(_seasonality(self.n_weeks, self.season_coefficients).min()) <= 0.0:
            raise ValueError("season_coefficients drive the seasonal multiplier non-positive")


@dataclass(frozen=True)
class Response:
    """The noiseless world: what sales would be for a given spend matrix.

    Attributes:
        baseline: (T,) £k per week, media excluded.
        contributions: (T, C) £k per week, per channel.
        noiseless_sales: (T,) £k per week, baseline plus all contributions, no ε.
    """

    baseline: NDArray[np.float64]
    contributions: NDArray[np.float64]
    noiseless_sales: NDArray[np.float64]


@dataclass(frozen=True)
class SimResult:
    """One simulated dataset plus the latent series the estimator must never see."""

    channel_names: tuple[str, ...]
    spend: NDArray[np.float64]
    """(T, C) £k per week."""

    sales: NDArray[np.float64]
    """(T,) £k per week, with ε. This and `spend` are all a real analyst would have."""

    noiseless_sales: NDArray[np.float64]
    baseline: NDArray[np.float64]
    contributions: NDArray[np.float64]
    demand: NDArray[np.float64]
    """(T,) latent demand d_t. Diagnostics only — never an estimator input."""

    season: NDArray[np.float64]
    mix_weight: float
    """Solved weight on the shared budget factor; 0 when ρ was left free."""

    placebo_weight: float
    """Solved weight for the D1 placebo coupling; 0 when there is no placebo."""

    params: DGPParams = field(repr=False)


def _seasonality(
    n_weeks: int, coefficients: tuple[float, float, float, float]
) -> NDArray[np.float64]:
    """(T,) multiplicative seasonal index, two Fourier pairs at period 52, centred on 1."""
    weeks = np.arange(n_weeks, dtype=np.float64)
    season = np.ones(n_weeks, dtype=np.float64)
    for harmonic, (cosine, sine) in enumerate(
        ((coefficients[0], coefficients[1]), (coefficients[2], coefficients[3])), start=1
    ):
        angle = 2.0 * np.pi * harmonic * weeks / WEEKS_PER_YEAR
        season += cosine * np.cos(angle) + sine * np.sin(angle)
    return season


def _standardise(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Zero mean, unit in-sample standard deviation."""
    spread = float(values.std())
    if spread <= 0.0:
        raise ValueError("cannot standardise a constant series")
    return np.asarray((values - values.mean()) / spread, dtype=np.float64)


def _latent_demand(params: DGPParams, rng: np.random.Generator) -> NDArray[np.float64]:
    """(T,) latent demand: AR(1) at φ_AR plus an annual sinusoid, standardised to demand_sd.

    Standardising in sample is deliberate. The prereg fixes φ_AR but not the innovation
    scale, so without it γ and φ would have no interpretable magnitude; with it, γ=0.5 means
    "a one-standard-deviation demand shock multiplies baseline by exp(0.5·demand_sd)".

    The sinusoid matters for what the study measures: it is the part of latent demand the
    estimator's Fourier controls *can* absorb, leaving the AR(1) part as the genuinely
    unobservable confounder.
    """
    innovations = rng.standard_normal(params.n_weeks + _AR_BURN_IN)
    autoregressive = np.asarray(
        lfilter([1.0], [1.0, -params.demand_ar], innovations)[_AR_BURN_IN:], dtype=np.float64
    )
    weeks = np.arange(params.n_weeks, dtype=np.float64)
    sinusoid = np.sin(2.0 * np.pi * weeks / WEEKS_PER_YEAR)
    share = params.demand_seasonal_share
    blended = np.sqrt(1.0 - share) * _standardise(autoregressive) + np.sqrt(share) * _standardise(
        sinusoid
    )
    return _standardise(blended) * params.demand_sd


def _quarterly_cycle(params: DGPParams) -> NDArray[np.float64]:
    """(T, C) budget-cycle multiplier, one fixed phase per channel.

    Phases differ by channel so that the shared budget factor is the *only* route by which
    channels become correlated. That keeps ρ a clean single knob, which the one-knob-at-a-
    time design of §5 requires; a cycle shared across channels would put a correlation floor
    under C0 that D5's ±0.02 tolerance could not accommodate.

    The phases come from the channel table, not from each channel's position in it, so the
    conditions that add a placebo leave the five real channels' cycles untouched.
    """
    weeks = np.arange(params.n_weeks, dtype=np.float64)[:, None]
    phases = np.array([channel.cycle_phase for channel in params.channels], dtype=np.float64)
    angle = 2.0 * np.pi * (weeks + phases[None, :]) / WEEKS_PER_QUARTER
    return np.asarray(1.0 + params.quarterly_amplitude * np.cos(angle), dtype=np.float64)


def _channel_index(params: DGPParams, name: str) -> int | None:
    for index, channel in enumerate(params.channels):
        if channel.name == name:
            return index
    return None


def _assemble_spend(
    params: DGPParams,
    shared: NDArray[np.float64],
    idiosyncratic: NDArray[np.float64],
    season: NDArray[np.float64],
    demand: NDArray[np.float64],
    cycle: NDArray[np.float64],
    mix_weight: float,
    placebo_weight: float,
) -> NDArray[np.float64]:
    """(T, C) weekly spend, £k. Pure in the pre-drawn normals, so the solvers can iterate.

    The random draws are made once per simulation and only remixed here, which is what keeps
    the correlation solves free of RNG side effects and the whole simulation deterministic.
    """
    factors = mix_weight * shared[:, None] + np.sqrt(1.0 - mix_weight**2) * idiosyncratic

    placebo_index = _channel_index(params, "placebo")
    if placebo_index is not None:
        search_index = _channel_index(params, "search")
        if search_index is None:
            raise ValueError("a placebo channel requires a 'search' channel to couple to")
        blend = (
            placebo_weight * factors[:, search_index]
            + placebo_weight * _standardise(season)
            + np.sqrt(max(1.0 - 2.0 * placebo_weight**2, 0.0)) * idiosyncratic[:, placebo_index]
        )
        factors[:, placebo_index] = _standardise(blend)

    log_mean = np.array(
        [np.log(channel.mean_spend) - 0.5 * params.spend_log_sd**2 for channel in params.channels]
    )
    spend = np.exp(log_mean[None, :] + params.spend_log_sd * factors) * cycle

    if params.endogeneity != 0.0:
        # Literally as §2 writes it: "spend scaled by exp(φ·d_t)", not recentred. That does
        # raise mean spend slightly — by exp(φ²·sd(d)²/2), which is 1.1% at C3's strongest
        # level — and the resulting drift in media share is reported rather than corrected,
        # per D4. Recentring was considered and rejected as an assumption not worth its
        # cost for a 1.1% effect.
        spend = spend * np.exp(params.endogeneity * demand)[:, None]

    return np.asarray(spend, dtype=np.float64)


def _mean_pairwise_correlation(spend: NDArray[np.float64], columns: list[int]) -> float:
    """Mean off-diagonal Pearson correlation of spend levels across the given columns."""
    matrix = np.corrcoef(spend[:, columns], rowvar=False)
    upper = np.triu_indices(len(columns), k=1)
    return float(matrix[upper].mean())


def _solve_weight(
    objective: Callable[[float], float],
    target: float,
    upper: float,
    description: str,
) -> float:
    """Find the mixing weight in [0, upper] that hits `target`, or say why it cannot.

    Raises rather than clipping. A silently clipped weight would produce a dataset whose
    correlation is not the one the condition claims, and the whole point of D5's tolerance
    is that such a mismatch has to be impossible rather than merely unlikely.
    """
    low = objective(0.0)
    high = objective(upper)
    if not low - 1e-9 <= target <= high + 1e-9:
        raise ValueError(
            f"{description}: target {target:.4f} is outside the achievable range "
            f"[{low:.4f}, {high:.4f}] for this configuration"
        )
    if abs(low - target) <= 1e-12:
        return 0.0
    if abs(high - target) <= 1e-12:
        return upper
    return float(brentq(lambda weight: objective(weight) - target, 0.0, upper, xtol=1e-12))


def evaluate(
    params: DGPParams,
    spend: NDArray[np.float64],
    demand: NDArray[np.float64],
    season: NDArray[np.float64],
) -> Response:
    """Noiseless sales for an arbitrary spend matrix — the interventional workhorse.

    Deterministic and free of RNG, so Step 3 can call it with a channel zeroed or scaled and
    difference the results to obtain exact causal truth.

    With every column of `spend` set to zero this returns `noiseless_sales == baseline`
    exactly, bit for bit, under both the correct and the misspecified forms. That is the D3
    invariant, and it is what makes "the placebo contributes exactly zero" a fact about the
    arithmetic rather than a statement about a tolerance.

    Args:
        params: the generating parameters.
        spend: (T, C) £k per week, non-negative, columns ordered as `params.channels`.
        demand: (T,) latent demand.
        season: (T,) seasonal multiplier.

    Returns:
        Baseline, per-channel contributions and noiseless sales, all £k per week.
    """
    expected = (params.n_weeks, len(params.channels))
    if spend.shape != expected:
        raise ValueError(f"spend must have shape {expected}; got {spend.shape}")

    weeks = np.arange(params.n_weeks, dtype=np.float64)
    trend = 1.0 + params.trend * weeks / params.n_weeks
    baseline = params.baseline_level * trend * season * np.exp(params.demand_coefficient * demand)

    contributions = np.empty(expected, dtype=np.float64)
    for index, channel in enumerate(params.channels):
        if params.misspecified:
            adstocked = weibull_adstock(
                spend[:, index],
                params.weibull_peak_lag,
                params.weibull_shape,
                params.weibull_max_lag,
            )
            saturated = logistic_saturation(
                adstocked,
                channel.half_saturation,
                channel.half_saturation * params.logistic_scale_ratio,
            )
        else:
            adstocked = geometric_adstock(spend[:, index], channel.decay)
            saturated = hill_saturation(adstocked, channel.half_saturation, channel.hill_shape)
        contributions[:, index] = channel.beta * saturated

    return Response(
        baseline=baseline,
        contributions=contributions,
        noiseless_sales=baseline + contributions.sum(axis=1),
    )


def simulate(params: DGPParams, seed: int) -> SimResult:
    """Draw one dataset. Pure in (params, seed) — no clock, no global RNG, no ordering.

    Draw order is fixed: demand innovations, then the shared budget factor, then the
    idiosyncratic factors, then ε. The correlation solves consume no randomness; they only
    remix draws already made, so a different ρ target changes the spend but never the noise.

    Args:
        params: the generating parameters.
        seed: the only source of randomness.

    Returns:
        A `SimResult`. Only `spend` and `sales` may be shown to an estimator.
    """
    rng = np.random.default_rng(seed)
    season = _seasonality(params.n_weeks, params.season_coefficients)
    demand = _latent_demand(params, rng)
    shared = rng.standard_normal(params.n_weeks)
    idiosyncratic = rng.standard_normal((params.n_weeks, len(params.channels)))
    cycle = _quarterly_cycle(params)

    real_columns = [
        index for index, channel in enumerate(params.channels) if channel.name != "placebo"
    ]

    def spend_at(mix_weight: float, placebo_weight: float) -> NDArray[np.float64]:
        return _assemble_spend(
            params, shared, idiosyncratic, season, demand, cycle, mix_weight, placebo_weight
        )

    mix_weight = 0.0
    if params.collinearity is not None:
        mix_weight = _solve_weight(
            lambda weight: _mean_pairwise_correlation(spend_at(float(weight), 0.0), real_columns),
            params.collinearity,
            upper=0.999,
            description="collinearity (ρ)",
        )

    placebo_weight = 0.0
    # A coupling of exactly 0 is C5's "spend independent": the placebo gets its own draw and
    # nothing else. That is not the same as solving for a realised correlation of 0, which
    # would be fighting sampling noise — the achievable floor is a small positive number, so
    # asking the solver for exactly 0 fails by construction rather than by mistake.
    if params.placebo_coupling is not None and params.placebo_coupling > 0.0:
        search_index = _channel_index(params, "search")
        placebo_index = _channel_index(params, "placebo")
        if search_index is None or placebo_index is None:
            raise ValueError("placebo coupling requires both a 'placebo' and a 'search' channel")

        def placebo_correlation(weight: float) -> float:
            spend = spend_at(mix_weight, float(weight))
            to_search = float(np.corrcoef(spend[:, placebo_index], spend[:, search_index])[0, 1])
            to_season = float(np.corrcoef(spend[:, placebo_index], season)[0, 1])
            return 0.5 * (to_search + to_season)

        placebo_weight = _solve_weight(
            placebo_correlation,
            params.placebo_coupling,
            upper=_PLACEBO_WEIGHT_CEILING - 1e-9,
            description="placebo coupling (D1)",
        )

    spend = spend_at(mix_weight, placebo_weight)
    response = evaluate(params, spend, demand, season)
    sigma = params.noise_fraction * float(response.noiseless_sales.mean())
    sales = response.noiseless_sales + rng.normal(0.0, sigma, params.n_weeks)

    return SimResult(
        channel_names=tuple(channel.name for channel in params.channels),
        spend=spend,
        sales=sales,
        noiseless_sales=response.noiseless_sales,
        baseline=response.baseline,
        contributions=response.contributions,
        demand=demand,
        season=season,
        mix_weight=mix_weight,
        placebo_weight=placebo_weight,
        params=params,
    )


def media_share(result: SimResult) -> float:
    """Total media contribution as a fraction of total noiseless sales.

    Calibrated to ≈25% under C0 only (D4). Everywhere else this drifts, and the drift is
    reported as a descriptive statistic rather than corrected.
    """
    return float(result.contributions.sum() / result.noiseless_sales.sum())


def solve_baseline_level(
    target_share: float = C0_MEDIA_SHARE_TARGET, n_seeds: int = CALIBRATION_SEEDS
) -> float:
    """Solve numerically for the B0 that puts C0's mean media share on target (D9).

    Baseline is exactly linear in B0 and media contribution does not depend on it at all, so
    a *single* draw's share inverts in closed form. The mean across draws does not: a mean of
    ratios is not a ratio of means, so the target is the root of ``mean_s[share_s(B0)] = t``
    and is found rather than derived. The two closed forms differ by which draw you pick —
    seed 0 alone gives 680.8, the 30-seed mean gives 684.1, and neither is the mean's root.

    Args:
        target_share: media contribution as a fraction of total noiseless sales.
        n_seeds: seeds 0..n_seeds-1 to average over.

    Returns:
        B0 in £k per week.
    """

    def gap(level: float) -> float:
        params = DGPParams(baseline_level=level)
        realised = [media_share(simulate(params, seed)) for seed in range(n_seeds)]
        return float(np.mean(realised)) - target_share

    return float(brentq(gap, 100.0, 5000.0, xtol=1e-10))


SYSTEMATIC_CORRELATION_ALLOWANCE = 0.025
"""D16's fixed term in the per-pair correlation bound — the part sampling noise cannot explain.

The realised per-pair spread has a component that does not shrink with more seeds: the
per-channel quarterly phases displace each pair's *mean* correlation from the target, ordered
exactly by phase gap. It is a property of the generator, not of the draw.
"""

MEASURED_SYSTEMATIC_CEILING = 0.021
"""The separate ceiling D16 requires on the *measured* systematic component.

The allowance above is deliberately larger than this. Without a second, independent assertion
a fixed allowance is a place a growing defect can hide: the systematic term could double and
the bound would simply absorb it. Asserting the measurement directly means the allowance
buys tolerance for the defect that was diagnosed, and for no other.
"""


def per_pair_correlation_bound(rho_target: float, n_weeks: int) -> float:
    """Per-pair tolerance: ``0.025 + 4·(1 - ρ²)/√(T - 3)`` — D11's sampling term, D16's offset.

    D11 replaced D5's fixed ±0.10, which was tighter than sampling noise at T=104, with four
    sampling standard errors. That is the ``4·(1 - ρ²)/√(T - 3)`` term, and it scales with
    both T and ρ; a four-sigma breach is about 6e-5 per pair, so it still fires on a real
    defect.

    **D16** adds the constant. The sampling term alone is not the whole story: the spread also
    has a systematic component of 0.017-0.019, measured on seeds 0-29 across all four levels,
    which does not shrink with more seeds. As ρ → 1 the sampling term collapses (0.0172 at
    ρ=0.95, T=520) while the systematic term does not, so a pure sampling bound is breached at
    ρ=0.95 and nowhere else. `SYSTEMATIC_CORRELATION_ALLOWANCE` covers it and
    `MEASURED_SYSTEMATIC_CEILING` keeps the cover honest — see both.

    This completes D5's concession that exactly equal pairwise ρ is unattainable with unequal
    channel volatilities. It is a generator self-consistency check, not one of the G-gates, so
    loosening it changes nothing about how hard the study is for MMM.
    """
    if not 0.0 <= rho_target < 1.0:
        raise ValueError(f"rho_target must lie in [0, 1); got {rho_target}")
    if n_weeks <= 3:
        raise ValueError(f"n_weeks must exceed 3 for the correlation SE; got {n_weeks}")
    sampling = 4.0 * (1.0 - rho_target**2) / np.sqrt(n_weeks - 3)
    return float(SYSTEMATIC_CORRELATION_ALLOWANCE + sampling)


def mean_pairwise_spend_correlation(result: SimResult) -> float:
    """Mean pairwise Pearson correlation of spend *levels* across the real channels (D5).

    The placebo is excluded: its correlation structure is set by its own coupling knob and
    graded by G6, so folding it into ρ would conflate two separate conditions.
    """
    columns = [index for index, name in enumerate(result.channel_names) if name != "placebo"]
    return _mean_pairwise_correlation(result.spend, columns)


CONDITION_LEVELS: dict[str, tuple[float | int | None, ...]] = {
    "C0": (None,),
    "C1": (0.5, 0.8, 0.95),
    "C2": (260, 156, 104),
    "C3": (0.3, 0.6),
    "C4": (None,),
    "C5": (None,),
    "C6": (0.3, 0.45, 0.6),
    "C7": (None,),
}
"""Levels per condition, as tabulated in §5 and amended by D1 for C6."""


def condition_params(condition: str, level: float | int | None = None) -> DGPParams:
    """Build the parameters for one cell of the §5 condition grid.

    Every condition is C0 with exactly one knob moved, except C7, which moves all of them at
    once because that is the configuration an actual client has.

    Args:
        condition: one of C0 … C7.
        level: the level for conditions that have them; None otherwise.

    Returns:
        The `DGPParams` for that cell.
    """
    if condition not in CONDITION_LEVELS:
        raise ValueError(
            f"unknown condition {condition!r}; expected one of {sorted(CONDITION_LEVELS)}"
        )
    if level not in CONDITION_LEVELS[condition]:
        raise ValueError(f"{condition} takes level in {CONDITION_LEVELS[condition]}; got {level!r}")

    def required_level() -> float:
        if level is None:
            raise ValueError(f"{condition} requires a level")
        return float(level)

    with_placebo = (*REAL_CHANNELS, PLACEBO)
    match condition:
        case "C0":
            return DGPParams()
        case "C1":
            return DGPParams(collinearity=required_level())
        case "C2":
            return DGPParams(n_weeks=int(required_level()))
        case "C3":
            return DGPParams(demand_coefficient=0.5, endogeneity=required_level())
        case "C4":
            return DGPParams(misspecified=True)
        case "C5":
            return DGPParams(channels=with_placebo, placebo_coupling=0.0)
        case "C6":
            return DGPParams(channels=with_placebo, placebo_coupling=required_level())
        case _:
            return DGPParams(
                channels=with_placebo,
                n_weeks=104,
                collinearity=0.7,
                demand_coefficient=0.5,
                endogeneity=0.6,
                # D10: 0.45 rather than 0.6, which was unreachable on 1.8% of seeds. It is
                # also C6's middle level, so C7 decomposes against C6[0.45].
                placebo_coupling=0.45,
                misspecified=True,
            )
