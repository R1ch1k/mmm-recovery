"""The start-at-the-truth optimiser diagnostic, regenerated as a committed module — D43.

The claim this module exists to support is quoted in `README.md`, in the strict-xfail reason
string in `tests/test_estimator.py`, and in `PREREGISTRATION.md`: *Nelder–Mead started exactly
at the true hyperparameters walks away from them.* It is load-bearing rather than decorative,
because it is one of the two results that distinguish **non-identification** from **search
failure**. If the optimiser handed the right answer for free chooses to leave it, then no amount
of search power reaches the truth, and the failure is not a weak optimiser.

It was published from a harness that was never committed, and its quoted numbers — CV RMSE
3.63160 → 3.06272, median |bias| 2.4% → 57.3%, max 3.7% → 7133% — share the retired 3.63160
that D39 established belongs to §4's **superseded six-column control block**. So the figure was
presumed stale in the same way the plateau centrepiece was, and presumption is not a finding.

**This module closes the last figure in the study with no artefact behind it, as the second and
final pre-committed exception to the scope freeze.** The justification is D39's: regenerating an
already-published number can only expose an error, never manufacture a finding. The handling was
fixed in writing before the run — the regenerated figure is authoritative, every downstream site
is updated, the discrepancy is logged with both values, and the diagnostic is **not** tuned to
recover 3.63160.

What the optimiser is allowed to move
-------------------------------------

All sixteen hyperparameters, in the coordinates the random search itself draws in: `λ` and `α`
linearly, `κ` as a log multiple of each channel's *observed* mean adstocked spend, and the shared
ridge penalty on a log scale. `κ` is recomputed from the current `λ` at every evaluation, exactly
as `_draw_hyperparameters` does, so the optimiser and the search explore the same space. Every
bound is `SearchBounds`, so the optimiser can go nowhere the 200-draw search could not.

The objective is `_cross_validated_rmse` — the quantity the estimator actually minimises. The fit
path is `plateau._fit_at`, reused rather than copied so the diagnostic is scored on identical
terms to the identification plateau.

What differs from the original, recorded because the original cannot be re-read
-------------------------------------------------------------------------------

* **Condition and seed are C0, seed 0** — the same draw `plateau.py` establishes from the
  original's own bias arithmetic.
* **The control block is D22's ten columns, not the six the original used.** This is the
  substantive difference and the reason the original's numbers cannot recur: a truth CV of
  3.63160 is the size of the structured residual D22 records for the superseded list.
* **Both series are run.** The noisy series is what the estimator sees and what every gate is
  computed on; the noiseless arm is the contrast that separates "unidentifiable form" from
  "unidentifiable from this data".
* **Convergence settings are stated, not defaulted silently**: `maxfev` and the simplex
  tolerances below, with the exit status recorded in the output rather than assumed.

Nelder–Mead's initial simplex is a deterministic function of the starting point, so this module
takes a seed only for the DGP draw and reproduces byte-identically from a clean checkout.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from mmm_recovery.dgp import DGPParams, SimResult, condition_params, simulate
from mmm_recovery.estimator import Hyperparameters, SearchBounds, adstock_spend
from mmm_recovery.plateau import _fit_at, truth_hyperparameters
from mmm_recovery.truth import incremental_contribution, response_surface

RESULTS_DIR: Final = Path(__file__).resolve().parents[2] / "results"
DIAGNOSTIC_CSV: Final = RESULTS_DIR / "nelder_mead_diagnostic.csv"

DIAGNOSTIC_SEED: Final = 0
"""C0 seed 0 — the draw `plateau.py` established the original sweep used."""

MAX_EVALUATIONS: Final = 60_000
"""Recorded, not defaulted. The noisy arm — the binding one — converges at 32,407 evaluations,
so this is roughly double what it needs. Measured insensitivity: capping at 20,000 stops the
simplex early and still gives CV 30.60899 with median |bias| 70.8% against the converged 70.9%,
so the result is a property of the objective rather than of the budget."""

SIMPLEX_TOLERANCE: Final = 1e-10
"""`xatol` and `fatol`. Far below the CV differences at stake, so convergence is decided by the
objective flattening rather than by a tolerance chosen to reach it."""


@dataclass(frozen=True)
class Stage:
    """The diagnostic at one point: where the optimiser was, and what it believed there.

    Attributes:
        series: ``noisy`` or ``noiseless`` — which sales vector was fitted.
        stage: ``start`` (the true hyperparameters) or ``end`` (where Nelder–Mead stopped).
        cv_rmse: pooled expanding-window out-of-sample RMSE, £k per week.
        hyper: the transform at this point.
        contribution: (C,) implied incremental contribution per channel, £k over the horizon.
        relative_bias: (C,) ``contribution / true - 1``.
        n_evaluations: objective evaluations spent reaching this point; 0 at ``start``.
        converged: the optimiser's own exit flag; ``True`` at ``start`` by convention.
        message: the optimiser's exit message.
    """

    series: str
    stage: str
    cv_rmse: float
    hyper: Hyperparameters
    contribution: NDArray[np.float64]
    relative_bias: NDArray[np.float64]
    n_evaluations: int
    converged: bool
    message: str

    @property
    def median_absolute_bias(self) -> float:
        """G1's own metric: median |relative bias| across channels."""
        return float(np.median(np.abs(self.relative_bias)))

    @property
    def max_absolute_bias(self) -> float:
        return float(np.max(np.abs(self.relative_bias)))


def _pack(hyper: Hyperparameters, spend: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hyperparameters -> the (3C+1,) search-coordinate vector the optimiser moves in."""
    mean_adstocked = adstock_spend(spend, hyper.decay).mean(axis=0)
    return np.asarray(
        np.concatenate(
            [
                hyper.decay,
                hyper.hill_shape,
                np.log(hyper.half_saturation / mean_adstocked),
                [np.log(hyper.ridge_penalty)],
            ]
        ),
        dtype=np.float64,
    )


def _unpack(vector: NDArray[np.float64], spend: NDArray[np.float64]) -> Hyperparameters:
    """The inverse of `_pack`. κ is rebuilt from the *current* λ, as the search draws it."""
    n_channels = spend.shape[1]
    decay = np.asarray(vector[:n_channels], dtype=np.float64)
    hill_shape = np.asarray(vector[n_channels : 2 * n_channels], dtype=np.float64)
    ratio = np.exp(np.asarray(vector[2 * n_channels : 3 * n_channels], dtype=np.float64))
    mean_adstocked = adstock_spend(spend, decay).mean(axis=0)
    return Hyperparameters(
        decay=decay,
        hill_shape=hill_shape,
        half_saturation=ratio * mean_adstocked,
        ridge_penalty=float(np.exp(vector[3 * n_channels])),
    )


def _search_bounds(n_channels: int, bounds: SearchBounds) -> list[tuple[float, float]]:
    """`SearchBounds` in packed coordinates, log-scaled where the search draws log-uniformly."""
    ratio, ridge = bounds.half_saturation_ratio, bounds.ridge_penalty
    log_ratio = (float(np.log(ratio[0])), float(np.log(ratio[1])))
    log_ridge = (float(np.log(ridge[0])), float(np.log(ridge[1])))
    return (
        [bounds.decay] * n_channels
        + [bounds.hill_shape] * n_channels
        + [log_ratio] * n_channels
        + [log_ridge]
    )


def _evaluate(
    sim: SimResult,
    sales: NDArray[np.float64],
    hyper: Hyperparameters,
    true_contribution: NDArray[np.float64],
    *,
    series: str,
    stage: str,
    n_evaluations: int,
    converged: bool,
    message: str,
) -> Stage:
    score, contribution = _fit_at(sim, sales, hyper)
    return Stage(
        series=series,
        stage=stage,
        cv_rmse=score,
        hyper=hyper,
        contribution=contribution,
        relative_bias=contribution / true_contribution - 1.0,
        n_evaluations=n_evaluations,
        converged=converged,
        message=message,
    )


def diagnose(
    series: str, seed: int = DIAGNOSTIC_SEED, params: DGPParams | None = None
) -> tuple[Stage, Stage]:
    """Start Nelder–Mead at the truth and report where it stops.

    Args:
        series: ``noisy`` for `sim.sales`, ``noiseless`` for `sim.noiseless_sales`.
        seed: the C0 draw to run on.
        params: override the condition, for tests. Defaults to C0.

    Returns:
        The ``start`` stage (the true hyperparameters) and the ``end`` stage.
    """
    if series not in ("noisy", "noiseless"):
        raise ValueError(f"series must be 'noisy' or 'noiseless'; got {series!r}")

    params = condition_params("C0") if params is None else params
    sim = simulate(params, seed)
    sales = sim.sales if series == "noisy" else sim.noiseless_sales
    true_contribution = incremental_contribution(response_surface(sim))

    truth = truth_hyperparameters(params)
    start = _evaluate(
        sim,
        sales,
        truth,
        true_contribution,
        series=series,
        stage="start",
        n_evaluations=0,
        converged=True,
        message="the true hyperparameters, not an optimiser result",
    )

    def objective(vector: NDArray[np.float64], /) -> float:
        score, _ = _fit_at(sim, sales, _unpack(vector, sim.spend))
        return score

    outcome = minimize(
        objective,
        _pack(truth, sim.spend),
        method="Nelder-Mead",
        bounds=_search_bounds(sim.spend.shape[1], SearchBounds()),
        options={
            "maxfev": MAX_EVALUATIONS,
            "xatol": SIMPLEX_TOLERANCE,
            "fatol": SIMPLEX_TOLERANCE,
        },
    )
    end = _evaluate(
        sim,
        sales,
        _unpack(np.asarray(outcome.x, dtype=np.float64), sim.spend),
        true_contribution,
        series=series,
        stage="end",
        n_evaluations=int(outcome.nfev),
        converged=bool(outcome.success),
        message=str(outcome.message),
    )
    return start, end


def write_csv(stages: list[Stage], channel_names: tuple[str, ...], path: Path) -> None:
    """One row per (series, stage, channel). `cv_rmse` repeats across a stage's channel rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "series",
                "stage",
                "channel",
                "cv_rmse",
                "decay",
                "hill_shape",
                "half_saturation",
                "ridge_penalty",
                "contribution",
                "relative_bias",
                "n_evaluations",
                "converged",
            ]
        )
        for entry in stages:
            for index, channel in enumerate(channel_names):
                writer.writerow(
                    [
                        entry.series,
                        entry.stage,
                        channel,
                        f"{entry.cv_rmse:.10g}",
                        f"{entry.hyper.decay[index]:.10g}",
                        f"{entry.hyper.hill_shape[index]:.10g}",
                        f"{entry.hyper.half_saturation[index]:.10g}",
                        f"{entry.hyper.ridge_penalty:.10g}",
                        f"{entry.contribution[index]:.10g}",
                        f"{entry.relative_bias[index]:.10g}",
                        entry.n_evaluations,
                        entry.converged,
                    ]
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start-at-the-truth Nelder-Mead diagnostic.")
    parser.add_argument("--seed", type=int, default=DIAGNOSTIC_SEED)
    parser.add_argument("--out", type=Path, default=DIAGNOSTIC_CSV)
    args = parser.parse_args(argv)

    sim = simulate(condition_params("C0"), args.seed)
    stages: list[Stage] = []
    for series in ("noisy", "noiseless"):
        start, end = diagnose(series, args.seed)
        stages.extend([start, end])

        print(f"--- {series}, C0 seed {args.seed} ---")
        for entry in (start, end):
            print(
                f"  {entry.stage:<5} CV RMSE {entry.cv_rmse:>12.5f}   "
                f"median |bias| {entry.median_absolute_bias:>8.1%}   "
                f"max |bias| {entry.max_absolute_bias:>10.1%}"
            )
        # The claim is about BIAS, not about CV. Labelling this on the CV movement alone would
        # report "walks away" on the noiseless arm, where the optimiser shaves the score by a
        # rounding error and leaves recovery exact.
        recovery = end.median_absolute_bias - start.median_absolute_bias
        verdict = (
            "WALKS AWAY: CV improves and recovery gets worse"
            if end.cv_rmse < start.cv_rmse and recovery > 0.01
            else "STAYS: the objective does not pull the optimiser off the truth"
        )
        print(
            f"  {verdict}\n"
            f"  CV {start.cv_rmse:.5f} -> {end.cv_rmse:.5f}, "
            f"median |bias| {start.median_absolute_bias:.1%} -> {end.median_absolute_bias:.1%} "
            f"({end.n_evaluations} evaluations, converged={end.converged})"
        )
        print(f"  exit: {end.message}")

    write_csv(stages, sim.channel_names, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a command, not in the suite
    raise SystemExit(main())
