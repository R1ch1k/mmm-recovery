"""The identification plateau, regenerated as a committed module — D39.

The README's centrepiece claim is that a wide set of very different worlds fit the data
equally well: "177 of 780 grid points sit within 1% of the true parameters' cross-validation
score, and across that near-tied set the implied TV contribution ranges from £43,938k to
£240,522k". Those numbers came from a harness that was never committed. They survived only as
prose in the deviations log and in the strict-xfail reason string, which is the reproducibility
gap `README.md` names as the first outstanding task.

**This module closes it, and the closing is a deliberate, pre-committed exception to a scope
freeze.** The justification is that a regeneration of an already-published number can only
expose an error, never manufacture a finding, which is categorically different from a run whose
result is not yet known. The handling of a mismatch was fixed in writing before this ran: the
regenerated figure is authoritative, the sweep is *not* tuned to recover 177, and any difference
is logged with both values and the likely cause.

What is held fixed and what is swept
------------------------------------

Four channels sit at their true `(λ, α, κ)`. TV's `λ` is also held at truth. Only TV's
saturation pair `(α, κ)` moves, over a 26 × 30 grid — 780 points, matching the original's count.
`α` spans `SearchBounds.hill_shape` on a linear grid; `κ` spans `SearchBounds.half_saturation_
ratio` on a log grid, because that is how `_draw_hyperparameters` draws it. Both ranges are the
estimator's own published search bounds, so no grid point is anywhere the search could not go.
The true ratio, 1.097, is interior to the swept range.

What differs from the original, recorded because the original cannot be re-read
-------------------------------------------------------------------------------

* **Seed and condition are the same.** C0, seed 0. This is established rather than assumed: the
  original's maximum bias of 362.6% against a contribution of £240,522k implies a true TV
  contribution of £51,989k, and `truth.incremental_contribution` on C0 seed 0 returns exactly
  £51,989k.
* **The control block is not the same, and this is the substantive difference.** The original's
  quoted truth CV RMSE of 3.63160, with 2.4% bias at the true hyperparameters, is reproducible
  under neither current configuration: with D22's ten-column controls the noiseless truth scores
  **0.00002** with 0.00% bias, and the noisy series scores **30.94**. A CV RMSE of 3.63 £k per
  week is the size of the structured residual D22 records for §4's original six-column control
  list (2.45 £k per week), so the original sweep was run **before D22** and its numbers describe
  a control block the study has since abandoned.
* **The primary series is the noisy one**, which is what the estimator actually sees and what
  every gate in the study is computed on. The noiseless arm is run alongside and is the contrast
  that makes the mechanism precise: with correct controls and no noise the truth is recovered
  exactly, so the plateau is not a property of the functional form on its own.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from mmm_recovery.dgp import DGPParams, SimResult, condition_params, simulate
from mmm_recovery.estimator import (
    FittedSurface,
    Hyperparameters,
    SearchBounds,
    _cross_validated_rmse,
    _normal_equations,
    _solve_bounded_ridge,
    adstock_spend,
    control_matrix,
    design_matrix,
    estimated_contribution,
    expanding_window_folds,
)
from mmm_recovery.truth import incremental_contribution, response_surface

RESULTS_DIR: Final = Path(__file__).resolve().parents[2] / "results"
PLATEAU_CSV: Final = RESULTS_DIR / "plateau_sweep.csv"

PLATEAU_SEED: Final = 0
"""C0 seed 0 — the draw the original sweep used, established from its own bias arithmetic."""

SWEPT_CHANNEL: Final = "tv"
"""TV. The largest channel, and the one the original swept."""

N_ALPHA: Final = 26
N_RATIO: Final = 30
"""26 × 30 = 780, the original's grid size. Resolution is matched; nothing else is tuned."""

BAND: Final = 0.01
"""'Within 1% of the true parameters' CV score' — the original's band, applied multiplicatively
to the truth's CV RMSE. Stated on the figure and in the CSV so it cannot drift silently."""


@dataclass(frozen=True)
class PlateauCell:
    """One grid point: what the model would believe if it chose this transform.

    Attributes:
        series: ``noisy`` or ``noiseless`` — which sales vector was fitted.
        hill_shape: α̂ for TV at this point.
        half_saturation_ratio: κ̂ as a multiple of TV's mean adstocked spend, the units the
            search draws in.
        half_saturation: κ̂ in £k per week.
        cv_rmse: pooled expanding-window out-of-sample RMSE, £k per week. The quantity the
            estimator actually minimises.
        tv_contribution: implied incremental contribution of TV, £k over the horizon.
        relative_bias: ``tv_contribution / true - 1``.
    """

    series: str
    hill_shape: float
    half_saturation_ratio: float
    half_saturation: float
    cv_rmse: float
    tv_contribution: float
    relative_bias: float


def _fit_at(
    sim: SimResult, sales: NDArray[np.float64], hyper: Hyperparameters
) -> tuple[float, NDArray[np.float64]]:
    """CV score and implied contributions at one fixed hyperparameter set.

    This mirrors the tail of ``RidgeMMM.fit`` with the random search removed: same design, same
    bounded ridge solve, same ``estimated_contribution``. It is duplicated rather than factored
    out of ``fit`` because ``fit``'s contract is "search then fit" and a public "fit at these
    exact hyperparameters" entry point would be a new way to hand the estimator the answer.
    """
    n_weeks, n_media = sim.spend.shape
    design = design_matrix(sim.spend, hyper)
    folds = expanding_window_folds(n_weeks)
    score = _cross_validated_rmse(design, sales, hyper.ridge_penalty, n_media, folds)

    gram, moment = _normal_equations(design, sales)
    coefficients = _solve_bounded_ridge(gram, moment, hyper.ridge_penalty, n_media)
    surface = FittedSurface(
        adstocked=adstock_spend(sim.spend, hyper.decay),
        coefficients=coefficients[:n_media],
        half_saturation=hyper.half_saturation,
        hill_shape=hyper.hill_shape,
        baseline=control_matrix(n_weeks) @ coefficients[n_media:],
        spend_totals=np.asarray(sim.spend.sum(axis=0), dtype=np.float64),
    )
    return score, estimated_contribution(surface)


def truth_hyperparameters(params: DGPParams) -> Hyperparameters:
    """The generating process's own transform, at the search's ridge floor.

    The floor rather than zero because that is what `SearchBounds` permits and what the CV
    comparison in `tests/test_estimator.py` uses, so the truth is scored on the same terms every
    random draw is.
    """
    return Hyperparameters(
        decay=np.array([channel.decay for channel in params.channels], dtype=np.float64),
        hill_shape=np.array([channel.hill_shape for channel in params.channels], dtype=np.float64),
        half_saturation=np.array(
            [channel.half_saturation for channel in params.channels], dtype=np.float64
        ),
        ridge_penalty=SearchBounds().ridge_penalty[0],
    )


def sweep(series: str, seed: int = PLATEAU_SEED) -> tuple[list[PlateauCell], float, float]:
    """The 780-point sweep over TV's ``(α, κ)``, everything else held at truth.

    Args:
        series: ``noisy`` for `sim.sales`, ``noiseless`` for `sim.noiseless_sales`.
        seed: the C0 draw to sweep on.

    Returns:
        The cells, the truth's CV RMSE on this series, and the true TV contribution in £k.
    """
    if series not in ("noisy", "noiseless"):
        raise ValueError(f"series must be 'noisy' or 'noiseless'; got {series!r}")

    params = condition_params("C0")
    sim = simulate(params, seed)
    sales = sim.sales if series == "noisy" else sim.noiseless_sales

    index = sim.channel_names.index(SWEPT_CHANNEL)
    truth = truth_hyperparameters(params)
    true_contribution = float(incremental_contribution(response_surface(sim))[index])
    truth_score, _ = _fit_at(sim, sales, truth)

    bounds = SearchBounds()
    mean_adstocked = float(adstock_spend(sim.spend, truth.decay)[:, index].mean())
    alphas = np.linspace(bounds.hill_shape[0], bounds.hill_shape[1], N_ALPHA)
    ratios = np.geomspace(bounds.half_saturation_ratio[0], bounds.half_saturation_ratio[1], N_RATIO)

    cells: list[PlateauCell] = []
    for alpha in alphas:
        for ratio in ratios:
            hill_shape = truth.hill_shape.copy()
            half_saturation = truth.half_saturation.copy()
            hill_shape[index] = alpha
            half_saturation[index] = ratio * mean_adstocked
            hyper = Hyperparameters(
                decay=truth.decay,
                hill_shape=hill_shape,
                half_saturation=half_saturation,
                ridge_penalty=truth.ridge_penalty,
            )
            score, contribution = _fit_at(sim, sales, hyper)
            cells.append(
                PlateauCell(
                    series=series,
                    hill_shape=float(alpha),
                    half_saturation_ratio=float(ratio),
                    half_saturation=float(half_saturation[index]),
                    cv_rmse=score,
                    tv_contribution=float(contribution[index]),
                    relative_bias=float(contribution[index]) / true_contribution - 1.0,
                )
            )
    return cells, truth_score, true_contribution


def near_tied(
    cells: list[PlateauCell], truth_score: float, band: float = BAND
) -> list[PlateauCell]:
    """The cells the objective cannot distinguish from the truth, at the stated band."""
    ceiling = truth_score * (1.0 + band)
    return [cell for cell in cells if cell.cv_rmse <= ceiling]


def write_csv(cells: list[PlateauCell], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "series",
                "hill_shape",
                "half_saturation_ratio",
                "half_saturation",
                "cv_rmse",
                "tv_contribution",
                "relative_bias",
            ]
        )
        for cell in cells:
            writer.writerow(
                [
                    cell.series,
                    f"{cell.hill_shape:.10g}",
                    f"{cell.half_saturation_ratio:.10g}",
                    f"{cell.half_saturation:.10g}",
                    f"{cell.cv_rmse:.10g}",
                    f"{cell.tv_contribution:.10g}",
                    f"{cell.relative_bias:.10g}",
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=PLATEAU_SEED)
    parser.add_argument("--out", type=Path, default=PLATEAU_CSV)
    args = parser.parse_args(argv)

    rows: list[PlateauCell] = []
    for series in ("noisy", "noiseless"):
        cells, truth_score, true_contribution = sweep(series, args.seed)
        rows.extend(cells)
        tied = near_tied(cells, truth_score)

        print(f"--- {series}, C0 seed {args.seed}, {len(cells)} grid points ---")
        print(f"  true TV contribution        {true_contribution:,.0f} GBPk")
        print(f"  truth CV RMSE               {truth_score:.5f}")
        print(f"  within {BAND:.0%} of it            {len(tied)} of {len(cells)}")
        if tied:
            low = min(cell.tv_contribution for cell in tied)
            high = max(cell.tv_contribution for cell in tied)
            worst = max(abs(cell.relative_bias) for cell in tied)
            best = min(abs(cell.relative_bias) for cell in tied)
            print(f"  TV contribution across them {low:,.0f} to {high:,.0f} GBPk")
            print(f"  spread                      {high / low:.1f}x")
            print(f"  |bias| across them          {best:.1%} to {worst:.1%}")

    write_csv(rows, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a command, not in the suite
    raise SystemExit(main())
