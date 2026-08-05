"""D26's exploratory spend-variation sweep. **Not pre-registered.**

This module answers one post-hoc question: how much deliberate week-to-week variation in
spend does a team need before MMM identifies anything? It sweeps `spend_log_sd` across
0.15 / 0.30 / 0.60 / 1.00 at C0 and reports the six §7 gates at each level.

Three things about it are deliberate.

**It is exploratory and it sets no gate.** D26 records that it was added after C0 failed. The
§7 thresholds are printed beside each level because they are the reader's yardstick, not
because any level here can pass or fail the study.

**The 0.30 level is a control, not a data point.** 0.30 *is* C0, so that column must
reproduce D23's table — 0.540 / 0.417 / 0.650 / 2.355 / 0.200 — or this harness measures
something other than what the study measured and the other three columns mean nothing.
`verify_control` asserts exactly that, and `main` refuses to write results without it.

**G6 does not exist here.** §7 applies G6 to C5, C6 and C7 only, and C0 has no placebo
channel to flag, so this module reports it as not-applicable rather than inventing a number.
There are six gates in §7 and five of them are measurable at C0.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from mmm_recovery.dgp import DGPParams, condition_params, simulate
from mmm_recovery.estimator import RidgeMMM, allocation_regret, recommended_allocation
from mmm_recovery.truth import incremental_contribution, optimal_allocation, response_surface

SWEEP_LEVELS: Final = (0.15, 0.30, 0.60, 1.00)
"""σ of log weekly spend. 0.30 is C0's assumed value and serves as the control."""

CONTROL_LEVEL: Final = 0.30

N_SWEEP_SEEDS: Final = 200
"""Matched to D23 so the control column is comparable seed for seed."""

D23_CONTROL: Final = {
    "G1_median_abs_relative_bias": 0.540,
    "G2_coverage": 0.417,
    "G3_median_spearman": 0.650,
    "G4_median_regret": 2.355,
    "G5_beats_status_quo": 0.200,
}
"""D23's C0 table at 200 seeds. The control column must reproduce this."""

CONTROL_TOLERANCE: Final = 5e-4
"""D23 quotes three decimals, so agreement is asserted at half a unit in the last place."""

OPTIMISER_SEED: Final = 0
"""The start seed for **both** SLSQP solves. Fixed, and deliberately not the data seed.

Recovered rather than documented — see D28. D23's harness was never committed, and a faithful
re-implementation passing `seed=seed` to the recommendation solve reproduces G1, G2 and G3 to
the printed digit but lands G4 at 2.386 against D23's 2.355 and G5 at 0.195 against 0.200. A
fixed seed of 0 reproduces all five exactly (G4 2.3546, G5 0.2000).

The truth-side solve is insensitive to this: `seed=seed` and `seed=0` give identical aggregates,
because the structured starts (§ `_structured_starts`) win on the true surface and the single
screened random start never decides the optimum there. Only the *fitted* surface's solve moves,
which is D17's non-concavity showing up exactly where the surface is a bad estimate of a
non-concave truth.

Holding it fixed is also the defensible choice on its own terms: the optimiser is a deterministic
tool applied to each fitted surface, and there is no reason its starting points should co-vary
with the seed that generated the data. The gap between the two conventions — 1.3% relative on
median regret, one seed of 200 on beats-status-quo — is reported in the limitations as the
sensitivity of the decision gates to optimiser starts.
"""

GATE_THRESHOLDS: Final = {
    "G1_median_abs_relative_bias": ("<", 0.20),
    "G2_coverage": (">=", 0.80),
    "G3_median_spearman": (">=", 0.80),
    "G4_median_regret": ("<", 0.20),
    "G5_beats_status_quo": (">=", 0.90),
}

RESULTS_DIR: Final = Path(__file__).resolve().parents[2] / "results"
SWEEP_CSV: Final = RESULTS_DIR / "spend_variation_sweep.csv"


@dataclass(frozen=True)
class SeedOutcome:
    """Everything one (level, seed) cell produces.

    Attributes:
        level: the `spend_log_sd` this cell ran at.
        seed: the seed, shared by the DGP, the fit and both optimiser solves.
        relative_bias: (C,) signed relative bias of estimated incremental contribution.
        covered: (C,) whether the true contribution fell inside the nominal 90% interval.
        spearman: rank correlation of true against estimated contribution across channels.
        regret: §6 rung 3, `(S_opt - S_model) / (S_opt - S_sq)`. Unclipped.
        beats_status_quo: whether the recommended allocation beat doing nothing.
        achievable_lift_share: `(S_opt - S_sq) / S_sq`, the headroom that existed.
        shortfall_from_optimum_share: `(S_opt - S_model) / S_sq`.
        loss_vs_status_quo_share: `(S_sq - S_model) / S_sq`. Positive means value destroyed.
        n_agreeing: multi-start agreement count on the truth solve, carried as a diagnostic
            so an optimiser failure cannot be read as a low-regret result.
    """

    level: float
    seed: int
    relative_bias: NDArray[np.float64]
    covered: NDArray[np.bool_]
    spearman: float
    regret: float
    beats_status_quo: bool
    achievable_lift_share: float
    shortfall_from_optimum_share: float
    loss_vs_status_quo_share: float
    n_agreeing: int

    @property
    def median_abs_relative_bias(self) -> float:
        """G1's per-seed quantity: median across channels of |relative bias|."""
        return float(np.median(np.abs(self.relative_bias)))


@dataclass(frozen=True)
class SolveFailure:
    """A cell where SLSQP failed and no allocation exists to score.

    `truth.optimal_allocation` raises on any failed start, which is deliberate and stays that
    way: CLAUDE.md names "optimiser convergence failures reported as low regret" as a specific
    failure mode, and every number already published rests on that strictness. Softening it
    inside `truth.py` would move results that are already in the deviations log.

    So the harness records the cell instead. Measured scope at 200 seeds per level: **2 cells
    out of 800**, both on the *fitted* surface (sd 0.15 seed 78, sd 1.00 seed 160), both from a
    structured start, both with healthy coefficients — a local linesearch failure on a
    non-concave estimate, not a degenerate model. The truth solve never failed anywhere.

    These cells are excluded from the gate medians and the exclusion is reported. At one cell in
    two hundred it cannot move a median; `g5_worst_case` in `descriptives` carries the adverse
    bound anyway, by counting every failure as a loss.
    """

    level: float
    seed: int
    stage: str
    message: str


CellResult = SeedOutcome | SolveFailure

SLSQP_SIGNATURE: Final = "SLSQP failed from start"
"""Only this failure is recorded. Anything else is a real bug and must crash."""


def sweep_params(level: float) -> DGPParams:
    """C0 with `spend_log_sd` moved to `level` and nothing else touched."""
    return replace(condition_params("C0"), spend_log_sd=level)


def run_cell(level: float, seed: int) -> CellResult:
    """Simulate, fit, optimise and score one (level, seed) cell.

    Pure in `(level, seed)`: the DGP, the random search and the bootstrap all take their stream
    from `seed`, and both SLSQP solves take the fixed `OPTIMISER_SEED`, so the result does not
    depend on worker count or completion order.
    """
    sim = simulate(sweep_params(level), seed)
    surface = response_surface(sim)
    truth_contribution = incremental_contribution(surface)
    try:
        optimum = optimal_allocation(surface, seed=OPTIMISER_SEED)
    except ValueError as exc:
        if SLSQP_SIGNATURE not in str(exc):
            raise
        return SolveFailure(level, seed, "truth-solve", str(exc))

    fit = RidgeMMM().fit(sim.spend, sim.sales, seed)
    try:
        recommended = recommended_allocation(fit.surface, seed=OPTIMISER_SEED)
    except ValueError as exc:
        if SLSQP_SIGNATURE not in str(exc):
            raise
        return SolveFailure(level, seed, "recommendation-solve", str(exc))
    model_sales = surface.total_sales(recommended.multipliers)

    status_quo_sales = optimum.status_quo_sales
    lower, upper = fit.contribution_interval[:, 0], fit.contribution_interval[:, 1]
    rank = spearmanr(truth_contribution, fit.contribution).statistic

    return SeedOutcome(
        level=level,
        seed=seed,
        relative_bias=(fit.contribution - truth_contribution) / truth_contribution,
        covered=(truth_contribution >= lower) & (truth_contribution <= upper),
        spearman=float(rank),
        regret=allocation_regret(surface, recommended.multipliers, optimum),
        beats_status_quo=bool(model_sales > status_quo_sales),
        achievable_lift_share=optimum.achievable_lift / status_quo_sales,
        shortfall_from_optimum_share=(optimum.total_sales - model_sales) / status_quo_sales,
        loss_vs_status_quo_share=(status_quo_sales - model_sales) / status_quo_sales,
        n_agreeing=optimum.n_agreeing,
    )


def _run_cell_packed(job: tuple[float, int]) -> CellResult:
    """`ProcessPoolExecutor.map` takes one argument; this unpacks it."""
    return run_cell(*job)


def successes(results: Sequence[CellResult]) -> list[SeedOutcome]:
    """The cells that produced an allocation, in input order."""
    return [result for result in results if isinstance(result, SeedOutcome)]


def failures(results: Sequence[CellResult]) -> list[SolveFailure]:
    """The cells where SLSQP failed, in input order."""
    return [result for result in results if isinstance(result, SolveFailure)]


def gates(results: Sequence[CellResult]) -> dict[str, float]:
    """The five §7 gates measurable at C0, aggregated across seeds.

    G2 pools every (seed, channel) pair, which is what "empirical coverage of the 90%
    interval" means. The other four take the median or mean across seeds of a per-seed
    quantity, so a single catastrophic seed cannot carry a gate on its own.

    Cells where SLSQP failed are excluded — see `SolveFailure` for the scope and for the
    adverse bound that `descriptives` carries alongside.
    """
    outcomes = successes(results)
    if not outcomes:
        raise ValueError("no outcomes to aggregate")
    return {
        "G1_median_abs_relative_bias": float(
            np.median([outcome.median_abs_relative_bias for outcome in outcomes])
        ),
        "G2_coverage": float(np.mean(np.concatenate([o.covered for o in outcomes]))),
        "G3_median_spearman": float(np.median([o.spearman for o in outcomes])),
        "G4_median_regret": float(np.median([o.regret for o in outcomes])),
        "G5_beats_status_quo": float(np.mean([o.beats_status_quo for o in outcomes])),
    }


Z_95: Final = 1.959963984540054
"""Two-sided normal quantile for a 95% interval."""


def wilson_interval(n_successes: int, n_trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    G5 is a Bernoulli rate over independent seeds and CLAUDE.md requires numbers to carry their
    uncertainty, so it cannot be reported bare. Wilson rather than a bootstrap on purpose: Xu,
    Thomadsen & Zhang (2025) argue the ordinary bootstrap is invalid for post-optimisation
    quantities because the argmax is nonsmooth. That argument does not reach a proportion over
    independent replications, but a closed form sidesteps the question entirely and needs no
    resampling to be reproducible. Wilson also behaves near 0 and 1, where the normal
    approximation puts interval ends outside [0, 1].
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive; got {n_trials}")
    proportion = n_successes / n_trials
    denominator = 1.0 + z**2 / n_trials
    centre = (proportion + z**2 / (2 * n_trials)) / denominator
    half = (
        z
        / denominator
        * float(np.sqrt(proportion * (1 - proportion) / n_trials + z**2 / (4 * n_trials**2)))
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def descriptives(results: Sequence[CellResult]) -> dict[str, float]:
    """D19 and D27's absolute figures, which travel with every ratio.

    `shortfall_from_optimum` is what the model left on the table against the best possible
    allocation; `loss_vs_status_quo` is what it destroyed against doing nothing. Both are
    shares of total sales. D27 fixes the absolute figure as the one that leads.

    `g5_worst_case` is the adverse bound on G5: every excluded solve failure counted as a
    loss. It exists so that excluding those cells cannot be mistaken for hiding them.
    """
    outcomes = successes(results)
    n_failed = len(failures(results))
    shortfall = np.array([o.shortfall_from_optimum_share for o in outcomes])
    loss = np.array([o.loss_vs_status_quo_share for o in outcomes])
    lift = np.array([o.achievable_lift_share for o in outcomes])
    regret = np.array([o.regret for o in outcomes])
    g5_low, g5_high = wilson_interval(sum(o.beats_status_quo for o in outcomes), len(outcomes))
    above_low, above_high = wilson_interval(int((regret > 1.0).sum()), len(outcomes))
    return {
        "achievable_lift_share_median": float(np.median(lift)),
        "shortfall_from_optimum_median": float(np.median(shortfall)),
        "shortfall_from_optimum_p10": float(np.percentile(shortfall, 10)),
        "shortfall_from_optimum_p90": float(np.percentile(shortfall, 90)),
        "loss_vs_status_quo_median": float(np.median(loss)),
        "loss_vs_status_quo_p10": float(np.percentile(loss, 10)),
        "loss_vs_status_quo_p90": float(np.percentile(loss, 90)),
        "share_regret_above_1": float(np.mean(regret > 1.0)),
        "share_g1_passing": float(np.mean([o.median_abs_relative_bias < 0.20 for o in outcomes])),
        "min_n_agreeing": float(min(o.n_agreeing for o in outcomes)),
        "n_solve_failures": float(n_failed),
        "g5_worst_case": float(
            sum(o.beats_status_quo for o in outcomes) / (len(outcomes) + n_failed)
        ),
        "g5_ci_low": g5_low,
        "g5_ci_high": g5_high,
        "share_regret_above_1_ci_low": above_low,
        "share_regret_above_1_ci_high": above_high,
    }


def verify_control(results: Sequence[CellResult]) -> None:
    """Assert that the 0.30 column reproduces D23. Raises if it does not.

    This is the positive control. D23's numbers were produced by a harness that was not
    committed, so agreement here is the only evidence that this module measures the same
    quantities the study reported. A mismatch means the sweep is uninterpretable, not that
    D23 was wrong — either way it must stop the run rather than be written to disk.
    """
    if failures(results):
        raise AssertionError(
            f"the control cell must solve cleanly; {len(failures(results))} of "
            f"{len(results)} seeds failed SLSQP at spend_log_sd = {CONTROL_LEVEL}"
        )
    measured = gates(results)
    disagreements = [
        f"{name}: this harness {measured[name]:.4f} vs D23 {expected:.4f}"
        for name, expected in D23_CONTROL.items()
        if abs(measured[name] - expected) > CONTROL_TOLERANCE
    ]
    if disagreements:
        raise AssertionError(
            "the spend_log_sd = 0.30 control does not reproduce D23's C0 table, so the "
            "sweep is not comparable to the confirmatory result:\n  " + "\n  ".join(disagreements)
        )


Row = tuple[float, int, str, str, float]


def _rows(results: Iterable[CellResult], channel_names: Sequence[str]) -> Iterator[Row]:
    """Tidy long format: one row per (level, seed, channel, metric).

    Failed cells appear as a single `solve_failed = 1` row, so the CSV records that they were
    run rather than silently omitting them. Nothing downstream can mistake an absent seed for
    a seed that was never attempted.
    """
    for result in results:
        if isinstance(result, SolveFailure):
            yield (result.level, result.seed, "all", "solve_failed", 1.0)
            continue
        outcome = result
        for index, channel in enumerate(channel_names):
            yield (
                outcome.level,
                outcome.seed,
                channel,
                "relative_bias",
                float(outcome.relative_bias[index]),
            )
            yield (
                outcome.level,
                outcome.seed,
                channel,
                "covered",
                float(outcome.covered[index]),
            )
        for metric, value in (
            ("median_abs_relative_bias", outcome.median_abs_relative_bias),
            ("spearman", outcome.spearman),
            ("regret", outcome.regret),
            ("beats_status_quo", float(outcome.beats_status_quo)),
            ("achievable_lift_share", outcome.achievable_lift_share),
            ("shortfall_from_optimum_share", outcome.shortfall_from_optimum_share),
            ("loss_vs_status_quo_share", outcome.loss_vs_status_quo_share),
            ("n_agreeing", float(outcome.n_agreeing)),
        ):
            yield (outcome.level, outcome.seed, "all", metric, value)


def write_csv(results: Sequence[CellResult], channel_names: Sequence[str], path: Path) -> None:
    """Write the tidy long CSV, sorted so the bytes do not depend on completion order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["spend_log_sd", "seed", "channel", "metric", "value"])
        for level, seed, channel, metric, value in _rows(results, channel_names):
            writer.writerow([f"{level:.2f}", seed, channel, metric, f"{value:.10g}"])


def run_sweep(
    levels: Sequence[float], n_seeds: int, workers: int | None = None
) -> dict[float, list[CellResult]]:
    """Run every (level, seed) cell, in parallel, deterministically."""
    jobs = [(level, seed) for level in levels for seed in range(n_seeds)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_run_cell_packed, jobs, chunksize=1))
    by_level: dict[float, list[CellResult]] = {level: [] for level in levels}
    for outcome in results:
        by_level[outcome.level].append(outcome)
    return by_level


def format_table(by_level: dict[float, list[CellResult]]) -> str:
    """The reader-facing summary: five gates and the absolute figures, per level."""
    levels = sorted(by_level)
    header = ["metric", *[f"sd={level:.2f}" for level in levels], "threshold"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for name, (comparator, threshold) in GATE_THRESHOLDS.items():
        cells = []
        for level in levels:
            value = gates(by_level[level])[name]
            passed = value < threshold if comparator == "<" else value >= threshold
            cells.append(f"{value:.3f} {'pass' if passed else 'fail'}")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {comparator} {threshold:.2f} |")
    lines.append("| G6_placebo | " + " | ".join(["n/a"] * len(levels)) + " | C5/C6/C7 only |")

    for name in (
        "achievable_lift_share_median",
        "shortfall_from_optimum_median",
        "loss_vs_status_quo_median",
        "share_regret_above_1",
        "share_g1_passing",
        "g5_worst_case",
        "n_solve_failures",
    ):
        cells = [f"{descriptives(by_level[level])[name]:.4f}" for level in levels]
        lines.append(f"| {name} | " + " | ".join(cells) + " | — |")

    for name, low, high in (
        ("G5_beats_status_quo", "g5_ci_low", "g5_ci_high"),
        ("share_regret_above_1", "share_regret_above_1_ci_low", "share_regret_above_1_ci_high"),
    ):
        cells = [
            f"[{descriptives(by_level[level])[low]:.3f}, {descriptives(by_level[level])[high]:.3f}]"
            for level in levels
        ]
        lines.append(f"| {name} 95% CI | " + " | ".join(cells) + " | Wilson |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=N_SWEEP_SEEDS)
    parser.add_argument("--levels", type=float, nargs="+", default=list(SWEEP_LEVELS))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=SWEEP_CSV)
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="probe runs only; never for a result that will be reported",
    )
    args = parser.parse_args(argv)

    by_level = run_sweep(args.levels, args.seeds, args.workers)

    if not args.skip_control:
        if CONTROL_LEVEL not in by_level or args.seeds != N_SWEEP_SEEDS:
            raise SystemExit(
                f"the control cell (spend_log_sd = {CONTROL_LEVEL} at {N_SWEEP_SEEDS} seeds) "
                f"must be part of any reportable run; pass --skip-control for a probe"
            )
        verify_control(by_level[CONTROL_LEVEL])

    channel_names = simulate(sweep_params(args.levels[0]), 0).channel_names
    write_csv(
        [outcome for level in sorted(by_level) for outcome in by_level[level]],
        channel_names,
        args.out,
    )
    print(format_table(by_level))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_variable, "1")
    raise SystemExit(main())
