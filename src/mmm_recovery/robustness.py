"""D33's optimiser-bound robustness check. **Not pre-registered.**

§3 allows `m_c` ∈ [0, 3.0], so the model may recommend tripling a channel on a response curve
fitted from data that never went near that spend. Dew, Padilla & Shchetkina (2024) impose the
opposite restriction twice and say why: they "remove the candidate allocations that assign to at
least one of the channels a level of spending that falls outside of the historical range" (p.29),
because it "ensures that our optimal solution is not relying purely on extrapolation of the
response function" (fn.13, p.29).

That makes the bound the most likely way C0's headline gets dismissed. A median regret of 2.355
with 160 of 200 seeds above 100% is exactly what an optimiser extrapolating an S-shaped Hill
curve into unobserved spend would produce, and if the failure were an artefact of the constraint
set rather than of the estimator, this is where it would show.

It is not. At every bound down to 1.3 the recommendation is still worse than doing nothing at
the median. See D33 for the table and the reading.

**Both solves take the same bound.** D18 requires the truth solve and the recommendation solve to
be identically configured, so the bound is applied to `optimal_allocation` on each surface rather
than through `recommended_allocation`, which by design exposes no optimiser parameters at all.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from mmm_recovery.dgp import simulate
from mmm_recovery.estimator import RidgeMMM, allocation_regret
from mmm_recovery.sweep import (
    CONTROL_LEVEL,
    N_SWEEP_SEEDS,
    OPTIMISER_SEED,
    RESULTS_DIR,
    SLSQP_SIGNATURE,
    sweep_params,
    wilson_interval,
)
from mmm_recovery.truth import optimal_allocation, response_surface

BOUNDS: Final = ((0.0, 1.3), (0.0, 1.5), (0.0, 2.0), (0.0, 3.0), (0.7, 1.3))
"""`(floor, cap)` on m_c. `(0.0, 3.0)` is §3's pre-registered pair.

The first four vary only the cap and leave the floor at zero, so a channel can still be switched
off. `(0.7, 1.3)` is D35's two-sided planning guardrail: no channel may be cut by more than 30%
or raised by more than 30%, which is the rule a governed team actually operates under and the one
that forbids the "delete OOH" move carrying most of the true optimum's headroom (D20).
"""

PREREGISTERED_BOUND: Final = (0.0, 3.0)

GUARDRAIL: Final = (0.7, 1.3)

BOUND_CSV: Final = RESULTS_DIR / "optimiser_bound_check.csv"


@dataclass(frozen=True)
class BoundOutcome:
    """One (bound, seed) cell. `None` fields are impossible; a failed solve is dropped."""

    floor: float
    bound: float
    seed: int
    regret: float
    beats_status_quo: bool
    achievable_lift_share: float
    on_upper_bound: int
    """How many channels the model pushed to the upper bound. The extrapolation exposure."""

    on_lower_bound: int
    """How many channels the model zeroed. Added after the first version of this check claimed
    the recommendations were interior on the strength of `on_upper_bound` alone, which counts
    only one of the two bounds. They are not interior: the binding constraint is m_c = 0."""

    truth_on_lower_bound: int
    """The same count for the *true* optimum, so the comparison is like-for-like. The truth
    zeroes OOH — the one channel whose true ROAS is below 1 — so zeroing per se is not the
    error; zeroing more channels than the truth does is."""


def run_seed(seed: int) -> list[BoundOutcome]:
    """Fit once, then solve at every bound. The fit does not depend on the bound."""
    sim = simulate(sweep_params(CONTROL_LEVEL), seed)
    surface = response_surface(sim)
    fit = RidgeMMM().fit(sim.spend, sim.sales, seed)

    outcomes: list[BoundOutcome] = []
    for floor, bound in BOUNDS:
        try:
            optimum = optimal_allocation(
                surface, seed=OPTIMISER_SEED, max_multiplier=bound, min_multiplier=floor
            )
            recommended = optimal_allocation(
                fit.surface, seed=OPTIMISER_SEED, max_multiplier=bound, min_multiplier=floor
            )
        except ValueError as exc:
            if SLSQP_SIGNATURE not in str(exc):
                raise
            continue
        model_sales = surface.total_sales(recommended.multipliers)
        outcomes.append(
            BoundOutcome(
                floor=floor,
                bound=bound,
                seed=seed,
                regret=allocation_regret(surface, recommended.multipliers, optimum),
                beats_status_quo=bool(model_sales > optimum.status_quo_sales),
                achievable_lift_share=optimum.achievable_lift / optimum.status_quo_sales,
                on_upper_bound=int(np.isclose(recommended.multipliers, bound).sum()),
                on_lower_bound=int(np.isclose(recommended.multipliers, floor, atol=1e-8).sum()),
                truth_on_lower_bound=int(np.isclose(optimum.multipliers, floor, atol=1e-8).sum()),
            )
        )
    return outcomes


def summarise(outcomes: list[BoundOutcome]) -> dict[tuple[float, float], dict[str, float]]:
    """Per-bound-pair decision gates, with a Wilson interval on the rate."""
    summary: dict[tuple[float, float], dict[str, float]] = {}
    for floor, bound in BOUNDS:
        cells = [
            outcome for outcome in outcomes if outcome.bound == bound and outcome.floor == floor
        ]
        if not cells:
            continue
        regret = np.array([cell.regret for cell in cells])
        beats = sum(cell.beats_status_quo for cell in cells)
        low, high = wilson_interval(beats, len(cells))
        summary[floor, bound] = {
            "n": float(len(cells)),
            "median_regret": float(np.median(regret)),
            "beats_status_quo": beats / len(cells),
            "beats_ci_low": low,
            "beats_ci_high": high,
            "achievable_lift_share": float(
                np.median([cell.achievable_lift_share for cell in cells])
            ),
            "share_regret_above_1": float(np.mean(regret > 1.0)),
            "mean_channels_on_bound": float(np.mean([cell.on_upper_bound for cell in cells])),
            "mean_channels_zeroed": float(np.mean([cell.on_lower_bound for cell in cells])),
            "mean_truth_channels_zeroed": float(
                np.mean([cell.truth_on_lower_bound for cell in cells])
            ),
            "share_any_channel_zeroed": float(np.mean([cell.on_lower_bound > 0 for cell in cells])),
        }
    return summary


def format_table(summary: dict[tuple[float, float], dict[str, float]]) -> str:
    header = (
        "| m_c range | G4 median regret | G5 beats status quo | 95% CI | achievable lift "
        "| share regret > 1 | at cap | at floor (model) | at floor (truth) | n |"
    )
    lines = [header, "|" + "---|" * 10]
    for pair in sorted(summary):
        row = summary[pair]
        note = " (§3)" if pair == PREREGISTERED_BOUND else (" (D35)" if pair == GUARDRAIL else "")
        lines.append(
            f"| [{pair[0]:.1f}, {pair[1]:.1f}]{note} | {row['median_regret']:.3f} "
            f"| {row['beats_status_quo']:.3f} "
            f"| [{row['beats_ci_low']:.3f}, {row['beats_ci_high']:.3f}] "
            f"| {100 * row['achievable_lift_share']:.3f}% | {row['share_regret_above_1']:.3f} "
            f"| {row['mean_channels_on_bound']:.3f} | {row['mean_channels_zeroed']:.2f} "
            f"| {row['mean_truth_channels_zeroed']:.2f} | {row['n']:.0f} |"
        )
    lines.append("")
    lines.append(
        "G4 is NOT comparable across rows: a tighter range shrinks the achievable lift that "
        "forms regret's denominator. G5 is a rate and is comparable. (D35)"
    )
    return "\n".join(lines)


def write_csv(outcomes: list[BoundOutcome], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "min_multiplier",
                "max_multiplier",
                "seed",
                "regret",
                "beats_status_quo",
                "achievable_lift_share",
                "channels_on_upper_bound",
                "channels_zeroed_model",
                "channels_zeroed_truth",
            ]
        )
        for outcome in sorted(outcomes, key=lambda o: (o.floor, o.bound, o.seed)):
            writer.writerow(
                [
                    f"{outcome.floor:.1f}",
                    f"{outcome.bound:.1f}",
                    outcome.seed,
                    f"{outcome.regret:.10g}",
                    int(outcome.beats_status_quo),
                    f"{outcome.achievable_lift_share:.10g}",
                    outcome.on_upper_bound,
                    outcome.on_lower_bound,
                    outcome.truth_on_lower_bound,
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=N_SWEEP_SEEDS)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=BOUND_CSV)
    args = parser.parse_args(argv)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        nested = list(pool.map(run_seed, range(args.seeds), chunksize=1))
    outcomes = [outcome for group in nested for outcome in group]

    summary = summarise(outcomes)
    print(format_table(summary))
    write_csv(outcomes, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_variable, "1")
    raise SystemExit(main())
