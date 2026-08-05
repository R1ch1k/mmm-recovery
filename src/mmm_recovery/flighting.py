"""D34's flighted-spend validity check on C0. **Not pre-registered.**

D26's sweep varies the *amplitude* of spend jitter. Jitter of any size samples a narrow band of
the response curve — it moves spend around its mean and never approaches the origin. A flighted
plan does something categorically different: it traces the curve **near zero**, and it makes
adstock decay directly observable, because sales in a dark week are carryover and nothing else.

So this is not a third exploratory arm. It asks whether **C0's spend process was
unrepresentatively hard** — the same anti-strawman question D9 and D12 asked of other knobs. If
real media plans carry identifying information that a lognormal jitter lacks, C0 understates what
a real analyst has and the headline weakens.

**Both readings were committed in D34 before this module was run**, and neither is negotiable
after the fact:

- G1 still fails → C0's spend process was not the cause; the headline stands with one fewer
  attack surface.
- G1 passes → the finding is **not** "MMM works". It is that identification depends on spend
  having gone dark, which is a property of the **media plan** rather than of the model. A team
  whose channels are always-on cannot identify its own response curves however good its
  modelling is. That is more actionable than the current headline and leads the one-pager.

Design, fixed in D34 before running: TV, video and OOH flight; search and social stay always-on,
because that is how those channels behave. Bursts of 2-6 weeks, independent per channel so that
synchronisation cannot smuggle in C1's collinearity. Each channel's **total spend is preserved**,
so this is "same budget, bought in bursts" and not "half the budget".
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np

from mmm_recovery.dgp import DGPParams, duty_cycles, media_share, simulate
from mmm_recovery.sweep import (
    CONTROL_LEVEL,
    GATE_THRESHOLDS,
    N_SWEEP_SEEDS,
    RESULTS_DIR,
    CellResult,
    descriptives,
    failures,
    gates,
    passes,
    run_cell_with,
    successes,
    sweep_params,
)

FLIGHTED_CHANNELS: Final = ("tv", "video", "ooh")
"""Brand channels are bought in bursts; search and social are budget-capped and always-on."""

BURST_WEEKS: Final = (2, 6)

FLIGHTING_CSV: Final = RESULTS_DIR / "flighting_check.csv"


def baseline_params() -> DGPParams:
    """C0 exactly as the confirmatory result ran it."""
    return sweep_params(CONTROL_LEVEL)


def flighted_params() -> DGPParams:
    """C0 with the spend *shape* changed and nothing else."""
    return replace(
        baseline_params(),
        flighted_channels=FLIGHTED_CHANNELS,
        flight_burst_weeks=BURST_WEEKS,
    )


ARMS: Final = {"baseline": baseline_params, "flighted": flighted_params}


@dataclass(frozen=True)
class ArmDiagnostics:
    """What the spend process actually did, so the arms can be compared honestly (D4)."""

    media_share: float
    duty_cycle_flighted: float
    duty_cycle_always_on: float


def _label(arm: str) -> float:
    """`SeedOutcome.level` is a float; 0.0 tags the baseline arm and 1.0 the flighted one."""
    return 0.0 if arm == "baseline" else 1.0


def _run_packed(job: tuple[str, int]) -> tuple[str, CellResult, float, dict[str, float]]:
    arm, seed = job
    params = ARMS[arm]()
    sim = simulate(params, seed)
    return arm, run_cell_with(params, seed, _label(arm)), media_share(sim), duty_cycles(sim)


def diagnose(shares: list[float], cycles: list[dict[str, float]]) -> ArmDiagnostics:
    """Realised media share and duty cycles, averaged across seeds."""
    flighted = [
        value for cycle in cycles for name, value in cycle.items() if name in FLIGHTED_CHANNELS
    ]
    always_on = [
        value for cycle in cycles for name, value in cycle.items() if name not in FLIGHTED_CHANNELS
    ]
    return ArmDiagnostics(
        media_share=float(np.mean(shares)),
        duty_cycle_flighted=float(np.mean(flighted)),
        duty_cycle_always_on=float(np.mean(always_on)),
    )


def format_table(
    by_arm: dict[str, list[CellResult]], diagnostics: dict[str, ArmDiagnostics]
) -> str:
    """The comparison D34 pre-committed to reading."""
    arms = ["baseline", "flighted"]
    lines = [
        "| metric | C0 baseline | C0 flighted | threshold |",
        "|---|---|---|---|",
    ]
    for name, (comparator, threshold) in GATE_THRESHOLDS.items():
        cells = []
        for arm in arms:
            value = gates(by_arm[arm])[name]
            verdict = "pass" if passes(value, comparator, threshold) else "fail"
            cells.append(f"{value:.3f} {verdict}")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {comparator} {threshold:.2f} |")

    for metric in (
        "share_regret_above_1",
        "share_g1_passing",
        "shortfall_from_optimum_median",
        "achievable_lift_share_median",
        "n_solve_failures",
    ):
        cells = [f"{descriptives(by_arm[arm])[metric]:.4f}" for arm in arms]
        lines.append(f"| {metric} | " + " | ".join(cells) + " | — |")

    cells = [
        f"[{descriptives(by_arm[arm])['g5_ci_low']:.3f}, "
        f"{descriptives(by_arm[arm])['g5_ci_high']:.3f}]"
        for arm in arms
    ]
    lines.append("| G5 95% CI | " + " | ".join(cells) + " | Wilson |")

    lines.append("")
    lines.append("| diagnostic | C0 baseline | C0 flighted |")
    lines.append("|---|---|---|")
    for label, attribute in (
        ("realised media share (D4)", "media_share"),
        ("duty cycle, flighted channels", "duty_cycle_flighted"),
        ("duty cycle, always-on channels", "duty_cycle_always_on"),
    ):
        values = [getattr(diagnostics[arm], attribute) for arm in arms]
        lines.append(f"| {label} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    return "\n".join(lines)


def write_csv(by_arm: dict[str, list[CellResult]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["arm", "seed", "median_abs_relative_bias", "spearman", "regret", "beats_status_quo"]
        )
        for arm in ("baseline", "flighted"):
            for outcome in successes(by_arm[arm]):
                writer.writerow(
                    [
                        arm,
                        outcome.seed,
                        f"{outcome.median_abs_relative_bias:.10g}",
                        f"{outcome.spearman:.10g}",
                        f"{outcome.regret:.10g}",
                        int(outcome.beats_status_quo),
                    ]
                )
            for failure in failures(by_arm[arm]):
                writer.writerow([arm, failure.seed, "", "", "", ""])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=N_SWEEP_SEEDS)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=FLIGHTING_CSV)
    args = parser.parse_args(argv)

    jobs = [(arm, seed) for arm in ARMS for seed in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_run_packed, jobs, chunksize=1))

    by_arm: dict[str, list[CellResult]] = {arm: [] for arm in ARMS}
    shares: dict[str, list[float]] = {arm: [] for arm in ARMS}
    cycles: dict[str, list[dict[str, float]]] = {arm: [] for arm in ARMS}
    for arm, result, share, cycle in rows:
        by_arm[arm].append(result)
        shares[arm].append(share)
        cycles[arm].append(cycle)

    diagnostics = {arm: diagnose(shares[arm], cycles[arm]) for arm in ARMS}
    print(format_table(by_arm, diagnostics))
    write_csv(by_arm, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_variable, "1")
    raise SystemExit(main())
