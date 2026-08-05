"""Google Meridian as a verify-not-trust anchor — is the C0 failure a property of `RidgeMMM`?

PREREGISTRATION.md §4 names Meridian as the anchor and pre-commits the interpretation, which
is why running it here needs no amendment:

    Purpose is verify-not-trust: to establish that any conclusion is a property of MMM as a
    method rather than an artefact of `RidgeMMM`. … If Meridian's priors rescue a condition
    where `RidgeMMM` fails, the correct interpretation — the prior is supplying information
    the data does not contain — is stated explicitly.

**Optional.** `google-meridian` and `tensorflow-probability` are the `[meridian]` extra and
the core grid runs without them. Importing this module without them raises with the install
command rather than failing obscurely at first use.

Two things kept deliberately identical to the `RidgeMMM` path, because the anchor is only
evidence if the comparison is like for like:

* **The estimand.** Meridian's `incremental_outcome` is a counterfactual — outcome with the
  channel's execution present against it set to zero — which is §3's `do()` definition, not a
  coefficient. `analyzer.incremental_outcome` is used rather than any ROI summary for exactly
  that reason.
* **What the model may see.** Spend and sales. Meridian gets no control columns at all; its
  time-varying baseline is its own knot spline, which is how the package models trend and
  seasonality by default. Passing our Fourier terms in would be tuning it toward the DGP.

**Priors are left at their defaults and must stay there.** Meridian's default is an ROI prior,
and moving it is precisely the intervention §4 says to interpret rather than perform. If it
fails on defaults, that is the result.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mmm_recovery.dgp import SimResult

_INSTALL_HINT = (
    "google-meridian is not installed. It is the optional [meridian] extra, deliberately kept "
    "out of the core grid: `uv sync --extra meridian`."
)

try:
    from meridian.analysis import analyzer as _analyzer
    from meridian.data import load as _load
    from meridian.model import model as _model
    from meridian.model import spec as _spec

    MERIDIAN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where the extra is absent
    MERIDIAN_AVAILABLE = False


ANCHOR_CONDITIONS = ("C0", "C3", "C6", "C7")
"""§4 runs the anchor on these four only. C0 is the one that matters for K1."""

N_ANCHOR_SEEDS = 10
"""§4's seed count for the anchor."""

INTERVAL_LEVEL = 0.90
"""The same nominal interval `RidgeMMM` is graded on, so G2 compares like with like."""

_EPOCH = "2015-01-05"
"""A fixed Monday. Meridian wants dates; the study must not read a clock (CLAUDE.md rule 4)."""


@dataclass(frozen=True)
class SamplingSpec:
    """MCMC settings. **Not priors** — those stay at Meridian's defaults, per §4.

    These are a compute budget, and they are reported alongside every result so the reader can
    see what was spent. `r_hat_ceiling` is the convergence gate: a chain that has not mixed
    produces a posterior mean that is an artefact of where it started, and reporting one as a
    failure of the method would be exactly the strawman K3 exists to prevent.
    """

    n_chains: int = 4
    n_adapt: int = 500
    n_burnin: int = 500
    n_keep: int = 1000
    r_hat_ceiling: float = 1.2


@dataclass(frozen=True)
class AnchorFit:
    """One Meridian fit on one dataset.

    Attributes:
        channel_names: (C,) in the order the arrays use.
        contribution: (C,) posterior-mean incremental outcome, £k over the horizon — the same
            estimand `truth.incremental_contribution` computes on the DGP.
        interval: (C, 2) the 5th and 95th posterior percentiles, £k.
        worst_r_hat: the largest R-hat over all sampled parameters.
        converged: whether `worst_r_hat` is within the spec's ceiling.
    """

    channel_names: tuple[str, ...]
    contribution: NDArray[np.float64]
    interval: NDArray[np.float64]
    worst_r_hat: float
    converged: bool


def _require_meridian() -> None:
    if not MERIDIAN_AVAILABLE:
        raise ImportError(_INSTALL_HINT)


def to_input_data(sim: SimResult) -> Any:
    """Convert one simulated dataset into Meridian's `InputData`. Spend and sales only.

    A single national geo, weekly. Media *execution* is spend, because spend is all the study
    generates and all a real analyst in this setup would have; `media` and `media_spend`
    therefore carry the same series, which is Meridian's documented handling for a
    spend-only model.

    No control columns are supplied. That is not an oversight — see the module docstring.
    """
    _require_meridian()
    n_weeks = sim.spend.shape[0]
    names = list(sim.channel_names)
    dates = pd.date_range(_EPOCH, periods=n_weeks, freq="W-MON").strftime("%Y-%m-%d")

    frame = pd.DataFrame({"time": dates, "geo": "national", "kpi": sim.sales, "population": 1.0})
    for index, name in enumerate(names):
        frame[f"media_{name}"] = sim.spend[:, index]
        frame[f"spend_{name}"] = sim.spend[:, index]

    loader = _load.DataFrameDataLoader(
        df=frame,
        kpi_type="revenue",
        coord_to_columns=_load.CoordToColumns(
            time="time",
            geo="geo",
            kpi="kpi",
            population="population",
            media=[f"media_{name}" for name in names],
            media_spend=[f"spend_{name}" for name in names],
        ),
        media_to_channel={f"media_{name}": name for name in names},
        media_spend_to_channel={f"spend_{name}": name for name in names},
    )
    return loader.load()


def fit_anchor(sim: SimResult, seed: int, sampling: SamplingSpec | None = None) -> AnchorFit:
    """Fit Meridian at its default priors and read off the interventional contribution.

    Args:
        sim: one simulated dataset. Only `spend`, `sales` and `channel_names` are read; the
            latent series on it are never touched, exactly as for `RidgeMMM`.
        seed: explicit, per CLAUDE.md rule 4.
        sampling: MCMC budget. Priors are **not** configurable here by design.

    Returns:
        The fit, with the R-hat needed to decide whether its numbers mean anything.
    """
    _require_meridian()
    spec = sampling if sampling is not None else SamplingSpec()

    meridian = _model.Meridian(input_data=to_input_data(sim), model_spec=_spec.ModelSpec())
    meridian.sample_posterior(
        n_chains=spec.n_chains,
        n_adapt=spec.n_adapt,
        n_burnin=spec.n_burnin,
        n_keep=spec.n_keep,
        seed=seed,
    )

    # The keyword form, not the positional one: Meridian 1.7 deprecates passing the model
    # positionally, and the warning is real rather than cosmetic — that path is scheduled for
    # removal and a pinned reproduction should not break on a minor upgrade. Both keywords are
    # required together; supplying only `model_context` raises.
    engine = _analyzer.Analyzer(
        model_context=meridian.model_context, inference_data=meridian.inference_data
    )
    # (n_chains, n_draws, n_channels), aggregated over geos and time — £k over the horizon.
    outcome = np.asarray(engine.incremental_outcome(use_posterior=True), dtype=np.float64)
    draws = outcome.reshape(-1, outcome.shape[-1])

    tail = 100.0 * (1.0 - INTERVAL_LEVEL) / 2.0
    # `rhat_summary` reports per-parameter-block rows; `max_rhat` is the worst within each
    # block, so the worst over the whole model is the max of that column. Taken from the
    # summary rather than from `get_rhat` directly because the summary is what the package
    # documents, and it is the number a reader would reproduce.
    summary = engine.rhat_summary(bad_rhat_threshold=spec.r_hat_ceiling)
    worst = float(np.nanmax(summary["max_rhat"].to_numpy(dtype=np.float64)))

    return AnchorFit(
        channel_names=tuple(sim.channel_names),
        contribution=draws.mean(axis=0),
        interval=np.column_stack(
            [np.percentile(draws, tail, axis=0), np.percentile(draws, 100.0 - tail, axis=0)]
        ),
        worst_r_hat=worst,
        converged=worst <= spec.r_hat_ceiling,
    )
