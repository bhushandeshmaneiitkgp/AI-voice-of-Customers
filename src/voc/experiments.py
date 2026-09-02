"""
Layer 12 -- experiment plans.

An opportunity is a claim that changing something will improve something. This
turns each one into a plan that could actually settle it: a metric, a guardrail,
the sample size needed, and how long that takes at observed volume.

The sample-size calculation is the part that earns its place. "Run an A/B test"
is advice; "you need 6,300 reviews per arm to detect a 3-point move, which is
2.9 months at current volume" is a decision. Most experiment plans skip it, and
the result is underpowered tests whose null results get read as "no effect"
when they only ever meant "we could not have seen one".

Baselines come from Phase 5's measured platform rates rather than from
assumption, so the arithmetic is anchored to this corpus. The same caveat
carries through: these are review rates, not user rates, and an experiment run
on users will not have the same denominator.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

#: Conventional defaults. Stated as constants because an experiment plan that
#: hides its alpha and power is not a plan, it is a suggestion.
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80

#: Minimum detectable effect, in absolute percentage points, when none is
#: given. Three points is roughly the smallest move worth a release on metrics
#: at these baselines; smaller effects need sample sizes this corpus cannot
#: reach in a quarter.
DEFAULT_MDE_PP = 3.0

#: Above this many months an experiment is not worth planning around: the
#: product will have changed underneath it. Reported rather than enforced.
IMPRACTICAL_DURATION_MONTHS = 6.0


@dataclass
class SampleSizePlan:
    """What it would take to detect the stated effect."""

    baseline_rate: float
    target_rate: float
    mde_pp: float
    alpha: float
    power: float
    per_arm: int
    total: int
    monthly_volume: float
    months_required: float
    practical: bool

    @property
    def verdict(self) -> str:
        if self.practical:
            return f"{self.months_required:.1f} months at current volume"
        return (
            f"{self.months_required:.1f} months at current volume — too slow to "
            "act on; widen the effect you are willing to call a win, or "
            "instrument the product directly instead of waiting on reviews"
        )


@dataclass
class ExperimentPlan:
    """One testable plan attached to one opportunity."""

    product_area: str
    issue_type: str
    title: str
    hypothesis: str
    primary_metric: str
    #: What must NOT get worse. An experiment without one can "win" by moving
    #: the target metric at the expense of something more important.
    guardrail_metric: str
    sample: SampleSizePlan
    rollback: str
    notes: list[str] = field(default_factory=list)


def required_sample_per_arm(
    baseline_rate: float,
    mde_pp: float,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> int:
    """Two-proportion sample size per arm, for a two-sided test.

    Uses the standard normal approximation. Exact methods differ by a few
    percent at these sizes, which is far inside the error introduced by
    guessing the baseline -- so the extra precision would be false comfort.
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(f"baseline_rate must be strictly between 0 and 1, got {baseline_rate}")
    if mde_pp <= 0:
        raise ValueError("mde_pp must be positive -- an effect of zero needs infinite data")

    delta = mde_pp / 100.0
    target = baseline_rate - delta if baseline_rate - delta > 0 else baseline_rate + delta
    if not 0.0 < target < 1.0:
        raise ValueError(
            f"a {mde_pp}pp move from {baseline_rate:.1%} leaves the valid range"
        )

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    pooled = (baseline_rate + target) / 2

    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_beta * math.sqrt(
            baseline_rate * (1 - baseline_rate) + target * (1 - target)
        )
    ) ** 2
    return int(math.ceil(numerator / (delta**2)))


def plan_sample_size(
    baseline_rate: float,
    monthly_volume: float,
    mde_pp: float = DEFAULT_MDE_PP,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> SampleSizePlan:
    """Sample size plus how long it takes at the volume actually observed."""
    per_arm = required_sample_per_arm(baseline_rate, mde_pp, alpha, power)
    total = per_arm * 2
    months = total / monthly_volume if monthly_volume > 0 else float("inf")

    delta = mde_pp / 100.0
    target = baseline_rate - delta if baseline_rate - delta > 0 else baseline_rate + delta

    return SampleSizePlan(
        baseline_rate=baseline_rate,
        target_rate=target,
        mde_pp=mde_pp,
        alpha=alpha,
        power=power,
        per_arm=per_arm,
        total=total,
        monthly_volume=monthly_volume,
        months_required=months,
        practical=months <= IMPRACTICAL_DURATION_MONTHS,
    )


def build_experiment_plan(
    product_area: str,
    issue_type: str,
    title: str,
    hypothesis: str,
    baseline_rate: float,
    monthly_volume: float,
    primary_metric: str,
    guardrail_metric: str,
    mde_pp: float = DEFAULT_MDE_PP,
) -> ExperimentPlan:
    """Assemble one plan, with the caveats that apply to it attached."""
    sample = plan_sample_size(baseline_rate, monthly_volume, mde_pp)

    notes: list[str] = []
    if not sample.practical:
        notes.append(
            f"Underpowered at review volume: {sample.total:,} reviews needed, "
            f"{monthly_volume:,.0f} arrive per month."
        )
    if baseline_rate < 0.02:
        notes.append(
            f"Baseline of {baseline_rate:.1%} is very low; rare-event effects "
            "need far more data than the default MDE assumes."
        )

    return ExperimentPlan(
        product_area=product_area,
        issue_type=issue_type,
        title=title,
        hypothesis=hypothesis,
        primary_metric=primary_metric,
        guardrail_metric=guardrail_metric,
        sample=sample,
        rollback=(
            f"Revert if {guardrail_metric} degrades beyond its own confidence "
            "interval, or if the primary metric moves the wrong way at any "
            "interim look."
        ),
        notes=notes,
    )


def baseline_for(
    metrics: pd.DataFrame, metric_key: str, platform: str | None = None
) -> float | None:
    """Pull a measured baseline rate out of the Phase 5 metrics table.

    Returns None rather than a default when the metric is absent: silently
    substituting a plausible-looking rate would produce a sample size for an
    experiment nobody can actually run.
    """
    if metrics.empty or "metric" not in metrics.columns:
        return None

    subset = metrics[metrics["metric"] == metric_key]
    if platform:
        subset = subset[subset["platform"] == platform]
    if subset.empty:
        return None

    # Weight by reviews so a pooled baseline is not dragged by a small platform.
    if "reviews" in subset.columns and subset["reviews"].sum() > 0:
        return float(
            (subset["rate"] * subset["reviews"]).sum() / subset["reviews"].sum()
        )
    return float(subset["rate"].mean())


def plans_to_frame(plans: Sequence[ExperimentPlan]) -> pd.DataFrame:
    """Flatten plans for storage."""
    return pd.DataFrame(
        [
            {
                "product_area": plan.product_area,
                "issue_type": plan.issue_type,
                "title": plan.title,
                "hypothesis": plan.hypothesis,
                "primary_metric": plan.primary_metric,
                "guardrail_metric": plan.guardrail_metric,
                "baseline_rate": round(plan.sample.baseline_rate, 4),
                "target_rate": round(plan.sample.target_rate, 4),
                "mde_pp": plan.sample.mde_pp,
                "alpha": plan.sample.alpha,
                "power": plan.sample.power,
                "sample_per_arm": plan.sample.per_arm,
                "sample_total": plan.sample.total,
                "months_required": round(plan.sample.months_required, 2),
                "practical": plan.sample.practical,
                "rollback": plan.rollback,
                "notes": " | ".join(plan.notes),
            }
            for plan in plans
        ]
    )
