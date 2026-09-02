"""
Turn diagnosed pain points into scored opportunities and testable experiment plans.

This is the Phase 7 deliverable. It generates candidate changes from the Phase 6
root causes, scores them with RICE, and attaches a sample-size calculation to
each so the resulting experiment could actually settle the question.

**Effort is not generated.** Reach, impact and confidence come from the corpus;
effort is a property of the codebase and the team, which no amount of review
text contains. Without it this prints RIC and refuses a final ranking. Supply
estimates to get real RICE:

    # generate opportunities and score what the data supports
    VOC_SYNTHESIS_MODEL=llama70b python scripts/09_build_roadmap.py

    # write a CSV for someone to put person-weeks into
    python scripts/09_build_roadmap.py --write-effort-template

    # score properly, once that CSV is filled in
    python scripts/09_build_roadmap.py --effort data/processed/effort_template.csv

    # estimate cost and exit
    python scripts/09_build_roadmap.py --dry-run

Reads  : data/processed/pain_points.parquet
         data/processed/root_causes.parquet
         data/processed/platform_metrics.parquet
         data/processed/area_rates_by_platform.parquet
         data/processed/reviews_enriched.parquet
Writes : data/processed/opportunities.parquet
         data/processed/rice_scores.parquet
         data/processed/experiment_plans.parquet
         reports/ROADMAP.md
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- must precede project imports
import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

from config.settings import Paths, get_settings
from voc.experiments import DEFAULT_MDE_PP, baseline_for, build_experiment_plan, plans_to_frame
from voc.llm import create_provider
from voc.opportunities import (
    DEFAULT_OPPORTUNITIES,
    MAX_OUTPUT_TOKENS,
    build_system_prompt,
    generate_opportunities,
)
from voc.providers import ProviderError
from voc.rice import (
    CONFIDENCE_WEIGHTS,
    build_rice_inputs,
    load_effort,
    to_frame,
    write_effort_template,
)

#: Metric each opportunity is measured against maps to a Phase 5 baseline.
METRIC_TO_BASELINE = {
    "negative_share": "negative_share",
    "severe_share": "severe_share",
    "escalation_rate": "escalation_rate",
    "churn_share": "churn_share",
    "praise_share": "praise_share",
}

#: Guardrail paired with each primary metric: the thing that must not get worse
#: while the target moves. Chosen so a change cannot "win" by trading one
#: customer harm for another.
GUARDRAIL = {
    "escalation_rate": "negative_share",
    "negative_share": "praise_share",
    "severe_share": "negative_share",
    "churn_share": "negative_share",
    "praise_share": "negative_share",
    "pain_point_volume": "negative_share",
}


def build_report(rice: pd.DataFrame, opportunities: pd.DataFrame,
                 plans: pd.DataFrame, months: int, has_effort: bool,
                 profile) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = []
    add = lines.append

    add("# Roadmap — Scored Opportunities and Experiment Plans")
    add("")
    add(f"**{generated}** · {len(rice)} pain points scored · "
        f"{len(opportunities)} opportunities · {len(plans)} experiment plans")
    add("")
    add("---")
    add("")

    # --- The refusal, stated before the table ------------------------------
    add("## Effort is missing, and that is deliberate")
    add("")
    if has_effort:
        add("Effort estimates were supplied, so the table below is a real RICE "
            "ranking. Rows without an estimate are listed separately and scored "
            "RIC only — they are not comparable with the scored ones.")
    else:
        add("**No effort estimates were supplied, so this is a RIC ranking, not "
            "RICE.**")
        add("")
        add("Reach, impact and confidence are properties of *the problem*, which "
            "is what customer reviews describe. Effort is a property of *the "
            "solution and the codebase that would carry it* — how many services "
            "it touches, what the migration looks like, who is free next sprint. "
            "No amount of review text contains that.")
        add("")
        add("A model asked to guess it would produce a confident number, and the "
            "arithmetic would lend that guess authority. A RICE table with an "
            "invented denominator ranks work by fiction while looking "
            "quantitative — worse than no table.")
        add("")
        add("To get a real ranking:")
        add("")
        add("```bash")
        add("python scripts/09_build_roadmap.py --write-effort-template")
        add("```")
        add("")
        add("Fill in `effort_person_weeks`, then:")
        add("")
        add("```bash")
        add("python scripts/09_build_roadmap.py --effort data/processed/effort_template.csv")
        add("```")
    add("")
    add("---")
    add("")

    # --- How confidence was derived ---------------------------------------
    add("## Confidence is measured, not felt")
    add("")
    add("Standard RICE picks confidence at 100/80/50% by feel. Here it is "
        "derived from evidence quality, so it answers *how much do we know* "
        "rather than *how sure does someone feel today*:")
    add("")
    add("| Component | Weight | What it measures |")
    add("|---|---:|---|")
    add(f"| Grounding | {CONFIDENCE_WEIGHTS['grounding']:.2f} | were the labels' quotes verbatim in the reviews |")
    add(f"| Sample | {CONFIDENCE_WEIGHTS['sample']:.2f} | is the volume enough to be a pattern |")
    add(f"| Label confidence | {CONFIDENCE_WEIGHTS['label_confidence']:.2f} | what the enrichment model reported |")
    add(f"| Mechanism | {CONFIDENCE_WEIGHTS['mechanism']:.2f} | did Phase 6 find a grounded root cause |")
    add(f"| Competitive | {CONFIDENCE_WEIGHTS['competitive']:.2f} | did a platform difference survive correction |")
    add("")
    add("---")
    add("")

    # --- Ranking -----------------------------------------------------------
    add("## Ranking")
    add("")
    add(f"Reach is **reviews per month** over {months} observed months — not "
        "customers. People who write app-store reviews are a small, "
        "self-selecting, annoyed slice of users. It is a consistent relative "
        "signal, which is what RICE needs; multiplying it by a user base would "
        "be inventing a number.")
    add("")
    header = "| # | Area | Issue | Reach/mo | Impact | Confidence |"
    rule = "|---|---|---|---:|---|---:|"
    if has_effort:
        header += " Effort | RICE |"
        rule += "---:|---:|"
    else:
        header += " RIC |"
        rule += "---:|"
    add(header)
    add(rule)
    for row in rice.itertuples():
        line = (f"| {row.rank} | `{row.product_area}` | `{row.issue_type}` | "
                f"{row.reach_per_month:.0f} | {row.impact_label} ({row.impact}) | "
                f"{row.confidence:.2f} |")
        if has_effort:
            effort = "—" if pd.isna(row.effort_person_weeks) else f"{row.effort_person_weeks:.1f}"
            score = "—" if pd.isna(row.rice) else f"**{row.rice:.1f}**"
            line += f" {effort} | {score} |"
        else:
            line += f" **{row.ric:.1f}** |"
        add(line)
    add("")
    add("---")
    add("")

    # --- Opportunities and their experiments -------------------------------
    add("## Opportunities")
    add("")
    if opportunities.empty:
        add("No opportunities survived validation.")
    for row in opportunities.itertuples():
        add(f"### {row.title}")
        add("")
        add(f"`{row.product_area}` / `{row.issue_type}`")
        add("")
        add(f"**Change.** {row.change}")
        add("")
        add(f"**Addresses.** {row.addresses_hypothesis}")
        add("")
        add(f"**Success.** `{row.primary_metric}` should **{row.expected_direction}**.")
        add("")
        add(f"**Risk if wrong.** {row.risk_if_wrong}")
        add("")

        plan = plans[(plans["title"] == row.title)]
        if not plan.empty:
            p = plan.iloc[0]
            add("**Experiment.**")
            add("")
            add(f"| | |")
            add(f"|---|---|")
            add(f"| Primary metric | `{p['primary_metric']}` |")
            add(f"| Guardrail | `{p['guardrail_metric']}` |")
            add(f"| Baseline | {p['baseline_rate'] * 100:.1f}% |")
            add(f"| Target | {p['target_rate'] * 100:.1f}% ({p['mde_pp']:.0f}pp) |")
            add(f"| Sample needed | {p['sample_per_arm']:,} per arm ({p['sample_total']:,} total) |")
            add(f"| Duration | {p['months_required']:.1f} months at review volume |")
            add(f"| Powered? | {'yes' if p['practical'] else '**no**'} |")
            add("")
            if p["notes"]:
                add(f"> {p['notes']}")
                add("")
        add("---")
        add("")

    # --- Limits ------------------------------------------------------------
    add("## Caveats")
    add("")
    add("**Sample sizes are in reviews, not users.** An experiment run on users "
        "has a different denominator and will usually reach power far sooner. "
        "These numbers say how long it would take to *see the effect in "
        "reviews*, which is the slowest possible instrument.")
    add("")
    add("**Opportunities are model-generated.** Each is tied to a validated "
        "root-cause hypothesis and names a metric this pipeline measures, which "
        "makes it checkable — not correct.")
    add("")
    add("**Impact is derived from severity, churn and escalation**, all of which "
        "are model labels over self-reported customer text. They describe how a "
        "review reads, not operational cost.")
    add("")
    add("**An underpowered experiment cannot produce a null result.** Where "
        "`Powered? no` appears, a flat outcome means the test could not have "
        "detected the effect, not that there was none.")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=10, help="Pain points to work through.")
    parser.add_argument("--opportunities", type=int, default=DEFAULT_OPPORTUNITIES,
                        help="Opportunities requested per pain point.")
    parser.add_argument("--effort", type=str, default=None,
                        help="CSV of person-week estimates. Without it, RIC only.")
    parser.add_argument("--write-effort-template", action="store_true",
                        help="Write a CSV for a human to fill in, then exit.")
    parser.add_argument("--mde", type=float, default=DEFAULT_MDE_PP,
                        help="Minimum detectable effect, percentage points.")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost and exit.")
    parser.add_argument("--yes", action="store_true", help="Skip the cost confirmation.")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format="%(levelname)-8s %(name)s | %(message)s")
    log = logging.getLogger("roadmap")
    Paths.ensure_output_dirs()

    for path in (Paths.pain_points, Paths.root_causes, Paths.enriched_reviews):
        if not path.exists():
            log.error("%s not found. Run the earlier phases first.", path.name)
            return 1

    pain_points = pd.read_parquet(Paths.pain_points).head(args.top)
    root_causes = pd.read_parquet(Paths.root_causes)
    reviews = pd.read_parquet(Paths.enriched_reviews)
    area_rates = (
        pd.read_parquet(Paths.area_rates) if Paths.area_rates.exists() else pd.DataFrame()
    )
    metrics = (
        pd.read_parquet(Paths.platform_metrics)
        if Paths.platform_metrics.exists() else pd.DataFrame()
    )

    window = reviews[reviews["in_comparable_window"]] if "in_comparable_window" in reviews else reviews
    months = max(1, window["year_month"].nunique())
    monthly_volume = len(window) / months

    effort = load_effort(args.effort) if args.effort else {}
    rice_inputs = build_rice_inputs(
        pain_points, reviews, months, root_causes, area_rates, effort
    )

    if args.write_effort_template:
        write_effort_template(rice_inputs, Paths.effort_template)
        print()
        print(f"  Effort template written to {Paths.effort_template}")
        print("  Fill in effort_person_weeks, then re-run with --effort <that file>.")
        print()
        return 0

    profile = settings.synthesis_profile
    requests = len(pain_points)
    input_tokens = requests * (len(build_system_prompt(args.opportunities)) // 4 + 400)
    output_tokens = requests * MAX_OUTPUT_TOKENS // 2
    estimate = profile.estimate_cost_usd(input_tokens, output_tokens)

    print()
    print("=" * 78)
    print("  ROADMAP")
    print("=" * 78)
    print(f"  Model        : {profile.display_name}  ({profile.model_id})")
    print(f"  Pain points  : {requests}")
    print(f"  Observed     : {months} month(s), ~{monthly_volume:,.0f} reviews/month")
    print(f"  Effort       : " + (f"{len(effort)} estimate(s) supplied"
                                  if effort else "NOT SUPPLIED — RIC only, no RICE"))
    print("-" * 78)
    print(f"  ESTIMATE     : ~{input_tokens:,} in / ~{output_tokens:,} out -> ${estimate:,.2f}")
    print("=" * 78)
    print()

    if args.dry_run:
        log.info("Dry run - no API calls made.")
        return 0

    if not args.yes:
        try:
            if input("Proceed and spend this? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Aborted.")
                return 0
        except EOFError:
            log.error("No TTY for confirmation. Re-run with --yes.")
            return 1

    try:
        provider = create_provider(profile, settings)
    except (RuntimeError, ProviderError) as exc:
        log.error("%s", exc)
        log.error("With only an OpenRouter key set, run: "
                  "VOC_SYNTHESIS_MODEL=llama70b python scripts/09_build_roadmap.py")
        return 1

    opportunity_rows, plans, issues = [], [], {}
    for index, row in enumerate(pain_points.itertuples(), start=1):
        print(f"    {index}/{requests}  {row.product_area}/{row.issue_type}", end="\r", flush=True)
        hypotheses = root_causes[
            (root_causes["product_area"] == row.product_area)
            & (root_causes["issue_type"] == row.issue_type)
        ].to_dict("records")

        result = generate_opportunities(
            row.product_area, row.issue_type,
            {
                "volume": int(row.volume),
                "mean_severity": float(row.mean_severity),
                "escalation_rate": float(row.escalation_rate),
                "churn_rate": float(row.churn_rate),
            },
            hypotheses, profile, provider, n=args.opportunities,
        )
        for issue in result.issues:
            issues[issue.kind] = issues.get(issue.kind, 0) + 1

        for item in result.opportunities:
            opportunity_rows.append(
                {
                    "product_area": row.product_area,
                    "issue_type": row.issue_type,
                    "title": item.title,
                    "change": item.change,
                    "addresses_hypothesis": item.addresses_hypothesis,
                    "primary_metric": item.primary_metric,
                    "expected_direction": item.expected_direction,
                    "risk_if_wrong": item.risk_if_wrong,
                }
            )

            baseline = baseline_for(metrics, METRIC_TO_BASELINE.get(item.primary_metric, ""))
            if baseline is None:
                # pain_point_volume has no measured platform rate; fall back to
                # the pain point's own share of reviews, which is measured.
                baseline = float(row.volume) / max(1, len(reviews))
            plans.append(
                build_experiment_plan(
                    row.product_area, row.issue_type, item.title,
                    f"{item.change} -> {item.primary_metric} will {item.expected_direction}",
                    baseline, monthly_volume,
                    item.primary_metric,
                    GUARDRAIL.get(item.primary_metric, "negative_share"),
                    mde_pp=args.mde,
                )
            )
    print()

    opportunities = pd.DataFrame(opportunity_rows)
    rice = to_frame(rice_inputs)
    plan_frame = plans_to_frame(plans)

    if not opportunities.empty:
        opportunities.to_parquet(Paths.opportunities, index=False)
    if not rice.empty:
        rice.to_parquet(Paths.rice_scores, index=False)
    if not plan_frame.empty:
        plan_frame.to_parquet(Paths.experiment_plans, index=False)
    if not effort:
        write_effort_template(rice_inputs, Paths.effort_template)

    Paths.roadmap_report.write_text(
        build_report(rice, opportunities, plan_frame, months, bool(effort), profile),
        encoding="utf-8",
    )

    powered = int(plan_frame["practical"].sum()) if not plan_frame.empty else 0
    print("=" * 78)
    print("  COMPLETE")
    print("=" * 78)
    print(f"  Opportunities   : {len(opportunities)}")
    print(f"  Experiment plans: {len(plan_frame)}  ({powered} adequately powered)")
    print(f"  Scoring         : " + ("RICE" if effort else "RIC (effort not supplied)"))
    print(f"  Validation      : {issues or 'no issues'}")
    print("-" * 78)
    if not effort:
        print(f"  Effort template : {Paths.effort_template}")
    print(f"  Report          : {Paths.roadmap_report}")
    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
