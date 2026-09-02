"""
Run AI enrichment over the cleaned corpus.

Requires an API key for the selected model's provider (see .env.example).
The model — and therefore the provider — is chosen by environment variable,
never hardcoded:

    # see every model, provider and price for this workload; costs nothing
    python scripts/04_run_enrichment.py --all --dry-run

    # smoke test — 20 reviews on the default open model, costs ~1 cent
    python scripts/04_run_enrichment.py --sample 20

    # benchmark two models on the SAME reviews (the Phase 9 deliverable).
    # --sample 100 yields 99 reviews: strata round down independently.
    VOC_ENRICHMENT_MODEL=llama70b python scripts/04_run_enrichment.py --sample 100
    VOC_ENRICHMENT_MODEL=qwen72b  python scripts/04_run_enrichment.py --sample 100

    # re-benchmark with fresh live answers -- bypasses cached reads so token,
    # cost and coverage figures describe THIS run. Adds to the cache, never
    # clears it, so nothing already paid for is lost.
    VOC_ENRICHMENT_MODEL=llama70b python scripts/04_run_enrichment.py --sample 100 --no-cache

    # full corpus; --batch is honoured only where the provider supports it,
    # otherwise it falls back to concurrent live requests
    python scripts/04_run_enrichment.py --all --batch

Reads  : data/interim/reviews_clean.parquet   (read-only)
         config/taxonomy.yaml
Writes : data/processed/reviews_enriched.parquet
         data/processed/review_labels.parquet
         data/processed/enrichment_report.json
         artifacts/enrichment_cache_<model>.json

Every run prints a cost estimate first and asks before spending money.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- must precede project imports
import argparse
import json
import logging
import sys

import pandas as pd

from config.settings import Paths, get_settings, load_model_registry
from voc.enrich import (
    EnrichmentCache,
    build_run_report,
    cache_key,
    collect_batch_results,
    enrich_sync,
    RunAborted,
    poll_batch,
    stratified_sample,
    submit_batch,
    to_dataframes,
)
from voc.llm import (
    DEFAULT_REVIEWS_PER_REQUEST,
    create_provider,
    estimate_cost,
    resolve_effort,
)
from voc.providers import ProviderError
from voc.prompts import build_system_prompt
from voc.taxonomy import get_taxonomy


class RefusingProvider:
    """Satisfies the provider protocol and refuses to make a request.

    Used by ``--from-cache``, where the guarantee wanted is not "we intend not
    to call the API" but "calling it is impossible". A flag that promises to
    spend nothing should be enforced by something other than the reader's trust
    in the surrounding code.
    """

    name = "none"
    supports_batch = False

    def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise ProviderError(
            "--from-cache reached a review with no stored enrichment. Refusing to "
            "spend; the cache filter is wrong and the run stopped rather than bill.",
            retryable=False,
        )


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sample", type=int, metavar="N", help="Enrich a stratified sample of N reviews.")
    selection.add_argument("--all", action="store_true", help="Enrich the full corpus.")
    parser.add_argument("--batch", action="store_true", help="Use the Batch API (50%% cheaper, async).")
    parser.add_argument(
        "--reviews-per-request", type=int, default=DEFAULT_REVIEWS_PER_REQUEST,
        help="Reviews per API call. Higher amortises the prompt; missing ones are retried.",
    )
    parser.add_argument(
        "--effort", default=settings.enrichment_effort,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Override the model's default effort. Lower is cheaper; thinking tokens bill as output.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="Parallel requests for providers without a batch endpoint.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore cached answers and call the API for every review. Results are still "
             "written to the cache; existing entries are kept, never deleted.",
    )
    parser.add_argument(
        "--from-cache", action="store_true",
        help="Rebuild the dataset from cached enrichments only. Never calls the API.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost and exit without calling the API.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format="%(levelname)-8s %(name)s | %(message)s")
    log = logging.getLogger("enrich")
    Paths.ensure_output_dirs()

    if not Paths.clean_reviews.exists():
        log.error("Cleaned dataset not found. Run: python scripts/01_build_clean.py")
        return 1

    taxonomy = get_taxonomy()
    profile = settings.enrichment_profile
    frame = pd.read_parquet(Paths.clean_reviews)
    frame["platform"] = frame["platform"].astype(str)
    frame["rating_bucket"] = frame["rating_bucket"].astype(str)

    if args.sample:
        # Stratify by platform and rating bucket so a small sample still
        # contains positives and every platform -- a uniform sample of a 77.6%
        # negative corpus would barely exercise the strength vocabulary.
        #
        # Uses groupby().sample() rather than groupby().apply(): in pandas 3.0
        # apply() operates on each group *excluding* the grouping columns, so
        # the result silently loses `platform` and `rating_bucket`. sample()
        # returns whole rows and keeps every column.
        frame = stratified_sample(frame, args.sample, seed=args.seed)

        missing = {"platform", "rating_bucket", "review_text", "review_id"} - set(frame.columns)
        if missing:
            log.error("Sampling dropped required column(s): %s", sorted(missing))
            return 1

        log.info(
            "Stratified sample: %d reviews across %d platform(s), buckets=%s",
            len(frame),
            frame["platform"].nunique(),
            dict(frame["rating_bucket"].value_counts()),
        )

    if args.from_cache:
        # Re-materialise the parquet outputs from answers already paid for.
        # Needed whenever the *flattening* changes rather than the model -- a
        # fixed polarity column, a new derived field -- where re-running the
        # enrichment would spend money to obtain answers already on disk, and
        # would also re-attempt the reviews that failed last time, silently
        # changing coverage and every figure derived from it.
        if args.no_cache:
            log.error("--from-cache and --no-cache are contradictory.")
            return 1
        stored = EnrichmentCache(Paths.enrichment_cache(profile.key))
        before = len(frame)
        frame = frame[frame["review_id"].map(lambda r: cache_key(r, profile) in stored)]
        log.info(
            "Rebuilding from cache: %d of %d reviews have a stored enrichment. "
            "The remaining %d are left exactly as they were.",
            len(frame), before, before - len(frame),
        )
        if frame.empty:
            log.error("No cached enrichments for %s; nothing to rebuild.", profile.key)
            return 1

    system_prompt = build_system_prompt(taxonomy)
    effort = resolve_effort(profile, args.effort)
    estimate = estimate_cost(
        profile, len(frame), system_prompt, args.reviews_per_request, effort=effort
    )

    print()
    print("=" * 78)
    print("  AI ENRICHMENT")
    print("=" * 78)
    print(f"  Provider         : {profile.provider}"
          + (f"  ({profile.base_url})" if profile.base_url else ""))
    print(f"  Model            : {profile.display_name}  ({profile.model_id})")
    print(f"  Structured output: {profile.structured_output}")
    print(f"  Selected via     : VOC_ENRICHMENT_MODEL={profile.key}")
    print(f"  Thinking         : {profile.thinking_style}"
          + (f", effort={effort}" if effort else " (effort unsupported by this model)"))
    print(f"  Reviews          : {len(frame):,}")
    print(f"  Prompt           : ~{len(system_prompt) // 4:,} tokens (cached across requests)")
    batch_note = "Batch API" if (args.batch and profile.provider == "anthropic") else f"live, {args.concurrency} concurrent"
    print(f"  Transport        : {batch_note}")
    print(f"  Cache            : "
          + ("REBUILD ONLY (--from-cache) — every review is already stored"
             if args.from_cache else
             "BYPASSED (--no-cache) — every review billed live; existing entries kept"
             if args.no_cache else "read-through — cached reviews cost nothing"))
    print("-" * 78)
    if args.from_cache:
        print("  ESTIMATE         : $0.00 — no request can be made on this path")
    else:
        print(f"  ESTIMATE         : {estimate.summary(args.batch)}")
        print(f"  (standard ${estimate.usd_standard:,.2f} / batch ${estimate.usd_batch:,.2f}. "
              "Includes estimated thinking tokens; excludes prompt-cache savings.)")

    if args.dry_run:
        print("-" * 78)
        print("  OPTIONS for this workload (best available pricing per provider):")
        print(f"  {'key':<12}{'provider':<12}{'effort':<8}{'est. $':>9}   {'struct':<12}notes")
        for key, candidate in load_model_registry().items():
            levels = ("low", "medium", "high") if candidate.supports_effort else (None,)
            for level in levels:
                candidate_effort = resolve_effort(candidate, level)
                option = estimate_cost(
                    candidate, len(frame), system_prompt,
                    args.reviews_per_request, effort=candidate_effort,
                )
                # Only Anthropic has a discounted batch endpoint; everyone else
                # pays list price, so compare each at its own best rate.
                price = option.usd_batch if option.batch_available else option.usd_standard
                print(
                    f"  {key:<12}{candidate.provider:<12}{candidate_effort or 'n/a':<8}"
                    f"{price:>9,.2f}   {candidate.structured_output:<12}"
                    f"{'no batch discount' if not option.batch_available else ''}"
                )
        print()
        print("  Open models are ~10x cheaper but paraphrase more, which shows up as a")
        print("  lower grounding rate. Benchmark before committing: run --sample 100 on")
        print("  two models (99 reviews after stratified rounding) and compare grounding")
        print("  and validation issues. Add --no-cache to re-benchmark on fresh answers.")
        print("  That comparison is itself a Phase 9 deliverable.")
    print("=" * 78)
    print()

    if args.dry_run:
        log.info("Dry run - no API calls made.")
        return 0

    if not args.yes and not args.from_cache:
        try:
            if input("Proceed and spend this? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Aborted.")
                return 0
        except EOFError:
            log.error("No TTY for confirmation. Re-run with --yes to proceed non-interactively.")
            return 1

    if args.from_cache:
        # A provider that cannot spend, rather than a promise not to. The frame
        # has already been narrowed to cached reviews so nothing should reach
        # this, and if the narrowing is ever wrong the run fails loudly instead
        # of quietly billing.
        provider = RefusingProvider()
    else:
        try:
            provider = create_provider(profile, settings)
        except (RuntimeError, ProviderError) as exc:
            log.error("%s", exc)
            return 1

    # A read bypass, not a reset: the file is still loaded and still written,
    # so a fresh benchmark costs its own requests without discarding anyone
    # else's. The cost estimate above already assumed every review is billed,
    # so --no-cache makes the run match the quote rather than beat it.
    cache = EnrichmentCache(
        Paths.enrichment_cache(profile.key), read_through=not args.no_cache
    )
    if args.no_cache and len(cache):
        log.info(
            "Bypassing %d cached enrichment(s) in %s; they are preserved and "
            "will be updated, not removed.",
            len(cache), Paths.enrichment_cache(profile.key).name,
        )

    use_batch = args.batch and provider.supports_batch
    if args.batch and not provider.supports_batch:
        log.warning(
            "Provider %r has no batch endpoint; running %d concurrent live requests instead.",
            provider.name, args.concurrency,
        )

    if use_batch:
        batch_id, group_map = submit_batch(
            frame, taxonomy, profile, provider, args.reviews_per_request, effort=effort
        )
        print(f"  Batch submitted: {batch_id}")
        print("  Polling every 60s. Most batches finish within an hour; safe to Ctrl-C —")
        print(f"  the batch keeps running and can be collected later with id {batch_id}.")

        def report(batch) -> None:
            counts = batch.request_counts
            print(f"    status={batch.processing_status} "
                  f"succeeded={counts.succeeded} errored={counts.errored} "
                  f"processing={counts.processing}")

        poll_batch(provider, batch_id, on_poll=report)
        result = collect_batch_results(
            provider, batch_id, frame, group_map, taxonomy, cache=cache, profile=profile
        )
    else:
        def progress(done: int, total: int) -> None:
            print(f"    request {done}/{total}", end="\r", flush=True)

        try:
            result = enrich_sync(
                frame, taxonomy, profile, provider,
                reviews_per_request=args.reviews_per_request,
                cache=cache, progress=progress, effort=effort,
                max_concurrency=args.concurrency,
            )
        except RunAborted as exc:
            # Whatever succeeded before the breaker tripped is still worth
            # keeping: a re-run resumes from it instead of re-paying.
            cache.save()
            print()
            log.error("RUN ABORTED: %s", exc)
            log.error("Partial results kept in %s -- a re-run resumes from them.",
                      Paths.enrichment_cache(profile.key).name)
            return 2
        print()

    cache.save()

    reviews, labels = to_dataframes(result, frame)
    report = build_run_report(result, frame, profile, use_batch, cache_bypassed=args.no_cache)

    if not reviews.empty:
        reviews.to_parquet(Paths.enriched_reviews, index=False)
    if not labels.empty:
        labels.to_parquet(Paths.enriched_labels, index=False)

    if args.from_cache:
        # The run report describes an API run: coverage against what was
        # *requested*, grounding measured at parse time, issues raised by the
        # validators. A rebuild made no requests and re-validated nothing, so
        # writing one here would replace a true record with a report claiming
        # 100% coverage and no grounding measurement at all.
        log.info(
            "Left %s untouched -- it records the run that produced these answers, "
            "which this rebuild did not repeat.",
            Paths.enrichment_report.name,
        )
    else:
        Paths.enrichment_report.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

    grounding = report["grounding"]
    print()
    print("=" * 78)
    print("  ENRICHMENT COMPLETE")
    print("=" * 78)
    print(f"  Enriched         : {report['reviews_enriched']:,} / {report['reviews_requested']:,} "
          f"({report['coverage_pct']:.1f}%)")
    print(f"  Labels produced  : {len(labels):,}  "
          f"({len(labels) / max(1, len(reviews)):.2f} areas per review)")
    print(f"  Requests / cache : {report['requests_made']} made, {report['cache_hits']} reused")
    if grounding["mean_rate"] is not None:
        print(f"  Evidence grounded: {grounding['mean_rate'] * 100:.1f}% of spans verified verbatim")
        print(f"                     {grounding['fully_grounded_pct']:.1f}% of reviews fully grounded")
    if report["issue_counts"]:
        print(f"  Validation issues: {report['issue_counts']}")
    else:
        print("  Validation issues: none")
    if report["failed_review_ids"]:
        print(f"  FAILED           : {len(report['failed_review_ids'])} review(s) — see report JSON")
    usage = report["usage"]
    if usage:
        print(f"  Tokens           : {usage.get('input_tokens', 0):,} in / "
              f"{usage.get('output_tokens', 0):,} out | "
              f"cache read {usage.get('cache_read_input_tokens', 0):,}")
    print("-" * 78)
    print(f"  Reviews : {Paths.enriched_reviews}")
    print(f"  Labels  : {Paths.enriched_labels}")
    print(f"  Report  : {Paths.enrichment_report}")
    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
