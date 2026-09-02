"""
Draw the reviews a human should hand-label, and write the sheet to label them on.

This is the first half of Phase 9. It makes no API calls and costs nothing; the
expensive input is an afternoon of somebody's attention, and the job here is to
spend it well.

    # default: 100 randomly sampled + up to 50 where two models disagreed
    python scripts/10_build_goldset.py

    # a smaller pilot pass, to check the guide before committing an afternoon
    python scripts/10_build_goldset.py --random 25 --disagreement 10

    # skip the disagreement stratum entirely
    python scripts/10_build_goldset.py --disagreement 0

Two strata come out, and they are scored separately for the rest of the
project's life:

* **random** -- proportionally stratified by platform and rating bucket. The
  only stratum that can support a corpus-level accuracy claim, because it is the
  only one drawn without reference to what the models found hard.
* **disagreement** -- reviews where the benchmarked models produced different
  area sets. Biased toward difficulty on purpose, so it finds failure modes
  fast and its accuracy must never be quoted as the corpus figure.

Writes three files, and the split is deliberate:

  data/eval/gold_template.csv       what the annotator fills in
  data/eval/gold_provenance.parquet which stratum each review came from
  data/eval/ANNOTATION_GUIDE.md     generated from the live taxonomy

The template carries no model prediction and no stratum. Both would tell the
annotator what somebody else thought, and a reference set that has been told
what to expect is not a reference set.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- must precede project imports
import argparse
import logging
import sys

import pandas as pd

from config.settings import Paths, get_settings
from voc.taxonomy import get_taxonomy
from evaluation.agreement import area_sets, load_cache
from evaluation.goldset import (
    build_annotation_guide,
    build_gold_sample,
    build_provenance,
    build_template,
    disagreement_ids,
)

SEED = 42


def collect_disagreements(models: list[str]) -> tuple[list[str], str]:
    """Reviews two cached models labelled differently, and a note on the coverage.

    Returns an empty list rather than failing when fewer than two caches exist:
    the random stratum is the one that carries the accuracy claim, and a missing
    second model should cost the triage stratum, not the whole run.
    """
    caches = {}
    for key in models:
        payloads = load_cache(Paths.enrichment_cache(key))
        if payloads:
            caches[key] = payloads

    if len(caches) < 2:
        return [], (
            "Only "
            + (f"one model cache ({next(iter(caches))}) was" if caches else "no model caches were")
            + " found, so no disagreement stratum could be drawn."
        )

    (left_name, left), (right_name, right) = sorted(caches.items())[:2]
    left_areas, right_areas = area_sets(left), area_sets(right)
    ids = disagreement_ids(left_areas, right_areas)
    overlap = len(set(left_areas) & set(right_areas))
    return ids, (
        f"{left_name} vs {right_name}: {len(ids)} of {overlap} co-labelled reviews "
        f"disagree on the product-area set."
    )


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random", type=int, default=settings.gold_random_size,
                        help="Reviews in the random stratum.")
    parser.add_argument("--disagreement", type=int, default=settings.gold_disagreement_size,
                        help="Reviews in the model-disagreement stratum.")
    parser.add_argument("--seed", type=int, default=SEED, help="Sampling seed.")
    parser.add_argument("--models", nargs="*", default=["llama70b", "qwen72b"],
                        help="Model keys whose caches define the disagreement stratum.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing template.")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format="%(levelname)s  %(message)s")
    Paths.ensure_output_dirs()

    if not Paths.enriched_reviews.exists():
        print("Enriched reviews not found. Run: python scripts/04_run_enrichment.py --all")
        return 1

    # Overwriting a part-filled template destroys work that cannot be
    # regenerated, so it takes an explicit flag. Everything else in this
    # pipeline is reproducible; annotation is the one thing that is not.
    if Paths.gold_template.exists() and not args.force:
        print(f"{Paths.gold_template} already exists. Re-run with --force to replace it.")
        print("If it has been filled in, copy it to gold_labels.csv first -- the")
        print("annotations cannot be regenerated.")
        return 1

    reviews = pd.read_parquet(Paths.enriched_reviews)
    taxonomy = get_taxonomy()

    hard_ids, coverage_note = collect_disagreements(args.models)
    hard_ids = [rid for rid in hard_ids if rid in set(reviews["review_id"])]

    sample = build_gold_sample(
        reviews,
        n_random=args.random,
        hard_ids=hard_ids,
        n_hard=args.disagreement,
        seed=args.seed,
    )
    if sample.empty:
        print("No reviews sampled -- the enriched corpus is empty.")
        return 1

    strata = sample["stratum"].value_counts().to_dict()

    template = build_template(sample)
    provenance = build_provenance(sample, args.seed)
    guide = build_annotation_guide(taxonomy, len(sample), strata)

    template.to_csv(Paths.gold_template, index=False, encoding="utf-8")
    provenance.to_parquet(Paths.gold_provenance, index=False)
    Paths.gold_guide.write_text(guide, encoding="utf-8")

    print(f"\nGold set drawn -- {len(sample)} reviews, seed {args.seed}\n")
    for name, count in sorted(strata.items()):
        print(f"  {name:<14} {count:>4}")
    print(f"\n  {coverage_note}")
    if args.disagreement and strata.get("disagreement", 0) < args.disagreement:
        print(f"  Disagreement stratum is capped by that overlap, not by the corpus.")

    print(f"\n  template   {Paths.gold_template.relative_to(Paths.root)}")
    print(f"  guide      {Paths.gold_guide.relative_to(Paths.root)}")
    print(f"  provenance {Paths.gold_provenance.relative_to(Paths.root)}")
    print(
        "\nThe template shows no model prediction and no stratum, on purpose:\n"
        "an annotator shown a plausible label agrees with it far more often than\n"
        "they would have chosen it, and the gold set drifts toward the system it\n"
        "is meant to audit.\n"
    )
    print(f"Fill it in, save as {Paths.gold_labels.name}, then run:")
    print("  python scripts/11_run_evaluation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
