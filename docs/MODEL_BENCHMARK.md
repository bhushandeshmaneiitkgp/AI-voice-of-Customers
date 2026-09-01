# Model Benchmark — Enrichment

**2026-08-21** · 99 reviews · identical prompt, schema, and validators · [`config/models.yaml`](../config/models.yaml)

Input to the Phase 9 evaluation framework. The point of the configurable-model
architecture was to turn "which model should we use?" from an assumption into a
measurement; this is that measurement.

---

## Method

Both models received:

- the **same 99 reviews** — a stratified sample by platform × rating bucket, so a
  78%-negative corpus still exercises the strength vocabulary. The flag was
  `--sample 100`; proportional stratification samples each stratum at `frac` and
  rounds down independently, so the strata sum to 99. Every figure below is over
  99, and the run is described as a 99-review benchmark for that reason
- the **same system prompt**, generated from `config/taxonomy.yaml`
- the **same JSON schema**, with enums injected from the taxonomy
- the **same three validators**: schema → taxonomy → grounding

A test (`test_both_providers_receive_the_identical_schema`) asserts the first two,
because a benchmark where the adapter varies measures the adapter, not the model.

The star rating is deliberately withheld from the prompt, so `sentiment` is derived
from text alone.

---

## Results

| Metric | **llama70b** | **qwen72b** |
|---|---|---|
| Model | `meta-llama/llama-3.3-70b-instruct` | `qwen/qwen-2.5-72b-instruct` |
| **Coverage** | **99/99 (100%)** | 95/99 (96.0%) |
| **Grounding** (spans verbatim) | 98.2% | **98.7%** |
| Fully-grounded reviews | 96.0% | 96.0% |
| Areas per review | 2.43 | 2.47 |
| Output tokens per review | **251** | 881 |
| Mean confidence | 0.868 | 0.918 |
| **Cost, full 4,620 corpus** | est. ~$0.96 → **actual $1.64** | est. ~$3.20 (unmeasured) |

### Validation issues

| Issue | llama70b | qwen72b |
|---|---|---|
| `unparseable_response` | 2 | **6** |
| `unknown_area` (invented category) | 4 | 2 |
| `duplicate_label` | 4 | **0** |
| `ungrounded_evidence` | 4 | 1 |
| `unknown_issue_type` | 3 | 2 |
| `missing_polarity` | 3 | 2 |

The split is consistent: **qwen has better taxonomy discipline, llama has better
JSON reliability.** qwen's 6 parse failures are what cost it 4 reviews of coverage.

---

## What this overturned

**The prediction was wrong.** Before running, the stated expectation was 50–70%
grounding for open models against 85–95% for a frontier model, on the reasoning that
smaller models paraphrase rather than quote. Both open models landed at **98%+**.

That was the single largest risk identified in moving off a frontier provider, and it
did not materialise. Recording it because a benchmark that only ever confirms the
prior is not doing any work.

**A second assumption also fell.** The registry described qwen as "often better at
strict JSON adherence". It produced three times as many unparseable responses. The
note in `config/models.yaml` now records the contradiction rather than quietly
dropping the claim.

---

## Model agreement — the useful signal

Measured on the 95 reviews both models labelled:

| Field | Agreement |
|---|---|
| `sentiment` | 96.8% |
| `customer_intent` | 90.5% |
| `support_escalation` | 89.5% |
| `product_area` (Jaccard) | **82.5%** |
| `product_area` (exact set match) | **58/95 (61%)** |

The models agree strongly on *how the customer feels* and much less on *which product
areas apply*. Only 61% of reviews get an identical area set.

This is the most useful number in the document. It says the disagreement is
concentrated in exactly the task the Phase 9 gold set exists to adjudicate, and that
a single-model run would give a false impression of certainty about area assignment.
It also means **inter-model agreement is a usable proxy for label difficulty**: the
39% of reviews where they differ are the ones a human should label first.

Both models independently ranked `customer_support` as the largest area (41 and 42
labels), matching the Phase 2 keyword-probe estimate of 28.0%. Areas-per-review of
2.43 and 2.47 also sit close to the probe estimate of 2.19 — the multi-label design
is corroborated by three independent methods.

---

## The validators earned their place

Not theoretical — each caught something in this run:

**The elision check** rejected stitched quotes from *both* models, e.g.
`"Products are not availab...y use are not available"`. This is precisely the
paraphrase-instead-of-quote failure the grounding metric exists to catch, and it was
caught at the schema boundary before it could enter the dataset.

**The individual-retry path** recovered 9 llama reviews from failed groups, taking
coverage from ~91% to 100%. Without it, grouping 5 reviews per request would have
cost real coverage.

**Taxonomy validation** caught invented categories in both models (4 and 2). These
were well-formed, plausible-sounding labels that simply are not in the taxonomy —
undetectable by schema validation alone.

---

## Caveats

**qwen's numbers are not from a single clean run.** Its first attempt hit OpenRouter's
`in_flight_budget_exhausted` limit (HTTP 402) on 32 requests — an account concurrency
ceiling, not a model failure. It was completed at lower concurrency with a different
`--reviews-per-request` (3 vs 5), which inflates its input-token count through extra
system-prompt repeats. The **output tokens per review** figure (251 vs 881) is the
clean cost signal; the $3.20 extrapolation is derived from it rather than from the raw
two-run total of $5.08.

**99 reviews is a small sample.** Differences of a few percentage points here are not
significant. The findings that carry weight are the large ones: 3.5× output verbosity,
the 61% exact-match rate, and grounding being high for both rather than low.
(Since resolved for llama70b: the full 4,620-run reproduced its grounding to within
0.2 points — see *What the full run then measured*.)

**No ground truth yet.** Agreement is not accuracy. Both models could be
wrong in the same way, and neither number becomes an accuracy claim until the Phase 9
hand-labelled gold set exists.

---

## What the full run then measured

The benchmark picked llama70b; the full corpus was enriched with it on
**2026-09-02**. The 99-review sample held up:

| Metric | 99-review benchmark | Full corpus (4,620) |
|---|---|---|
| Coverage | 99/99 (100%) | **4,568/4,620 (98.9%)** |
| Grounding (spans verbatim) | 98.2% | **98.4%** |
| Fully-grounded reviews | 96.0% | 96.6% |
| Areas per review | 2.43 | 2.36 |
| Largest area | `customer_support` | `customer_support` (17.8%) |

Coverage fell 1.1 points at 47× the scale and grounding did not move. The
individual-retry path did the work again: 411 reviews went missing from group
responses and **359 (87.3%) were recovered**, without which coverage would have
been 90.0%.

**The cost estimate was wrong, and not by rounding.** Estimated $0.79, billed
**$1.6353** — 2.07×. About $0.11 of that is real extra tokens, because each of
the 411 individual retries repeats the ~4,451-token system prompt. The other
~$0.74 is a pricing error: `config/models.yaml` held **$0.10/$0.32 per MTok**,
which is DeepInfra — the *cheapest* of the 13 provider endpoints OpenRouter
routes this model across. The spread runs to $1.04/$1.04 at Together, roughly
10×, and the blended rate actually charged was **$0.2462/MTok**.

The registry now records the measured effective rate rather than the floor.
Pinning a provider in the request would buy the floor price back, and is the
obvious next lever if the corpus grows — but it trades cost against the
availability that load-balancing provides, so it is a decision, not a fix.

This is the same failure the qwen pricing note below already records once: a
number in the registry that nobody had checked against an invoice. It has now
happened twice, which makes it a property of the registry rather than an
accident.

---

## Decision

**llama70b**, for the full corpus.

Equal grounding, better coverage, cleaner JSON, and ~3.3× cheaper. qwen's advantage in
taxonomy discipline does not offset losing 4% of reviews to parse failures — and the
taxonomy errors are caught and reported by the validators anyway, while dropped
reviews are simply absent.

**Not chosen on price.** `gptoss` (`openai/gpt-oss-120b`) is now the cheapest entry in
the registry at $0.03/$0.17 per MTok, below llama70b. It is deliberately not the
default because it has not been benchmarked, and selecting on price alone is the exact
mistake the qwen pricing error above already caused once.

### Reproduce

`--sample 100` is what was run, and is kept verbatim; it selects the same 99 reviews
under seed 42.

```bash
VOC_ENRICHMENT_MODEL=llama70b python scripts/04_run_enrichment.py --sample 100
```

```bash
VOC_ENRICHMENT_MODEL=qwen72b python scripts/04_run_enrichment.py --sample 100 --concurrency 2
```

Per-model caches in `artifacts/enrichment_cache_<model>.json` keep runs isolated, so a
re-run costs nothing and cannot cross-contaminate a comparison.

### Re-benchmarking on fresh answers

That last property is also a trap. Both caches are now populated, so re-running either
command replays stored answers: coverage and agreement still hold, but `requests_made`,
token counts, latency and cost describe the original run, not the new one. Any
comparison against a *new* model would then be measuring one live model against one
replay.

`--no-cache` is the escape hatch. It makes every lookup miss, so each review costs a
live request and the resulting metrics belong to that run:

```bash
VOC_ENRICHMENT_MODEL=llama70b python scripts/04_run_enrichment.py --sample 100 --no-cache
```

It bypasses cache **reads** only. Existing entries are still loaded and written back,
so the file is added to and never cleared — a fresh measurement does not cost the work
already paid for. The run report records `"cache_bypassed": true`, so a later reader can
tell a fresh run from a replayed one without having to reconstruct the command.
