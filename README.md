# AI Voice of Customer Copilot for Quick-Commerce

Turns 4,620 unstructured customer reviews from **Blinkit**, **Zepto**, and **JioMart**
into evidence-backed product insights: recurring pain points, supporting evidence,
product opportunities, RICE prioritisation, and experiment plans.

> **Status: Phases 1–3 of 10 complete.** Data foundation, the corpus-derived
> product taxonomy, full exploratory analysis, and the AI enrichment pipeline are
> built and verified, with **291 passing tests**. RAG, LangGraph orchestration, and
> the Streamlit UI are scheduled — see [Roadmap](#roadmap).
>
> Start here: [`docs/EDA_FINDINGS.md`](docs/EDA_FINDINGS.md) (product intelligence
> summary) · [`docs/TAXONOMY.md`](docs/TAXONOMY.md) · [`docs/MODEL_BENCHMARK.md`](docs/MODEL_BENCHMARK.md)
> · [`reports/EDA_REPORT.md`](reports/EDA_REPORT.md)

---

## The problem

A quick-commerce PM has thousands of app reviews and no way to act on them. Reading
them does not scale; a generic AI summary is unusable because it cannot be checked.
The gap is not summarisation — it is getting from *raw feedback* to *a prioritised,
evidence-backed decision* without inventing anything along the way.

This project builds that path:

```
CUSTOMER FEEDBACK → STRUCTURED INSIGHTS → PAIN POINTS → EVIDENCE
                  → PRODUCT OPPORTUNITIES → PRIORITISATION → EXPERIMENTS
```

Two modes: **single-platform analysis** and **competitive comparison** across any
pair or all three platforms.

---

## What the data actually supports

Profiling drove the scope. These constraints are load-bearing, not disclaimers:

| Finding | Consequence for the product |
|---|---|
| **77.5% of reviews are negative** (71.5% are 1★) | A complaint-biased sample. Good for *finding problems*; unusable for measuring satisfaction. Every claim is scoped "within this review dataset". |
| **Blinkit 2.55★ vs Zepto 1.53★ vs JioMart 1.50★** | Tracks how each platform's reviews were collected, not how customers feel. Cross-platform *satisfaction ranking is deliberately not offered.* |
| **87% of reviews fall in Oct–Dec 2024**; Blinkit has 12 rows before October, Zepto has 1 | Trend analysis is scoped to three monthly buckets. No seasonality, no year-over-year, no statistical spike detection. |
| **Negative share drifts month to month** (96% → 75%) | Trends are reported as *share* changes, never raw counts — counts would measure scrape composition. |
| **8% of reviews hit a 500-character scraper cap** | Truncated reviews can lose their resolution clause, biasing sentiment negative. Flagged and excluded from outcome-dependent labels. |
| **Reviews average 1.85 themes each** | Classification must be multi-label. Single-label would be factually wrong. |
| **No user ID, app version, city, or order value** | Root-cause output is limited to *hypotheses* from text, never confirmed causes. |

Full generated profile: [`reports/data_profile.md`](reports/data_profile.md).

---

## Architecture

Layered, with pure importable logic separated from the UI so every layer is testable.

```
data/raw/reviews.csv          IMMUTABLE source, SHA-256 pinned
        │
        ▼
  [1] ingest.py               read · checksum · validate against contract
        │
        ▼
  [2] clean.py                normalise · review_id · truncation · near-duplicates
        │
        ▼
  data/interim/reviews_clean.parquet        ← Phase 1 output
        │
        ▼
  [3] profiling.py            reproducible data profile
        │
        ▼
  [4-15]  enrichment → clustering → trends → RAG → opportunities
          → RICE → experiments → LangGraph → Streamlit → evaluation
```

### Layer status

| # | Layer | Module | Status |
|---|---|---|---|
| 1 | Data ingestion | `src/voc/ingest.py` | ✅ Phase 1 |
| 2 | Data cleaning | `src/voc/clean.py` | ✅ Phase 1 |
| 3 | Profiling / EDA | `src/voc/profiling.py` | ✅ Phase 1 |
| 3b | Taxonomy discovery | `src/voc/taxonomy.py`, `discovery.py` | ✅ Phase 2 |
| 3c | EDA + figures | `src/voc/eda.py`, `plots.py` | ✅ Phase 2 |
| 4 | AI enrichment | `src/voc/enrich.py`, `llm.py`, `prompts.py` | ✅ Phase 3 |
| 5 | Pain-point discovery | `src/voc/painpoints.py` | Phase 4 |
| 6 | Embeddings + clustering | `src/voc/embed.py`, `cluster.py` | Phase 4 |
| 7 | Trend analysis | `src/voc/trends.py` | Phase 5 |
| 8 | RAG evidence retrieval | `src/voc/retrieval.py` | Phase 6 |
| 9 | Root-cause hypotheses | `src/voc/rootcause.py` | Phase 6 |
| 10 | Opportunity generation | `src/voc/opportunities.py` | Phase 7 |
| 11 | RICE prioritisation | `src/voc/rice.py` | Phase 7 |
| 12 | Experiment plans | `src/voc/experiments.py` | Phase 7 |
| 13 | LangGraph orchestration | `src/voc/graph/` | Phase 6 |
| 14 | Streamlit UI | `app/` | Phase 8 |
| 15 | Evaluation | `evaluation/` | Phase 9 |

---

## Quick start

Requires Python 3.11+ (built and tested on CPython 3.13).

```bash
git clone https://github.com/bhushandeshmaneiitkgp/AI-voice-of-Customers.git
```

```bash
python -m venv .venv
```

**Windows (PowerShell)**

```bash
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Phase 1 needs **no API key**. Copy the env template anyway so it is ready for Phase 3:

```bash
cp .env.example .env
```

Run the pipeline:

```bash
python scripts/00_profile_data.py
```

```bash
python scripts/01_build_clean.py
```

```bash
python scripts/02_discover_taxonomy.py
```

```bash
python scripts/03_run_eda.py
```

Phase 3 needs an API key for whichever provider your model uses — by default an
[OpenRouter](https://openrouter.ai/keys) key in `.env`. See every model and its cost
for this corpus without spending anything:

```bash
python scripts/04_run_enrichment.py --all --dry-run
```

Then smoke-test on 20 reviews (about one cent):

```bash
python scripts/04_run_enrichment.py --sample 20
```

Answers are cached per model in `artifacts/`, so a repeated run is free — and, for the
same reason, reports whatever the first run measured. When the numbers themselves are
the point, `--no-cache` forces a live request per review. It bypasses cache *reads*
only: existing entries are kept and updated, never cleared.

```bash
python scripts/04_run_enrichment.py --sample 20 --no-cache
```

Run the tests:

```bash
pytest -q
```

---

## Design decisions worth explaining

### Raw data is immutable — enforced, not just promised

`data/raw/reviews.csv` is opened read-only and pinned by SHA-256. A test
(`test_raw_dataset_is_unmodified`) fails if a single byte changes. Every derived
artefact is regenerable from raw by re-running the scripts.

### Cleaning flags problems; it does not delete rows

All 4,620 rows enter and all 4,620 leave, annotated. Dropping is reserved for
structurally unusable rows, and every drop is counted in `cleaning_report.json`.
Deleting truncated or duplicated reviews would silently change the denominator of
every percentage the product later shows a PM.

### Deterministic review IDs

The source has no ID column. `review_id` is a SHA-256 content hash of
`platform | date | rating | normalised_text`, truncated to 16 hex characters.
Stability matters because Phase 9's hand-labelled gold set is keyed on this ID — a
non-reproducible ID would silently invalidate the evaluation set on the next run.

### Near-duplicates are found with TF-IDF cosine + connected components

Exact duplicates are almost absent (1 pair), but templated reviews are not: ten
Zepto wallet complaints differ only in the rupee amount. Those would inflate a
cluster and manufacture a pain-point signal that is really one complaint repeated.

TF-IDF vectors are L2-normalised, so a dot product *is* cosine similarity. Pairs at
or above the threshold become graph edges, and connected components become groups —
so A~B and B~C correctly puts all three together. Chosen over MinHash/LSH because at
4.6k documents the exact computation runs in about a second, which removes a tuning
parameter that would otherwise need defending.

**Sensitivity check** — the result is stable across a wide threshold range, so `0.80`
sits on a plateau rather than a cliff:

| Threshold | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|
| Reviews flagged | 53 (1.1%) | 50 (1.1%) | 49 (1.1%) | 35 (0.8%) |
| Largest group | 18 | 15 | 10 | 7 |

### Sampling pattern is not customer behaviour

Review volume rises 26× from July to December 2024. That is the shape of the scrape,
not a demand curve, and treating it as growth is the easiest way to get a confident
wrong answer out of this dataset. Every temporal view reports normalised shares.

EDA turned this from a caveat into a **measured finding**: restricting to the only
window where all three platforms are present, **JioMart and Zepto swap rating rank**
(JioMart 1.50 → 1.64, moving above Zepto's 1.53). On the full corpus JioMart looks
like the worst-rated platform; it is not. Its mean is dragged down by 165 pre-October
reviews that no other platform has.

So the comparison window is **mandatory, not advisory** — and a test asserts the
confound so the finding cannot be quietly lost. Full analysis:
[`docs/EDA_FINDINGS.md`](docs/EDA_FINDINGS.md).

### The taxonomy was discovered, not assumed

The Phase 0 working hypothesis was 8 categories. Tested against the corpus it turned
out to be too coarse in three places and missing four areas, so the final taxonomy has
**15 product areas across 5 domains** — see [`docs/TAXONOMY.md`](docs/TAXONOMY.md).

The most consequential finding: **`wallet_and_credits` was absent from the hypothesis
and is the strongest platform discriminator in the whole corpus** — 27.9% of Zepto
reviews versus 1.2% Blinkit, a 22.7× spread. Merged into "Payments & Pricing" as the
hypothesis had it, a Zepto-specific trust crisis would have been averaged into a
rounding error.

Four concepts are kept deliberately separate, because collapsing them is what makes a
taxonomy unusable: **product_area** (where), **issue_type / strength_type** (what went
wrong or right there), **pain_point** (the specific instance, free text), and
**attributes** (sentiment, severity, intent, support escalation). An area is
polarity-neutral — a category named "Delivery Problems" cannot represent the 31% of
positive reviews that say delivery is the best thing about the app.

Discovery also showed support is **downstream**: P(support mentioned | order fulfilment
failure) is 48% against a 29.1% base rate, while pricing complaints sit at 19.2%.
People do not contact support because support is bad; they contact it because something
else broke. That is why `support_escalation` is a review-level attribute *as well as*
`customer_support` being an area — which is what makes contact-deflection analysis
possible later.

### Enrichment trusts nothing the model returns

A structured response is not a correct one. Three independent checks sit between
the model and the dataset:

1. **Schema** — Pydantic plus a strict JSON schema whose enums are injected from the
   taxonomy at runtime, so the contract cannot drift from `taxonomy.yaml`.
2. **Taxonomy validation** — every label must exist, and every issue type must belong
   to the area it was filed under. Models invent plausible categories; a real issue
   type under the wrong parent is the subtler failure, and both are caught.
3. **Grounding verification** — every `evidence_span` must appear **verbatim** in the
   review that produced it. This is the project's primary hallucination metric:
   deterministic, needs no judge model, and directly answers "is this insight actually
   supported by the review it cites".

Reviews are sent in groups of five to amortise the ~4,400-token taxonomy prompt, but
every returned `review_id` is reconciled against what was requested in **both**
directions. A skipped review is retried individually; an id that was never requested
is discarded. Without that check, one omission would silently shift every label in the
group onto the wrong rows.

### The corpus is configuration, not code

Three things are properties of a *dataset* rather than of the pipeline, and all
three live in [`config/dataset.yaml`](config/dataset.yaml):

```yaml
platforms:                       # the complete set of valid `platform` values
  - {id: blinkit, display_name: Blinkit}
date_formats: ["%d %B %Y"]       # strptime patterns, tried in order
domain:
  description: Indian quick-commerce grocery apps   # goes into the LLM prompt
```

Point `VOC_DATASET_CONFIG` at a different file and another business's reviews run
through the same ingestion, cleaning and enrichment with no code change:

```bash
VOC_DATASET_CONFIG=config/dataset_othercorp.yaml python scripts/01_build_clean.py
```

**This makes validation configurable, not optional** — the distinction that matters.
An undeclared platform is still rejected; a date matching no declared format is still
an error rather than a silently dropped row. What changed is that strictness now
travels with the dataset instead of being frozen to one of them. Under a food-delivery
profile, `swiggy` is accepted and `blinkit` is rejected — the exact inverse of the
shipped config, which a merely-permissive implementation could not do.

The domain description matters most for the LLM. Naming one industry in Python would
mean the model is told it is reading that industry's reviews regardless of what it was
handed, biasing every label invisibly. A test asserts no brand name appears in
`prompts.py`.

### One adapter reaches every open-source provider

Enrichment runs on **open-weight models via OpenRouter** by default. The key fact
that makes this cheap to support: OpenRouter, Groq, Together, Fireworks, DeepSeek,
Ollama and vLLM all speak the **OpenAI chat-completions** wire format. So there are
two adapters, not seven — and switching between those providers is a `base_url`
change in `config/models.yaml`, not an integration.

```
voc/providers/base.py              Protocol: complete(...) -> (text, usage)
voc/providers/anthropic_provider.py    first-party SDK + native Batch API
voc/providers/openai_compatible.py     OpenRouter / Groq / DeepSeek / Ollama / vLLM
```

Everything that makes enrichment trustworthy sits **above** that boundary and is
identical for every provider. That is what makes a cross-provider benchmark
meaningful: the only thing that varies between two runs is the model. Tests assert
both adapters receive a byte-identical system prompt and JSON schema.

Cost for the full 4,620-review corpus spans **144×**:

| Model | Provider | Est. cost | Structured output |
|---|---|---|---|
| `ollama` (local Llama 3.3) | local | **$0.00** | schema |
| `gptoss` | openrouter | **$0.39** | schema |
| `llama70b` *(default)* | openrouter | **$1.51** — measured **$1.64** | schema |
| `qwen72b` | openrouter | $2.11 | schema |
| `deepseek` | openrouter | $2.41 | json only |
| `haiku` | anthropic | $5.27 | schema |
| `opus` (low → high effort) | anthropic | $26 → $56 | schema |

`llama70b` is the only row with a *measured* number rather than an estimate: the
full corpus was enriched on 2026-09-02 for **$1.64**. Treat the rest as estimates
that have not yet met an invoice. An earlier version of this table said `$0.90`
for `llama70b`, taken from the cheapest of the 13 provider endpoints OpenRouter
routes across — a ~10× spread. Routing does not honour that floor, and the first
full run cost 2.07× its estimate as a result.

Two things degrade on open models, and the pipeline handles both rather than
pretending otherwise. **Schema enforcement** varies — a model declared `json_object`
gets the schema injected into its prompt instead, and output is validated identically
either way. **Verbatim quoting** is weaker: open models paraphrase, which shows up
directly as a lower grounding rate. That is a measurement, not a blocker — and
measuring it is the point.

### Model choice is configuration, not code

No model ID appears in any `.py` file — a test enforces this. Models are defined in
[`config/models.yaml`](config/models.yaml) and selected by environment variable:

```bash
VOC_ENRICHMENT_MODEL=qwen72b python scripts/04_run_enrichment.py --sample 100
```

That flag requests 100; proportional stratification rounds each stratum down
independently, so it yields 99 — which is why
[`docs/MODEL_BENCHMARK.md`](docs/MODEL_BENCHMARK.md) reports a **99-review** benchmark.

This exists so Phase 9 can run the *same* pipeline under different models and score
each against the gold set, turning a cost question into a documented result rather
than an untested assumption. The registry also records per-model API differences
(adaptive thinking vs. token budget, effort support) so Phase 3 branches correctly.

---

## Project layout

```
quickcommerce-voc-copilot/
├── config/
│   ├── settings.py            paths, tunables, secret handling
│   └── models.yaml            model registry (Decision 1)
├── data/
│   ├── raw/reviews.csv        IMMUTABLE source
│   ├── interim/               cleaned parquet + run reports  (gitignored)
│   ├── processed/             analysis outputs               (gitignored)
│   └── eval/                  gold labels (Phase 9)
├── src/voc/
│   ├── schemas.py             Pydantic contracts — the project's spine
│   ├── ingest.py              Layer 1
│   ├── clean.py               Layer 2
│   └── profiling.py           Layer 3
├── scripts/
│   ├── 00_profile_data.py
│   └── 01_build_clean.py
├── tests/                     97 tests
├── reports/data_profile.md    generated
└── requirements.txt
```

---

## Output schema

`data/interim/reviews_clean.parquet` — 4,620 rows × 20 columns.

| Column | Type | Description |
|---|---|---|
| `review_id` | string | Deterministic 16-char content hash |
| `source_row_index` | int32 | 0-based row in the raw CSV, for traceability |
| `platform` | category | `blinkit` \| `jiomart` \| `zepto` |
| `rating` | int8 | 1–5 |
| `rating_bucket` | category | `negative` (1–2) \| `neutral` (3) \| `positive` (4–5) |
| `review_date` | datetime | Parsed from `"30 December 2024"` |
| `year`, `month`, `year_month` | int16, int8, string | Grouping keys |
| `review_text` | string | Whitespace-normalised text used for analysis |
| `review_raw` | string | Original text, byte-identical to source |
| `char_len`, `word_count` | int16 | Length metrics |
| `is_truncated` | bool | Hit the 500-char scraper cap |
| `ends_without_terminal_punct` | bool | Possible mid-sentence cut |
| `has_non_latin` | bool | Non-Latin script present |
| `near_dup_group_id` | int32 | Shared group id; `-1` when unique |
| `near_dup_group_size` | int32 | Members in that group |
| `is_near_dup_representative` | bool | One `True` per group |
| `in_comparable_window` | bool | On/after 2024-10-01 (all platforms present) |

---

## Roadmap

| Phase | Scope |
|---|---|
| ✅ 1 | Scaffold, ingestion, cleaning, profiling, tests |
| ✅ 2 | Taxonomy discovery + full EDA, figures, and product intelligence summary |
| ✅ 3 | Multi-label LLM enrichment (Pydantic-validated, Batch API) |
| 4 | Embeddings, FAISS, clustering, pain-point scoring |
| 5 | Trend analysis + competitive metrics |
| 6 | RAG evidence retrieval + LangGraph orchestration |
| 7 | Opportunities, RICE, experiment plans |
| 8 | Streamlit dashboard |
| 9 | Evaluation framework + model benchmark |
| 10 | `PRODUCT_REQUIREMENTS.md`, `EVALUATION.md`, screenshots |

---

## Limitations

This dataset is a complaint-biased sample of app-store reviews, not a
representative survey of quick-commerce customers. It cannot support claims about
which platform's customers are happier, cannot attribute feedback to specific
orders or SKUs, and cannot confirm causes — only surface hypotheses with evidence.
The product is built to make those claims difficult to state by accident.
