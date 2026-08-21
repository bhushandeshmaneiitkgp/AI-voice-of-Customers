# AI Voice of Customer Copilot for Quick-Commerce

Turns 4,620 unstructured customer reviews from **Blinkit**, **Zepto**, and **JioMart**
into evidence-backed product insights: recurring pain points, supporting evidence,
product opportunities, RICE prioritisation, and experiment plans.

> **Status: Phase 1 of 10 complete.** Data foundation (ingestion, cleaning,
> profiling, tests) is built and verified. AI enrichment, RAG, LangGraph
> orchestration, and the Streamlit UI are scheduled — see [Roadmap](#roadmap).

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
| 4 | AI enrichment | `src/voc/enrich.py` | Phase 3 |
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

### Model choice is configuration, not code

No model ID appears in any `.py` file — a test enforces this. Models are defined in
[`config/models.yaml`](config/models.yaml) and selected by environment variable:

```bash
VOC_ENRICHMENT_MODEL=haiku VOC_SYNTHESIS_MODEL=opus python scripts/03_run_enrichment.py
```

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
| 2 | Product-area taxonomy discovery from the corpus |
| 3 | Multi-label LLM enrichment (Pydantic-validated, Batch API) |
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
