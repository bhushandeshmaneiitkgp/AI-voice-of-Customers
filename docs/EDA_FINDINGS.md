# EDA Findings & Product Intelligence Summary

**Phase 2** · 4,620 reviews · Blinkit, Zepto, JioMart · generated evidence in
[`reports/EDA_REPORT.md`](../reports/EDA_REPORT.md) · figures in [`reports/figures/`](../reports/figures)

Reproduce everything with `python scripts/03_run_eda.py`. No LLM was used in this
phase — every number below is deterministic and traceable to the cleaned dataset.

---

## 1. Methodology

Five deterministic passes over `data/interim/reviews_clean.parquet`:

| Pass | What it measures | Module |
|---|---|---|
| **Profile** | Volume, ratings, dates, lengths, quality flags | `profile_dataset` |
| **Temporal** | Monthly volume, per-platform coverage, comparison-window search, sampling drift | `analyse_temporal` |
| **Rating** | Distribution, per-platform means, over time, **confound test** | `analyse_ratings` |
| **Text** | Lengths, truncation bias, document-frequency terms, templates | `analyse_text` |
| **Theme** | Keyword-probe shares, full corpus vs comparison window | `analyse_themes` |

Two methodological choices shaped everything:

**Document frequency, not term frequency.** Terms are counted once per review.
"How many customers mentioned this" is the product question; raw frequency lets one
long rant dominate.

**Every comparison is computed twice** — once on the full corpus, once restricted to
the comparison window. If a difference disappears when date coverage is equalised, it
was an artefact. Section 3 shows this was not a hypothetical concern.

---

## 2. The distinction this phase rests on

> **DATASET SAMPLING PATTERN** ≠ **CUSTOMER BEHAVIOUR**

Review volume runs 83 (Jul) → 181 → 157 → 822 → 1,000 → **2,212** (Dec). That is a
26× rise in six months.

**This is the shape of the scrape, not a demand curve.** Presenting it as growth,
or reading rising complaint counts as deteriorating service, would be the single
easiest way to produce a confident and completely wrong product decision from this
dataset.

The mechanism is visible in the data. The **negative share of what was collected
drifts month to month** — 96.4% in July, 74.8% in October, 75.5% in December. So a
rising complaint count could be more scraping, a more negative scrape, or genuinely
worse service, and **the data cannot distinguish them.**

**Consequence for Phase 5:** trend analysis must use *share of that month's reviews*
and *share of that month's negative reviews*. Raw counts over time are not
interpretable and must not be plotted.

---

## 3. The confound is real, not hypothetical

Restricting to the comparison window equalises date coverage. If platform rating
differences were purely artefacts of *when* each was scraped, they would move.

| Platform | Mean (full) | Mean (window) | Shift | Rank change |
|---|---|---|---|---|
| blinkit | 2.55 | 2.55 | −0.01 | 1st → 1st |
| **jiomart** | **1.50** | **1.64** | **+0.14** | **3rd → 2nd** |
| **zepto** | **1.53** | **1.53** | −0.00 | **2nd → 3rd** |

**JioMart and Zepto swap rank.** JioMart's full-corpus mean is dragged down by its
pre-October reviews — 165 reviews from 2020–2024 that *no other platform has*. On the
full corpus JioMart looks like the worst-rated platform. Within the only period where
all three are comparable, it rates above Zepto.

This is direct, measured proof that **full-corpus cross-platform rating comparison on
this dataset produces the wrong answer.** It is not a caveat added out of caution; it
is a demonstrated error. Blinkit's top position is stable under both views.

> A test (`test_full_corpus_rating_comparison_is_confounded`) asserts this so the
> finding cannot be quietly lost in a future refactor.

### Recommended comparison window

> ### **2024-10-01 → 2024-12-31**
> 4,034 reviews · 87.3% of the corpus · Blinkit 1,368 · Zepto 1,739 · JioMart 927

**Selection rule:** a month qualifies only when *every* platform has ≥ 50 reviews in
it. Below that threshold a single templated complaint moves a share by several points.

Only 2024-10, 2024-11 and 2024-12 qualify. Before October, Blinkit contributes **12**
reviews in total and Zepto **1** — outside the window the corpus is effectively
JioMart-only.

**This window is derived algorithmically, not chosen.** It independently reproduces
the `COMPARABLE_WINDOW_START` constant that Phase 1 set by manual inspection, and a
test asserts the two agree.

**Cost of this constraint:** three monthly buckets. No seasonality, no year-over-year,
no statistical spike detection. Trend claims in Phase 5 must be scoped accordingly.

---

## 4. Product Intelligence Summary

### What are customers most visibly talking about?

By document frequency in negative reviews: **order** (1,701), **customer** (1,262),
**delivery** (1,225), **service** (911), **worst** (885), **refund** (603). In
positive reviews the vocabulary inverts entirely: **delivery** (456), **good** (271),
**fast** (158), **minutes** (111).

The same word — *delivery* — leads both lists. That is the clearest possible
demonstration of why keyword frequency cannot classify: it cannot separate "delivery
was fast" from "delivery was late". **Frequency sizes a topic; it cannot label one.**

### Strongest apparent pain-point areas

Probe estimates within the comparison window (**lower bounds** — see §6):

| Rank | Area | Share | Note |
|---|---|---|---|
| 1 | `customer_support` | 28.0% | Largest, and **downstream** of the others |
| 2 | `delivery_reliability` | 22.4% | Also the top strength — bimodal |
| 3 | `product_quality` | 19.5% | |
| 4 | `app_experience` | 16.7% | |
| 5 | `refunds` | 15.0% | Near-zero praise |

### Apparent customer strengths

Measured as an area's share of positive reviews ÷ its share of negative reviews:

| Area | pos:neg | Reading |
|---|---|---|
| `pricing_and_charges` | **1.59×** | Value is actively praised |
| `delivery_reliability` | **1.47×** | Speed is the category's core promise, and it lands |
| `product_quality` | **1.42×** | Freshness praised as often as criticised |
| `offers_and_promotions` | **1.37×** | |
| `availability_and_range` | **1.22×** | Range is the strongest single positive discriminator |

And the mirror image — surfaces that **only ever fail**:

| Area | pos:neg |
|---|---|
| `order_lifecycle` | 0.09× |
| `refunds` | 0.16× |
| `customer_support` | 0.25× |
| `wallet_and_credits` | 0.26× |

**Nobody writes a review to praise a refund.** These are invisible when they work and
infuriating when they do not — pure defect-reduction targets, a materially different
PM strategy from the strength areas where the play is amplification.

### Which issues are platform-specific?

Within the comparison window, spread ≥ 2×:

| Platform | Elevated areas | Signature |
|---|---|---|
| **Zepto** | `wallet_and_credits` **27.9%** vs 1.2% / 3.0% (**22.5×**), `payments` 15.6%, `offers_and_promotions` 16.0% | **A money-trust problem.** Stored value customers cannot spend, plus the highest support complaints (34.5%). |
| **JioMart** | `order_lifecycle` **22.5%** vs 8.7% (2.6×) | **A fulfilment-reliability problem.** Orders cancelled unilaterally; highest refund complaints (17.3%). |
| **Blinkit** | `pricing_and_charges` **16.7%** vs 6.9% (2.4×), `product_quality` 23.2% | **A value-perception problem** — platform and handling fees — on the strongest positive base of the three. |

**These signatures survive the window restriction.** The largest movement in any
area/platform share when date coverage is equalised is **4.3 pp**, so they are not
artefacts of coverage — unlike the rating ranking in §3, which is.

### Which issues are common across platforms?

Spread < 1.5×: `delivery_reliability`, `refunds`, `app_experience`,
`delivery_partner_conduct`, `availability_and_range`, `serviceability`.

**Read this as category-level, not company-level.** Late deliveries, slow refunds,
buggy apps and courier conduct are roughly equally present everywhere. They are the
cost of doing business in quick commerce — table stakes, not differentiators. The
platform-specific areas above are where competitive advantage actually sits.

### Which areas deserve deeper AI investigation?

Ranked by *what keyword matching cannot resolve*:

1. **`customer_support` (28.0%)** — the largest area and the one probes handle worst.
   Support is *mentioned* constantly but is only sometimes the actual problem.
   Separating "support failed me" from "X failed and support didn't help" is
   irreducibly a language-understanding task.
2. **`wallet_and_credits` (Zepto, 27.9%)** — biggest competitive signal in the corpus.
   Needs the exact failure mode: balance vanished, or visible-but-unspendable? Those
   are different bugs with different owners.
3. **`delivery_reliability` (22.4%)** — bimodal. Top pain point *and* top strength;
   probes cannot split them.
4. **`order_lifecycle` (JioMart, 22.5%)** — is the platform cancelling, or can the
   customer not cancel? Opposite fixes.
5. **Severity everywhere** — nothing deterministic distinguishes "delivery was 10
   minutes late" from "₹2,000 gone, no response in three weeks".

### What should AI enrichment prioritise?

1. **Multi-label area assignment with evidence spans.** Reviews average 2.19 areas;
   only 29% touch exactly one. Single-label output would discard most of the content.
2. **Separating support-as-problem from support-as-symptom.** P(support | fulfilment
   failure) = 48% against a 29.1% base rate. Getting this wrong either double-counts
   support or hides which upstream failures generate contact.
3. **Severity**, the field with no deterministic proxy and the largest effect on
   prioritisation.
4. **Sentiment from text, never from rating.** The corpus contains sarcastic 5★
   reviews describing cancellations.
5. **`customer_intent`** — separating feature requests and churn warnings from
   ordinary complaints. Both are high-value and both are invisible to keywords.

---

## 5. Platform comparison

Descriptive, restricted to the comparison window where cross-platform.

| | Blinkit | Zepto | JioMart |
|---|---|---|---|
| Reviews (full corpus) | 1,380 | 1,740 | 1,500 |
| Reviews (window) | 1,368 | 1,739 | 927 |
| Mean rating (window) | 2.55 | 1.53 | 1.64 |
| % negative (full) | 57.7% | 85.8% | 86.3% |
| % positive (full) | 36.7% | 10.6% | 10.5% |
| Median length | 274 chars | 310 chars | 277 chars |
| % truncated | 6.9% | 8.6% | 8.4% |
| Date coverage | 2024-07 → 12 | 2024-08 → 12 | **2020-07 → 2024-12** |
| Signature theme | pricing / quality | wallet / payments | cancellations / refunds |

**Sampling limitations that gate every cell above:**

- The corpus is **77.6% negative**. These are complaint distributions, not
  satisfaction measurements.
- **Coverage is radically unequal.** JioMart spans 54 months; Blinkit and Zepto are
  effectively three-month datasets.
- Platforms may have been scraped under **different sort orders or filters**. Blinkit's
  36.7% positive share against ~10.5% for the others is more plausibly a collection
  difference than a satisfaction difference.
- **This is not a survey.** App-store reviewers are self-selected toward the extremes.

**Therefore:** "Within this review dataset, Blinkit has a higher share of positive
reviews." **Not:** "Blinkit customers are more satisfied."

---

## 6. Dataset limitations

| Limitation | Evidence | Mitigation |
|---|---|---|
| **Complaint-biased sample** | 77.6% negative, 71.5% 1★ | Never present an area share as "% of customers affected" |
| **Extreme temporal concentration** | December alone = 47.9% | Trend analysis limited to 3 monthly buckets |
| **Unequal platform coverage** | Blinkit 12 and Zepto 1 review before October | Comparison window enforced |
| **Sampling composition drifts** | Negative share 96.4% → 74.8% → 75.5% | Trends as shares, never counts |
| **Rating comparison confounded** | **JioMart/Zepto swap rank when equalised** | Rating comparisons window-only |
| **Truncation bias** | 370 reviews (8.0%) at cap; **1.36★ vs 1.87★**, 90.8% vs 76.4% negative | `is_truncated` flag; severity lower-confidence on these |
| **Templated reviews** | 18 groups, 49 reviews; largest differs only in a rupee amount | Flagged not deleted; counts reported raw and representative-only |
| **Probe recall is limited** | 8.9% match no probe; residual is recall misses | All shares labelled lower bounds |
| **No identifiers** | No user ID, app version, city, order value | Root cause = hypotheses only, never confirmed causes |

**On truncation** — truncated reviews average **1.36★** against **1.87★** for intact
ones, and are **90.8%** negative against **76.4%**. The plausible mechanism is that
they are cut *before any resolution is described*: a review ending "…but support
sorted it out" loses its ending. This is why severity and resolution labels must carry
lower confidence on these rows.

---

## 7. Recommended taxonomy for Phase 3

Validated in Phase 2 and unchanged by this analysis. Full rationale:
[`docs/TAXONOMY.md`](TAXONOMY.md). Machine-readable:
[`config/taxonomy.yaml`](../config/taxonomy.yaml).

**15 product areas across 5 domains** — the 8-category hypothesis proved too coarse in
three places (`Refunds & Cancellations` matched 33.9% of the corpus; `Payments &
Pricing` merged two areas with opposite platform signatures) and missing four areas,
most consequentially `wallet_and_credits`.

**On Offers/Discounts specifically** (your question): **keep it separate.** Evidence —
it is 12.6% of the window corpus, larger than `payments` or `availability_and_range`;
its platform spread is **3.1×** (Zepto 16.0%, Blinkit 15.1%, JioMart 5.2%), among the
highest; and its `pos:neg` ratio of **1.37×** differs sharply from
`pricing_and_charges`. Most importantly the *failure mode* is different: pricing
complaints are "this costs too much", offer complaints are "you promised and did not
deliver" — a **trust** failure, not a value failure. Merging them would hide the
promised-free-gift cases entirely.

### Field design

| Field | Type | Source | Why |
|---|---|---|---|
| `product_areas[]` | **multi-label** | LLM | 2.19 areas/review; single-label discards most content |
| `issue_types[]` | **multi-label**, scoped to area | LLM | Requires the boundary rules in TAXONOMY.md |
| `strength_types[]` | **multi-label**, scoped to area | LLM | Areas are polarity-neutral |
| `pain_point` | free text | LLM | Not enumerated — a fixed list cannot surface the unanticipated |
| `sentiment` | single-select | LLM (**from text**) | Sarcastic 5★ reviews exist |
| `severity` | single-select | LLM | No deterministic proxy; biggest effect on prioritisation |
| `customer_intent` | single-select | LLM | Feature requests / churn warnings are invisible to keywords |
| `support_escalation` | boolean | LLM | Support is downstream; needs a flag, not just an area |
| `evidence_span[]` | text spans | LLM | **Required** — enables grounding measurement in Phase 9 |
| `confidence` | float | LLM | Gates what the UI shows a PM |
| `rating`, `rating_bucket` | int / enum | **deterministic** | Independent signal to score the model against |
| `platform`, `review_date`, `year_month` | — | **deterministic** | Facts, never inferred |
| `is_truncated`, `near_dup_group_id` | flags | **deterministic** | Quality gates on the labels |
| Area/theme **frequencies** | counts | **deterministic** | Computed by aggregation, never asked of an LLM |

**The dividing line:** an LLM interprets language; it must never produce a number that
can be counted. Frequencies, shares, trends and RICE arithmetic are all deterministic
aggregations over LLM-produced *labels*. Asking a model to estimate "how many reviews
mention delivery" invites confident hallucination where a `groupby` is exact.

### Sentiment, severity, and intent scales

Defined in `config/taxonomy.yaml` with per-value definitions:

- **sentiment:** `positive` · `negative` · `mixed` · `neutral` — `mixed` is not
  `neutral`; it names a working product with one broken surface, a distinct PM signal.
- **severity:** `critical` (money lost with no path to resolution, health risk,
  account locked) · `high` (failed and unresolved, or repeated) · `medium` (single,
  resolvable) · `low` (annoyance).
- **customer_intent:** `complaint` · `praise` · `feature_request` · `churn_warning` ·
  `comparison` · `question`.

### Implications for the AI pipeline

1. **Enrich the full 4,620** — the corpus is small and cheap enough that sampling buys
   nothing and costs comparability.
2. **Require evidence spans on every label.** Without them Phase 9 cannot measure
   grounding, and the UI cannot show a PM why a label was assigned.
3. **Carry `is_truncated` into enrichment** and down-weight severity on those rows.
4. **Gold set must over-sample the hard cases** — multi-area reviews, sarcastic
   ratings, and the eight borderline cases in TAXONOMY.md. A gold set drawn uniformly
   at random would be ~29% single-area and would overstate accuracy.
5. **Report every trend and comparison in the window**, as shares.

---

## 8. Validation

All reconciliation checks against Phase 1 pass, and the pipeline **exits non-zero if
any fail** rather than producing a report from a stale dataset.

| Check | Result |
|---|---|
| Row count matches Phase 1 | PASS (4,620 / 4,620) |
| Platform counts reconcile | PASS |
| Rating counts reconcile | PASS |
| No nulls introduced | PASS (0 null cells) |
| Review IDs unique | PASS |
| Dates valid and unchanged | PASS |
| Ratings within 1–5 | PASS |
| Truncation count reconciles | PASS |
| Raw dataset unmodified | PASS (SHA-256 verified) |

**173 tests pass**, including 33 new EDA tests covering reconciliation, comparison-window
construction, and the analytical invariants the findings above depend on.
