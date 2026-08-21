# Product Taxonomy

**Version 1.0.0** · derived from 4,620 reviews in `data/interim/reviews_clean.parquet`

Machine-readable source of truth: [`config/taxonomy.yaml`](../config/taxonomy.yaml).
Measured evidence: [`reports/taxonomy_discovery.md`](../reports/taxonomy_discovery.md),
regenerate with `python scripts/02_discover_taxonomy.py`.

---

## 1. Why this structure

The Phase 0 hypothesis was eight categories. Testing it against the corpus showed it
was **not wrong so much as too coarse in three places and missing four areas.** The
final taxonomy has **15 product areas across 5 domains**.

The central design rule is that four concepts are kept apart:

| Concept | Answers | Example |
|---|---|---|
| **product_area** | *Where* in the product does this land? | `wallet_and_credits` |
| **issue_type** | *What went wrong* on that surface? | `balance_unusable` |
| **strength_type** | *What went right* on that surface? | `wallet_convenient` |
| **pain_point** | The specific instance, in the customer's terms | "₹100 Zepto Cash visible but no option to spend it" |
| **attributes** | *How bad*, *how they feel*, *what they want* | severity `high`, sentiment `negative`, intent `complaint` |

A product area is **polarity-neutral**. It is a surface, not a verdict. The same area
carries praise and complaint, which is why sentiment and severity are separate
attributes rather than being baked into category names. A taxonomy with a category
called "Delivery Problems" cannot represent "delivery is the best thing about this
app" — and 31% of positive reviews say exactly that.

### The hierarchy

```
domain                     (5 — dashboard grouping only, never classified to)
└── product_area           (15 — the analysis and comparison unit, MULTI-LABEL)
    ├── issue_type         (negative manifestation, scoped to the area)
    ├── strength_type      (positive manifestation, scoped to the area)
    └── pain_point         (free text, generated per review — not enumerated)

review-level attributes    sentiment · severity · customer_intent
                           support_escalation · evidence_span · confidence
```

`pain_point` is deliberately **not** an enumerated level. An enumerated pain point is
just a narrower issue type, and a fixed list would stop the system from surfacing
anything we did not anticipate — which is the entire point of a discovery tool.

### Reviews are multi-label, and this is not a detail

Mean **2.19 areas per review**. Only 1,349 reviews (29%) touch exactly one area.

| Areas matched | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| Reviews | 411 | 1,349 | 1,362 | 800 | 363 | 335 |

Forcing a single label would discard the majority of what each review says. Worse, it
would systematically under-count whichever area tends to appear second — and because
support is usually the *second* thing mentioned, single-labelling would have hidden
the largest pain point in the corpus.

---

## 2. What changed from the 8-category hypothesis

| # | Hypothesis category | Verdict | Reason |
|---|---|---|---|
| 1 | Delivery Experience | **Renamed** → `delivery_reliability` | "Experience" invited conduct and fulfilment complaints. Renamed to what it actually measures: did it arrive, on time. |
| 2 | Order Fulfillment | **Split** → `order_fulfilment` + `availability_and_range` | Being sent the wrong item and never being able to buy the item are different failures with different owners (ops vs. category/supply). |
| 3 | Product Quality | **Kept**, packaging folded in | `packaging` alone is 2.5% — too small to be an area, and it fails for the same reason quality fails. |
| 4 | Refunds & Cancellations | **Split three ways** → `refunds` + `order_lifecycle` + `returns_and_replacement` | The single biggest problem with the hypothesis. See below. |
| 5 | Customer Support | **Kept**, plus a new attribute | Retained as an area, but `support_escalation` added as a review-level flag. See §4. |
| 6 | Payments & Pricing | **Split** → `payments` + `pricing_and_charges` | A failed transaction and an unfair fee share no root cause, no owner, and no fix. |
| 7 | App/Platform Performance | **Renamed** → `app_experience`, login folded in | `account_access` is 2.1% — an issue type, not an area. |
| 8 | Delivery Personnel Conduct | **Kept** → `delivery_partner_conduct` | Clean, distinct, well-populated at 12.7%. |
| — | *(missing)* | **Added** `wallet_and_credits` | See below — the most consequential gap. |
| — | *(missing)* | **Added** `offers_and_promotions` | 12.6% of the corpus; a distinct trust failure from pricing. |
| — | *(missing)* | **Added** `serviceability` | 4.7%; a coverage/network decision, not an inventory one. |

### 2.1 Why "Refunds & Cancellations" had to be split

As one bucket it matched **33.9% of all reviews and 40.9% of negatives** — a category
so large it is useless for prioritisation. Splitting it reveals three separately
actionable problems:

| Split-out area | % corpus | Platform spread | What a PM would do about it |
|---|---|---|---|
| `refunds` | 16.5% | 1.6× | Automate refund SLA, fix "initiated but never credited" |
| `order_lifecycle` | 15.9% | **3.1×** | Stop unilateral cancellations; ship a cancel/modify flow |
| `returns_and_replacement` | 12.8% | 1.8× | Build a returns path that exists at all |

The platform spread column is the giveaway. `order_lifecycle` runs **26.8% on JioMart
vs 8.7% on Blinkit** — a 3.1× gap that the merged category averaged into invisibility.

### 2.2 Why "Payments & Pricing" had to be split

The two halves have **opposite platform signatures**, so merging them cancels the
signal out:

| Area | Blinkit | Zepto | JioMart |
|---|---|---|---|
| `payments` | 9.1% | **15.6%** | 8.3% |
| `pricing_and_charges` | **16.7%** | 14.4% | 7.2% |

Merged, all three platforms look similar (~21%). Split, Zepto has a transaction
problem and Blinkit has a value-perception problem. Those go to different teams.

### 2.3 The most consequential gap: `wallet_and_credits`

This area does not appear anywhere in the 8-category hypothesis, and it is the
**single strongest platform discriminator in the entire corpus**:

| Area | Blinkit | Zepto | JioMart | Spread |
|---|---|---|---|---|
| `wallet_and_credits` | 1.2% | **27.9%** | 2.9% | **22.7×** |

Under the hypothesis it would have been absorbed into "Payments & Pricing" and diluted
from a 27.9% Zepto-specific crisis into a rounding error. Customers describe money
they already own sitting visible in a wallet with no way to spend it, and no phone
line to call. That is a **trust** failure, not a payments failure — the money never
had a problem moving *in*.

---

## 3. Discovery estimates

Measured with the keyword probes in `config/taxonomy.yaml`. **These are lower bounds,
not classifications** — probes are high-precision and mediocre-recall by design.

| Area | Domain | % corpus | % of neg | % of pos | pos:neg | Platform spread |
|---|---|---|---|---|---|---|
| `customer_support` | Service & Platform | 29.1% | 34.8% | 8.8% | 0.25× | 1.7× |
| `delivery_reliability` | Fulfilment | 22.9% | 21.1% | 31.0% | **1.47×** | 1.3× |
| `product_quality` | Goods | 18.9% | 17.6% | 24.9% | **1.42×** | 1.7× |
| `app_experience` | Service & Platform | 17.8% | 17.4% | 17.0% | 0.98× | 1.3× |
| `refunds` | Money | 16.5% | 20.1% | 3.3% | 0.16× | 1.6× |
| `order_lifecycle` | Fulfilment | 15.9% | 19.7% | 1.8% | **0.09×** | **3.1×** |
| `returns_and_replacement` | Goods | 12.8% | 15.0% | 4.5% | 0.30× | 1.8× |
| `pricing_and_charges` | Catalogue & Value | 12.8% | 11.3% | 18.0% | **1.59×** | 2.3× |
| `delivery_partner_conduct` | Service & Platform | 12.7% | 13.2% | 10.6% | 0.80× | 1.2× |
| `offers_and_promotions` | Catalogue & Value | 12.6% | 11.7% | 16.0% | **1.37×** | 2.5× |
| `wallet_and_credits` | Money | 11.8% | 14.0% | 3.7% | 0.26× | **22.7×** |
| `availability_and_range` | Catalogue & Value | 11.3% | 10.7% | 13.1% | 1.22× | 1.3× |
| `payments` | Money | 11.3% | 12.6% | 5.5% | 0.44× | 1.9× |
| `order_fulfilment` | Fulfilment | 7.4% | 8.7% | 2.2% | 0.26× | 1.7× |
| `serviceability` | Fulfilment | 4.7% | 5.2% | 2.4% | 0.45× | 1.3× |
| *unmatched* | — | 8.9% | — | — | — | — |

### Reading `pos:neg`

The ratio is an area's share of positive reviews divided by its share of negative
reviews. It splits the taxonomy cleanly into three kinds of surface:

- **Strengths (>1.2×)** — `pricing_and_charges` 1.59×, `delivery_reliability` 1.47×,
  `product_quality` 1.42×, `offers_and_promotions` 1.37×, `availability_and_range`
  1.22×. Customers actively praise these.
- **Neutral (~1.0×)** — `app_experience` 0.98×. Praised and criticised equally.
- **Failure-only (<0.3×)** — `order_lifecycle` 0.09×, `refunds` 0.16×,
  `customer_support` 0.25×, `wallet_and_credits` 0.26×, `order_fulfilment` 0.26×.

Nobody writes a review to praise a refund. These surfaces are invisible when they work
and infuriating when they do not, which makes them **pure defect-reduction targets** —
a materially different PM strategy from the strength areas, where the play is
amplification and differentiation.

---

## 4. Customer support is downstream — and why that needs an attribute

Base rate: 29.1% of reviews mention support. Lift = P(support mentioned | area) ÷ base.

| Area | P(support \| area) | Lift |
|---|---|---|
| `order_fulfilment` | 48.0% | **1.65×** |
| `refunds` | 43.1% | 1.48× |
| `returns_and_replacement` | 42.5% | 1.46× |
| `delivery_partner_conduct` | 40.4% | 1.39× |
| … | | |
| `pricing_and_charges` | 19.2% | 0.66× |
| `serviceability` | 14.6% | 0.50× |

Operational failures drive support contact far above base rate; pricing and coverage
complaints drive it far below. **Support is a failure amplifier, not an independent
driver.** People do not contact support because support is bad — they contact support
because something else broke, and *then* discover support is bad.

This creates a modelling problem. If support is only ever a product area, then a review
saying *"my refund never came and support ignored three emails"* either gets labelled
`customer_support` (hiding the refund failure) or `refunds` (hiding the support
failure). Both lose information a PM needs.

**Resolution:** keep `customer_support` as an area for reviews where support *is* the
problem, and add `support_escalation` as a review-level boolean that can be true
alongside any area. This makes contact-deflection analysis possible — *which upstream
failures generate the most support contact* is exactly the question that turns a VoC
dashboard into a roadmap.

---

## 5. The areas

Full definitions, inclusion and exclusion criteria, and issue/strength types live in
[`config/taxonomy.yaml`](../config/taxonomy.yaml). Below is each area with a real
review from the corpus. Examples are chosen from reviews matching **exactly one** area,
so each illustrates its area cleanly.

### Fulfilment

**`delivery_reliability`** — did it arrive, and on time.
> *Blinkit, 1★* — "Don't use print service. I need urgent document contains 6 pages.. they delivery only 3 pages.."

**`order_fulfilment`** — did the right things arrive.
> *Blinkit, 2★* — "ordered Faces canada matte foundation, instead received Faces Canada Concealer which is no use of mine. I reported this issue twice…"

**`order_lifecycle`** — can the customer control the order after placing it.
> *Blinkit, 1★* — "Accidentally I clicked on check out and the order got placed and with in fraction of seconds I tried to cancel it… the app said ur order is packed."

**`serviceability`** — will the platform serve this location at all.
> *Blinkit, 1★* — "It is not accepting even the locations near my house. I gave the location of Indore airport, but that too is not being accepted."

### Catalogue & Value

**`availability_and_range`** — what can be bought, and is it in stock.
> *Blinkit, 5★* — "As from tier2 city, I used to buy some great products & groceries from Amazon which takes days to deliver with higher cost. But Blinkit solved my problem…"

**`pricing_and_charges`** — what it costs and how that feels.
> *Blinkit, 1★* — "The platform fees have increased while the order takes half to 1 hr for delivery. This happened not once but multiple times."

**`offers_and_promotions`** — are advertised deals honoured.
> *Blinkit, 1★* — "They show free gift for suppose 1000rs, but free gift do not add automatically we have to add it, add button does not show properly so that people forget to add…"

### Money

**`payments`** — did the transaction execute.
> *Blinkit, 1★* — "All Payment options are showing Disabled due to technical problem. Tried reinstalling the app, cleared the cache but no use."

**`refunds`** — did owed money come back.
> *Blinkit, 2★* — "I paid online but the order was not placed at all. Thankfully they initiated refund but within one week. It was super annoying."

**`wallet_and_credits`** — can the customer use money the platform holds.
> *JioMart, 1★* — "you can't use tha money that you have in your jio mart wallets….. my request please solve this issue"

### Goods

**`product_quality`** — condition and authenticity of what arrived.
> *Blinkit, 1★* — "they can sell duplicate product also and If you complain then also they just ignore you… I had buyed a Sandisk 64…"

**`returns_and_replacement`** — can it be sent back or swapped.
> *Blinkit, 2★* — "Recently I ordered a projector and blink it sold me a used piece. It never gave me a replacement for the device even after raising complaint multiple times."

### Service & Platform

**`customer_support`** — is help reachable and useful.
> *Blinkit, 1★* — "That product is missed and so I contacted to customer care. but no response."

**`delivery_partner_conduct`** — how the courier behaved.
> *Blinkit, 1★* — "My issue is not with the app, but with the delivery partners. I had ordered groceries only to have the delivery partner refuse to walk up the stairs…"

**`app_experience`** — the software itself.
> *Blinkit, 1★* — "It takes over 5 minutes just to load, and even then, it often crashes or freezes."

---

## 6. Borderline cases

These are the boundaries where classification will actually go wrong. Each rule below
is fed verbatim into the Phase 3 prompt and each becomes a gold-set test case in Phase 9.

**1. Refund vs. return.** *Refund = money moving. Return = goods moving.* "I want my
money back" is `refunds`. "There's no way to send it back" is `returns_and_replacement`.
Customers routinely say "return or refund" in one breath — measured overlap is Jaccard
0.17, with 201 reviews matching both. **Both labels are correct in that case.** This is
a genuine multi-label situation, not an error to resolve.

**2. Refund vs. wallet.** A refund credited to the original payment method is `refunds`.
A refund forced into platform wallet credit the customer cannot spend is
`wallet_and_credits` — the money came back in name only. This distinction is what makes
the Zepto pattern legible.

**3. "Option not available".** This exact phrase appears in both senses. Referring to an
*item*, it is `availability_and_range`. Referring to an *app feature* — most often
"the option to use my wallet balance is not available" — it belongs to the area that
feature serves. This one caused measurable probe contamination during discovery: the
`availability_and_range` probe pulled in Zepto wallet-template reviews, which is why
n-gram output for that area showed "cash unavailable" and "emailed zepto".

**4. Delivery late vs. delivery partner.** The *event* being slow is
`delivery_reliability`. The *person* being rude, refusing stairs, or demanding a tip is
`delivery_partner_conduct`. "Delivery partner was 40 minutes late" is reliability; the
partner is incidental.

**5. Out of stock vs. not serviceable.** Item unavailable → `availability_and_range`.
Whole location unserved → `serviceability`. The fix differs completely: inventory
versus network expansion.

**6. Support mentioned vs. support at fault.** If support is named only as context for
another failure, label the underlying area and set `support_escalation = true`. Only use
`customer_support` as an area when the support interaction is itself the complaint.

**7. Sarcasm and rating–text mismatch.** The corpus contains reviews whose star rating
contradicts their text. A real example, rated **5★**:
> *JioMart, 5★* — "Wow, JioMart, hats off for redefining customer experience! Ordered groceries, waited eagerly, and voilà—a cancellation email with 'unforeseen circumstances' as the cherry on top."

This is `order_lifecycle` / `unwanted_cancellation` with **negative** sentiment despite
5 stars. Sentiment must be derived from the *text*, never inherited from the rating.
Keeping `rating_bucket` (rating-derived) separate from `sentiment` (text-derived) is
what allows Phase 9 to measure the model against an independent signal — and
disagreements between the two are a useful sarcasm detector, not noise.

**8. Feature requests are not complaints.** "The app should include a statement feature
that allows users to track orders over time" is `app_experience` / `missing_feature`
with intent `feature_request`, not `complaint`. Found in the unmatched residual;
easily mislabelled, and separately valuable to a PM.

---

## 7. Handling the dataset's limitations

Every limitation from Phase 1 constrains how this taxonomy may be used. These are
encoded in `dataset_caveats` in the YAML so the enrichment prompt and the UI both
inherit them.

| Limitation | Effect on the taxonomy | Mitigation |
|---|---|---|
| **77.5% negative sample** | Area shares describe complaints, not customers | Never render an area share as "% of customers affected". Report share-of-negative and share-of-positive separately — which is why `pos:neg` is a first-class metric. |
| **87% of reviews in Oct–Dec 2024** | Area trends only comparable in that window | `in_comparable_window` gates all cross-platform trend views. |
| **Uneven platform coverage** | Blinkit has 12 pre-October reviews, Zepto 1 | Platform comparison restricted to the comparable window. |
| **8% truncated at 500 chars** | Truncated reviews average 1.36★ vs 1.83★ and are 90.8% negative — they are cut *before* any resolution is described | `severity` and `support_escalation` flagged lower-confidence on truncated rows. |
| **Near-duplicate templates** | 18 groups, 49 reviews | Measured inflation is **< 0.4 pp on every area** — the taxonomy is not built on templated noise. Pain-point counts still reported raw and representative-only. |
| **Non-representative sample** | Cross-platform gaps are within-dataset only | Area comparisons phrased as "within this review dataset". |
| **Probes have limited recall** | All discovery estimates are floors | Labelled as estimates everywhere; Phase 3 replaces them. |

### The unmatched residual is not a gap

411 reviews (8.9%) match no probe. Mean rating **2.73** vs 1.83 for the corpus, and
median length 200 chars vs 290 — the residual is **shorter and more positive** than the
corpus. Reading it showed two populations:

1. **Generic sentiment with no product surface** — "worst app ever", "very good service".
   Legitimately unclassifiable; handled by `general_no_specific_area`.
2. **Lexical recall misses** — "received one item less" is plainly `order_fulfilment`
   but matches no probe phrase. "No one is there to resolve our problem" is plainly
   `customer_support`.

The second population is the majority, and it is the clearest argument for LLM
classification over keyword rules. **A monitoring rule is set in the YAML: if LLM
classification assigns `general_no_specific_area` to more than 10% of reviews, the
taxonomy has a real gap and must be revisited before results are trusted.**

---

## 8. Fitness for both analysis modes

**Single-platform.** All 15 areas are populated on all three platforms (minimum
observed share 1.2%), so no platform view has empty categories.

**Competitive comparison.** Areas were retained or split partly *because* they
discriminate between platforms. Ranked by spread, the taxonomy yields a clean read:

| Platform | Signature areas | Interpretation *(within this dataset)* |
|---|---|---|
| **Zepto** | `wallet_and_credits` 27.9%, `customer_support` 34.5%, `payments` 15.6% | A money-trust problem: stored value customers cannot spend, no reachable help. |
| **JioMart** | `order_lifecycle` 26.8%, `refunds` 21.1%, `app_experience` 20.4% | A fulfilment-reliability problem: orders cancelled, money slow to return. |
| **Blinkit** | `product_quality` 23.1%, `pricing_and_charges` 16.7%, `offers_and_promotions` 15.1% | A value-perception problem, on the strongest positive base of the three. |

Three platforms, three genuinely different failure signatures. The merged 8-category
hypothesis produced three near-identical profiles — which would have made the
competitive mode worthless.

---

## 9. Change process

The taxonomy is versioned (`version` in the YAML). It is expected to change after
Phase 3, when LLM classification exposes areas that keyword probes could not size.

To change it:
1. Edit `config/taxonomy.yaml` — never a Python file. A test enforces this.
2. Bump `version`.
3. Re-run `python scripts/02_discover_taxonomy.py`.
4. Run `pytest tests/test_taxonomy.py`.
5. Update this document, including the rationale for the change.

Any change after gold labels exist (Phase 9) **invalidates those labels for affected
areas** and requires re-labelling. That cost is the reason this phase happened before
enrichment rather than after.
