# Competitive Metrics — Quick-Commerce Voice of Customer

**2026-09-02** · 3,993 reviews in the comparable window · 3 platforms · α = 0.05

Phase 4 ranked what hurts. This asks whether it hurts *more here than elsewhere*, and whether it is getting worse.

---

## How to read this

**Every figure is a rate, never a count.** December holds three times October's reviews — that is scraping intensity, not customer behaviour. A count-based table would rank December worst on every measure for every platform, automatically.

**Every difference is tested.** With ~900–1,700 reviews per platform, a few points of gap is well inside noise. Rates carry Wilson 95% intervals; comparisons carry p-values corrected for multiplicity with Benjamini–Hochberg. The table below runs dozens of tests, and at α=0.05 a couple would clear the bar on noise alone — so **read the `significant` column, not the raw p-value**.

Reviews per platform in the window:

| Platform | Reviews |
|---|---:|
| `blinkit` | 1,356 |
| `jiomart` | 918 |
| `zepto` | 1,719 |

---

## Platform rates

Wilson 95% intervals. Overlapping intervals mean the difference is not established, whatever the point estimates suggest.

**Negative sentiment** (higher is worse)

| Platform | Rate | 95% CI |
|---|---:|---|
| `zepto` | 85.0% | 83.2–86.6% |
| `jiomart` | 84.6% | 82.2–86.8% |
| `blinkit` | 59.9% | 57.2–62.5% |

**High or critical severity** (higher is worse)

| Platform | Rate | 95% CI |
|---|---:|---|
| `jiomart` | 63.3% | 60.1–66.3% |
| `zepto` | 56.7% | 54.4–59.0% |
| `blinkit` | 41.3% | 38.7–43.9% |

**Drove a support contact** (higher is worse)

| Platform | Rate | 95% CI |
|---|---:|---|
| `zepto` | 41.4% | 39.1–43.7% |
| `jiomart` | 31.3% | 28.3–34.3% |
| `blinkit` | 28.8% | 26.4–31.2% |

**Stated intent to leave** (higher is worse)

| Platform | Rate | 95% CI |
|---|---:|---|
| `zepto` | 0.7% | 0.4–1.2% |
| `blinkit` | 0.4% | 0.2–1.0% |
| `jiomart` | 0.2% | 0.1–0.8% |

**Praise** (higher is better)

| Platform | Rate | 95% CI |
|---|---:|---|
| `zepto` | 7.4% | 6.3–8.8% |
| `jiomart` | 10.0% | 8.2–12.1% |
| `blinkit` | 25.5% | 23.3–27.9% |

### Differences that survive correction

10 of 15 comparisons survive.

| Metric | A | B | Rate A | Rate B | Difference | p |
|---|---|---|---:|---:|---:|---:|
| Negative sentiment | `blinkit` | `jiomart` | 59.9% | 84.6% | -24.8pp | 0.0000 |
| Negative sentiment | `blinkit` | `zepto` | 59.9% | 85.0% | -25.1pp | 0.0000 |
| High or critical severity | `blinkit` | `jiomart` | 41.3% | 63.3% | -22.0pp | 0.0000 |
| High or critical severity | `blinkit` | `zepto` | 41.3% | 56.7% | -15.4pp | 0.0000 |
| Praise | `blinkit` | `jiomart` | 25.5% | 10.0% | +15.5pp | 0.0000 |
| Praise | `blinkit` | `zepto` | 25.5% | 7.4% | +18.1pp | 0.0000 |
| Drove a support contact | `blinkit` | `zepto` | 28.8% | 41.4% | -12.6pp | 0.0000 |
| Drove a support contact | `jiomart` | `zepto` | 31.3% | 41.4% | -10.1pp | 0.0000 |
| High or critical severity | `jiomart` | `zepto` | 63.3% | 56.7% | +6.6pp | 0.0011 |
| Praise | `jiomart` | `zepto` | 10.0% | 7.4% | +2.6pp | 0.0227 |

The remaining 5 comparisons are **not established**. Their point estimates differ; the evidence does not support saying so.

---

## Where each platform over-indexes

Share of a platform's own reviews raising each area, against the same rate across the other platforms pooled. `lift` above 1.0 means the area is raised more often here than corpus-wide.

| Area | Platform | Rate | 95% CI | Corpus | Lift | p |
|---|---|---:|---|---:|---:|---:|
| `wallet_and_credits` | `zepto` | 26.6% | 24.6–28.7% | 12.3% | 2.16× | 0.0000 |
| `order_lifecycle` | `jiomart` | 22.7% | 20.1–25.5% | 12.4% | 1.82× | 0.0000 |
| `delivery_reliability` | `jiomart` | 29.6% | 26.8–32.7% | 16.4% | 1.81× | 0.0000 |
| `availability_and_range` | `jiomart` | 10.7% | 8.8–12.8% | 6.4% | 1.66× | 0.0000 |
| `offers_and_promotions` | `zepto` | 8.0% | 6.8–9.4% | 5.6% | 1.42× | 0.0000 |
| `product_quality` | `blinkit` | 16.7% | 14.8–18.7% | 11.8% | 1.42× | 0.0000 |
| `payments` | `zepto` | 6.5% | 5.4–7.7% | 4.7% | 1.38× | 0.0000 |
| `order_fulfilment` | `jiomart` | 19.5% | 17.1–22.2% | 14.2% | 1.38× | 0.0000 |
| `refunds` | `jiomart` | 16.0% | 13.8–18.5% | 12.1% | 1.32× | 0.0000 |
| `pricing_and_charges` | `blinkit` | 12.8% | 11.1–14.6% | 9.8% | 1.30× | 0.0000 |
| `returns_and_replacement` | `blinkit` | 9.7% | 8.2–11.3% | 7.8% | 1.24× | 0.0015 |
| `customer_support` | `zepto` | 44.7% | 42.3–47.0% | 37.4% | 1.19× | 0.0000 |
| `pricing_and_charges` | `zepto` | 11.6% | 10.2–13.2% | 9.8% | 1.19× | 0.0008 |
| `app_experience` | `jiomart` | 23.6% | 21.0–26.5% | 20.0% | 1.18× | 0.0015 |
| `serviceability` | `zepto` | 5.4% | 4.4–6.6% | 4.6% | 1.18× | 0.0298 |
| `app_experience` | `zepto` | 23.4% | 21.4–25.4% | 20.0% | 1.17× | 0.0000 |
| `product_quality` | `zepto` | 10.1% | 8.7–11.6% | 11.8% | 0.86× | 0.0036 |
| `order_lifecycle` | `zepto` | 10.4% | 9.0–11.9% | 12.4% | 0.83× | 0.0005 |
| `refunds` | `blinkit` | 9.8% | 8.3–11.5% | 12.1% | 0.81× | 0.0015 |
| `returns_and_replacement` | `zepto` | 6.2% | 5.2–7.5% | 7.8% | 0.80× | 0.0013 |
| `customer_support` | `blinkit` | 29.5% | 27.1–32.0% | 37.4% | 0.79× | 0.0000 |
| `order_fulfilment` | `zepto` | 10.9% | 9.5–12.4% | 14.2% | 0.77× | 0.0000 |
| `order_lifecycle` | `blinkit` | 8.2% | 6.8–9.8% | 12.4% | 0.66× | 0.0000 |
| `app_experience` | `blinkit` | 13.1% | 11.4–15.0% | 20.0% | 0.66× | 0.0000 |
| `product_quality` | `jiomart` | 7.7% | 6.2–9.6% | 11.8% | 0.66× | 0.0000 |
| `availability_and_range` | `zepto` | 4.0% | 3.2–5.0% | 6.4% | 0.62× | 0.0000 |
| `delivery_reliability` | `blinkit` | 9.1% | 7.7–10.7% | 16.4% | 0.55× | 0.0000 |
| `payments` | `jiomart` | 1.2% | 0.7–2.1% | 4.7% | 0.26× | 0.0000 |
| `offers_and_promotions` | `jiomart` | 1.4% | 0.8–2.4% | 5.6% | 0.25× | 0.0000 |
| `pricing_and_charges` | `jiomart` | 2.1% | 1.3–3.2% | 9.8% | 0.21× | 0.0000 |
| `wallet_and_credits` | `jiomart` | 2.2% | 1.4–3.3% | 12.3% | 0.18× | 0.0000 |
| `wallet_and_credits` | `blinkit` | 1.1% | 0.7–1.8% | 12.3% | 0.09× | 0.0000 |

---

## Trend

**Not computed — 3 comparable month(s) available, 6 needed. Outside the comparable window review volume tracks collection rather than customer behaviour, so extending the series would measure the scraper.**

This module requires **6 monthly observations** before it will describe a direction. Three points can be joined by a line, and that is exactly the danger: with one collection artefact anywhere in the series, the line *is* the artefact.

Phase 4 learned this the expensive way. An unguarded recent-vs-prior ratio reported growth of up to **197×**, which was entirely an artefact of when reviews were scraped — the corpus spans 50 months, but 42 of the earliest 47 hold fewer than ten reviews each.

**What would fix it:** roughly six months of collection at the current rate inside a window where all three platforms are present. That is a data-collection task, not an analysis one, and no amount of modelling substitutes for it.

### Monthly rates (description, not direction)

Safe to show at any series length because each value is a share of its own platform-month. Reading a direction into three points is what the guard above prevents.

**Negative sentiment**

| Platform | 2024-10 | 2024-11 | 2024-12 |
|---|---|---|---|
| `blinkit` | 65.0% | 61.4% | 56.5% |
| `jiomart` | 88.1% | 83.2% | 83.7% |
| `zepto` | 82.8% | 84.5% | 85.6% |

---

## Caveats

**Reviews are not users.** App-store reviews are written by people motivated enough to write one, which skews negative everywhere. These rates compare platforms against each other, not against reality.

**Labels are model output.** Grounding was verified at 98.4%, but no hand-labelled gold set exists until Phase 9. A systematic model bias would move every platform's rate together — which is partly why *differences* are the unit here rather than absolute levels.

**Review volume differs threefold across platforms.** Wilson intervals account for that; the narrower interval simply belongs to the platform with more reviews.

**Significance is not importance.** A difference can be real and too small to act on. The `difference` column is in percentage points for exactly that reason — judge the size, not just the asterisk.

