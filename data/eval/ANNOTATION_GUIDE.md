# Annotation guide — gold reference set

Schema `v1` · 134 reviews · 35 disagreement · 99 random

Fill in `gold_template.csv` and save it as `gold_labels.csv` in the same
folder. Do not reorder or delete rows; scoring joins on `review_id`.

## Why this exists

Every number this pipeline reports about label quality is currently a
measure of agreement, not correctness. Two models agreeing proves only that
they share a bias. These labels are the reference that turns agreement into
accuracy, so they need to be *your* reading of the review, not a check of
somebody else's — which is why the model's answer is deliberately not shown.

## Filling in `gold_labels`

One entry per thing the review is about, separated by `;`.
Each entry is `area/issue_type`, or
`area/+strength_type` when the review
is *praising* that area.

A review can criticise one area and praise another; record both. A review
that says nothing about any area gets an empty cell — that is a real answer,
not a skipped row.

Example:

```
delivery_reliability/late_delivery; order_fulfilment/missing_items; delivery_reliability/+fast_delivery
```

## Vocabulary

| Area | Issue types | Strength types |
|---|---|---|
| `delivery_reliability` | `late_delivery`, `never_delivered`, `eta_inaccurate`, `slot_missed` | `fast_delivery`, `reliable_timing` |
| `order_fulfilment` | `missing_items`, `wrong_item`, `short_quantity`, `marked_delivered_not_received` | `accurate_orders` |
| `order_lifecycle` | `unwanted_cancellation`, `cannot_cancel`, `cannot_modify`, `order_stuck` | `easy_cancellation` |
| `serviceability` | `location_not_serviceable`, `address_not_accepted`, `store_closed` | `wide_coverage` |
| `availability_and_range` | `out_of_stock`, `stock_status_wrong`, `limited_range` | `wide_range`, `good_availability` |
| `pricing_and_charges` | `high_prices`, `delivery_charge`, `handling_or_platform_fee`, `surge_pricing`, `hidden_charges`, `membership_value` | `good_value` |
| `offers_and_promotions` | `offer_not_honoured`, `coupon_failed`, `promised_gift_missing`, `misleading_promotion`, `free_delivery_threshold` | `good_offers` |
| `payments` | `payment_failed`, `money_deducted_no_order`, `cod_unavailable`, `limited_payment_options` | `smooth_payment` |
| `refunds` | `refund_not_received`, `refund_delayed`, `partial_refund`, `refund_refused` | `quick_refund` |
| `wallet_and_credits` | `balance_unusable`, `balance_disappeared`, `credit_not_applied`, `refund_locked_in_wallet` | `wallet_convenient` |
| `product_quality` | `expired_product`, `stale_or_rotten`, `damaged_product`, `counterfeit_or_used`, `poor_quality_general`, `packaging_failure` | `fresh_and_good_quality` |
| `returns_and_replacement` | `no_return_option`, `return_request_rejected`, `replacement_not_delivered`, `return_policy_unclear` | `easy_returns` |
| `customer_support` | `no_response`, `no_contact_channel`, `unhelpful_agent`, `chatbot_loop`, `issue_not_listed`, `unresolved_escalation` | `helpful_support` |
| `delivery_partner_conduct` | `rude_behaviour`, `did_not_deliver_to_door`, `tip_pressure`, `cash_collection_dispute`, `false_delivery_marking` | `courteous_partner` |
| `app_experience` | `crash_or_freeze`, `slow_performance`, `cart_or_checkout_bug`, `login_or_otp_failure`, `navigation_confusing`, `missing_feature` | `easy_to_use` |
| `general_no_specific_area` | *(free text; for reviews that fit nothing above)* | — |

## The other columns

| Column | Allowed values |
|---|---|
| `gold_sentiment` | `positive`, `negative`, `mixed`, `neutral` |
| `gold_severity` | `critical`, `high`, `medium`, `low` |
| `gold_customer_intent` | `complaint`, `praise`, `feature_request`, `churn_warning`, `comparison`, `question` |
| `gold_support_escalation` | `true` / `false` |
| `annotator` | your initials — needed to measure inter-annotator agreement |
| `notes` | anything the vocabulary could not express |

## When you are unsure

Write the note. A review you found genuinely ambiguous is a finding about
the taxonomy, and a label guessed to fill the cell is worse than a blank one:
it enters the reference silently and every future accuracy figure inherits it.

Leave a row blank rather than guessing. Blank rows are counted and reported,
not silently dropped.

## Second annotator

If a second person labels the same reviews, save their file as
`gold_labels_<initials>.csv`. Two independent passes give a Cohen's kappa
between annotators, which is the only thing that says whether the task is
well-defined enough for the model's score to mean anything. Without it, an
accuracy of 85% cannot be told apart from a task where humans agree 85% of
the time.
