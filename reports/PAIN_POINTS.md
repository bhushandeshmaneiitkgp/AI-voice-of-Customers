# Pain Points — Quick-Commerce Voice of Customer

**2026-09-02** · 4,568 enriched reviews · 57 scored pain points

Phase 4 output. Enrichment said which categories apply to each review; this ranks them by how much they appear to cost, and clusters the text to surface themes the taxonomy does not name.

---

## How the score is computed

A pain point is a `(product_area, issue_type)` pair. Five signals, weighted:

| Signal | Weight | Meaning |
|---|---|---|
| Volume | 0.35 | distinct reviews raising it (min-maxed) |
| Severity | 0.25 | mean severity, low=1 → critical=4 |
| Escalation | 0.15 | share that drove a support contact |
| Churn | 0.15 | share stating intent to leave |
| Negativity | 0.10 | share appearing in negative reviews |

Pain points under **15 reviews** are excluded: below that a pattern is an anecdote.

The weights are a **product judgement, not a discovered constant**. They live in `src/voc/painpoints.py` so disagreeing with them is a one-line change and a re-run, not an argument with a black box.

**There is no trend column, and that is a result.** The first attempt produced ratios up to 197× — which was collection, not customers. The corpus spans 50 months, but 42 of the earliest 47 hold fewer than ten reviews each while the final three hold 3,475. Restricted to the comparable window (from 2024-10, where every platform is meaningfully present) only three months remain — shorter than any honest trend needs. `add_trend` therefore refuses. Whether these pain points are growing is a **Phase 5 question that this corpus cannot answer**.

---

## Ranked pain points

| # | Area | Issue | Reviews | Severity | Escalation | Churn | Score |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | `customer_support` | `unhelpful_agent` | 954 | 3.22 | 92% | 1.1% | **0.789** |
| 2 | `customer_support` | `no_response` | 704 | 3.28 | 89% | 0.0% | **0.669** |
| 3 | `refunds` | `refund_not_received` | 348 | 3.53 | 63% | 0.0% | **0.534** |
| 4 | `order_lifecycle` | `unwanted_cancellation` | 438 | 3.22 | 39% | 0.2% | **0.493** |
| 5 | `refunds` | `partial_refund` | 18 | 3.06 | 61% | 5.6% | **0.486** |
| 6 | `delivery_reliability` | `never_delivered` | 240 | 3.47 | 54% | 0.4% | **0.483** |
| 7 | `order_fulfilment` | `missing_items` | 378 | 3.06 | 63% | 0.3% | **0.481** |
| 8 | `delivery_reliability` | `late_delivery` | 464 | 2.99 | 32% | 1.1% | **0.480** |
| 9 | `returns_and_replacement` | `return_request_rejected` | 99 | 3.41 | 82% | 0.0% | **0.453** |
| 10 | `product_quality` | `damaged_product` | 171 | 3.26 | 68% | 0.6% | **0.446** |
| 11 | `payments` | `money_deducted_no_order` | 95 | 3.52 | 68% | 0.0% | **0.443** |
| 12 | `order_fulfilment` | `wrong_item` | 239 | 3.19 | 60% | 0.0% | **0.437** |
| 13 | `refunds` | `refund_delayed` | 177 | 3.21 | 49% | 1.1% | **0.433** |
| 14 | `refunds` | `refund_refused` | 41 | 3.56 | 66% | 0.0% | **0.429** |
| 15 | `order_fulfilment` | `marked_delivered_not_received` | 64 | 3.48 | 67% | 0.0% | **0.427** |
| 16 | `returns_and_replacement` | `no_return_option` | 213 | 3.13 | 56% | 0.5% | **0.423** |
| 17 | `delivery_partner_conduct` | `rude_behaviour` | 230 | 3.13 | 42% | 0.9% | **0.420** |
| 18 | `order_lifecycle` | `cannot_modify` | 22 | 2.82 | 50% | 4.5% | **0.416** |
| 19 | `wallet_and_credits` | `balance_unusable` | 427 | 2.55 | 44% | 0.5% | **0.402** |
| 20 | `product_quality` | `counterfeit_or_used` | 17 | 3.76 | 35% | 0.0% | **0.397** |
| 21 | `product_quality` | `stale_or_rotten` | 51 | 3.18 | 45% | 2.0% | **0.396** |
| 22 | `delivery_partner_conduct` | `false_delivery_marking` | 20 | 3.50 | 45% | 0.0% | **0.382** |
| 23 | `app_experience` | `missing_feature` | 592 | 2.33 | 17% | 0.3% | **0.381** |
| 24 | `customer_support` | `no_contact_channel` | 148 | 3.14 | 49% | 0.0% | **0.376** |
| 25 | `order_lifecycle` | `order_stuck` | 35 | 3.29 | 49% | 0.0% | **0.362** |
| 26 | `product_quality` | `expired_product` | 91 | 3.23 | 46% | 0.0% | **0.362** |
| 27 | `returns_and_replacement` | `replacement_not_delivered` | 22 | 3.05 | 68% | 0.0% | **0.348** |
| 28 | `order_lifecycle` | `cannot_cancel` | 185 | 2.95 | 36% | 0.0% | **0.343** |
| 29 | `product_quality` | `poor_quality_general` | 214 | 2.78 | 39% | 0.0% | **0.325** |
| 30 | `wallet_and_credits` | `balance_disappeared` | 56 | 3.00 | 43% | 0.0% | **0.321** |
| 31 | `wallet_and_credits` | `refund_locked_in_wallet` | 21 | 3.05 | 43% | 0.0% | **0.310** |
| 32 | `refunds` | `refund_locked_in_wallet` | 22 | 3.09 | 36% | 0.0% | **0.307** |
| 33 | `returns_and_replacement` | `return_policy_unclear` | 16 | 2.88 | 62% | 0.0% | **0.305** |
| 34 | `order_fulfilment` | `short_quantity` | 36 | 2.89 | 56% | 0.0% | **0.305** |
| 35 | `offers_and_promotions` | `misleading_promotion` | 85 | 2.65 | 31% | 1.2% | **0.289** |
| 36 | `payments` | `payment_failed` | 38 | 2.89 | 37% | 0.0% | **0.285** |
| 37 | `availability_and_range` | `out_of_stock` | 173 | 2.60 | 17% | 1.2% | **0.284** |
| 38 | `pricing_and_charges` | `high_prices` | 246 | 2.38 | 11% | 1.2% | **0.267** |
| 39 | `delivery_reliability` | `eta_inaccurate` | 104 | 2.69 | 29% | 0.0% | **0.265** |
| 40 | `serviceability` | `location_not_serviceable` | 192 | 2.61 | 12% | 0.0% | **0.262** |
| 41 | `pricing_and_charges` | `surge_pricing` | 37 | 2.51 | 3% | 2.7% | **0.247** |
| 42 | `app_experience` | `crash_or_freeze` | 85 | 2.67 | 20% | 0.0% | **0.243** |
| 43 | `offers_and_promotions` | `offer_not_honoured` | 140 | 2.42 | 29% | 0.0% | **0.239** |
| 44 | `availability_and_range` | `stock_status_wrong` | 29 | 2.68 | 31% | 0.0% | **0.237** |
| 45 | `app_experience` | `login_or_otp_failure` | 53 | 2.50 | 36% | 0.0% | **0.236** |
| 46 | `payments` | `cod_unavailable` | 36 | 2.19 | 22% | 2.8% | **0.216** |
| 47 | `app_experience` | `slow_performance` | 191 | 2.29 | 12% | 0.0% | **0.210** |
| 48 | `pricing_and_charges` | `hidden_charges` | 68 | 2.57 | 9% | 0.0% | **0.206** |
| 49 | `delivery_reliability` | `slot_missed` | 18 | 2.67 | 11% | 0.0% | **0.206** |
| 50 | `wallet_and_credits` | `credit_not_applied` | 18 | 2.39 | 39% | 0.0% | **0.203** |
| 51 | `delivery_partner_conduct` | `did_not_deliver_to_door` | 35 | 2.63 | 17% | 0.0% | **0.202** |
| 52 | `app_experience` | `cart_or_checkout_bug` | 41 | 2.39 | 34% | 0.0% | **0.201** |
| 53 | `app_experience` | `navigation_confusing` | 111 | 2.23 | 10% | 0.0% | **0.173** |
| 54 | `availability_and_range` | `limited_range` | 125 | 2.10 | 9% | 0.0% | **0.131** |
| 55 | `pricing_and_charges` | `delivery_charge` | 76 | 2.13 | 4% | 0.0% | **0.113** |
| 56 | `pricing_and_charges` | `handling_or_platform_fee` | 21 | 2.00 | 10% | 0.0% | **0.092** |
| 57 | `payments` | `limited_payment_options` | 33 | 2.00 | 9% | 0.0% | **0.078** |

### What customers actually said

Verbatim spans, verified against the source text at enrichment time.

**1. `customer_support` / `unhelpful_agent`** — 954 reviews, score 0.789

> Customer support is unhelpful

> What they say if you complaint that the product is good you have paid,,, we are not giving your money back

> The Customer Support experience makes it even worse with only Chatbots and no real solution

**2. `customer_support` / `no_response`** — 704 reviews, score 0.669

> no solution even after multiple follow-up

> No customer support to deal directly .only email..that too 24 hrs waiting time

> No customer support

**3. `refunds` / `refund_not_received`** — 348 reviews, score 0.534

> no refund policy, will eat all your money

> I didn't get my refund back

> still I didn't receive my money

**4. `order_lifecycle` / `unwanted_cancellation`** — 438 reviews, score 0.493

> my order got cancelled heads-up more than once

> my money was deducted but order not confirmed and cancelled

> you all keep giving advertisements about delivering goods 10 minutes but in reality just keep cancelling orders

**5. `refunds` / `partial_refund`** — 18 reviews, score 0.486

> refund very less amount

> the refunded amount is less that of what I paid

> I got a refund of 2 items and the main item was not refunded

---

## Themes discovered in the text

k = **6**, chosen by silhouette score (0.1227) over the range 3–24. Scored rather than picked by eye, so the number is arguable:

| k | silhouette |
|---:|---:|
| 3 | 0.1051 |
| 4 | 0.1211 |
| 5 | 0.1218 |
| 6 | 0.1227  ← chosen |
| 7 | 0.1127 |
| 8 | 0.0906 |
| 9 | 0.0837 |
| 10 | 0.0863 |
| 11 | 0.0787 |
| 12 | 0.0815 |
| 13 | 0.0766 |
| 14 | 0.0831 |

| Cluster | Size | Share | Dominant area | Dominant issue | Severity | Escalation |
|---:|---:|---:|---|---|---:|---:|
| 0 | 1,285 | 28.1% | `app_experience` | `missing_feature` | 2.75 | 30% |
| 4 | 1,134 | 24.8% | `customer_support` | `unhelpful_agent` | 3.01 | 58% |
| 1 | 738 | 16.2% | `customer_support` | `balance_unusable` | 2.74 | 40% |
| 5 | 620 | 13.6% | `delivery_reliability` | `high_prices` | 2.02 | 6% |
| 2 | 437 | 9.6% | `delivery_reliability` | `unhelpful_agent` | 2.67 | 26% |
| 3 | 354 | 7.8% | `customer_support` | `unhelpful_agent` | 3.12 | 44% |

### Representative reviews per theme

Closest to each centroid — what the cluster is actually about.

**Cluster 0** (1,285 reviews, `app_experience`)

> Bad experience with the app. Out of the 3 times, I placed the order, only once was my order delivered. The remaining times, it got cancelled without telling any reasons. When checked with the customer

> Very pathetic service. Order gets automatically cancelled or returned. If we place order of 10 items , we get only 5 products that too are very rare chances. Complained many times but the service is n

> Worst app ever , try some other app but never choose this , the orders always gets cancelled without even calling me , it has happened to me for 2 times in a day if I contact customer support, they al

**Cluster 4** (1,134 reviews, `customer_support`)

> Worst service I ever had, order takes a month to deliver but they won't deliver on time, even some of my orders are not delivered but i often get message as returned and some are after few days they c

> One of the worst service by them. First I ordered couple of items then it got shipped then after a while I receive messages it says delivery partner has cancelled your order. After waiting for 9 hours

> Pathetic delivery service. They keep cancelling the order even though the address is correct. They bill for items that is not being delivered. Customer care will inform theybwill refund the amount .. 

**Cluster 1** (738 reviews, `customer_support`)

> Had a really Very bad experience on zepto recently. They even refused to return the money i added to wallet. Scam alert for new users. Worst delivery app ever seen

> Very bad experience with this app if you are offering some zepto cash then let us use it as well. All the time I am trying to order something it is showing that zepto cash unavailable and price of pro

> Zepto lures the customers by giving free cash but when you login they say the zepto wallet cash is unavailable, customer care associate say to order 2 times then it will be available but after orderin

**Cluster 5** (620 reviews, `delivery_reliability`)

> Exceptionally good service! Quick delivery hardly ever delays, fresh products never faced any kind of issues, I have used this app for last minute needs and it really provides super fast delivery!

> The deliveries are on time and the range of products is really good. Also I find the prices much cheaper compared to other instant grocery apps.

> This app is incredibly convenient! The deliveries are always on time, and I love that they occasionally surprise me with free gifts. It's reliable and has become my go-to for quick and hassle-free ser

**Cluster 2** (437 reviews, `delivery_reliability`)

> Pathetic shopping experience with blinkit.. they don't even send items packed properly in a shopping bag.. they sent me 20 items ( loose items) in a basket which had to be returned.. so much unnecessa

> Blinkit has completely changed the way I shop for groceries and essentials. The app is incredibly user-friendly, and the delivery is super fast—often within minutes! The product quality is excellent, 

> Have been a regular user of blinkit since the past 2 years and my experience with the app has been quite good, however recently they have made a change in delivery system where they deliver things fro

**Cluster 3** (354 reviews, `customer_support`)

> I had a terrible experience with JioMart. My order placed on 30th June was canceled without notice. I tried again on 7th July, and after multiple delays, it was canceled on 11th July without any intim

> I’ve had terrible experiences with JioMart. Multiple times, I’ve paid for items that were never delivered, with no prior notice or refund. The products that did arrive were often of poor quality and b

> Never order from Jiomart you can better opt for other similar services. I'll give you point wise explanation to this: 1. They didn't deliver on time. Don't value for your time, they keep postponing de

---

## Caveats

**The score ranks, it does not measure.** It combines five signals on one weighting; a different weighting produces a different order. It is a way to argue about priority with the data present, not a cost model.

**Labels are model output, not ground truth.** Grounding was verified at 98.4% and coverage at 98.9%, but no hand-labelled gold set exists until Phase 9. Volume figures inherit whatever bias the model has.

**Clusters are unsupervised.** Silhouette picks the most separable k, not the most useful one. A theme that splits across two clusters, or two themes sharing one, are both possible and neither is an error.

**The clusters are weakly separated.** A silhouette of 0.123 is low in absolute terms — the peak is real (scores rise to it and fall after), but it describes overlapping regions of one continuous space, not distinct groups. That is what short review text usually looks like. Treat the themes as a reading aid over the ranked pain points, not as a partition of the corpus, and note that two clusters here share a dominant area: the split between them is about phrasing, not category.

**Severity is self-reported through the model.** It reflects how the review reads, not operational impact. A calm review of a serious failure scores low.

