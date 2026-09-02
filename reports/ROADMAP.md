# Roadmap — Scored Opportunities and Experiment Plans

**2026-09-02** · 10 pain points scored · 20 opportunities · 20 experiment plans

---

## Effort is missing, and that is deliberate

**No effort estimates were supplied, so this is a RIC ranking, not RICE.**

Reach, impact and confidence are properties of *the problem*, which is what customer reviews describe. Effort is a property of *the solution and the codebase that would carry it* — how many services it touches, what the migration looks like, who is free next sprint. No amount of review text contains that.

A model asked to guess it would produce a confident number, and the arithmetic would lend that guess authority. A RICE table with an invented denominator ranks work by fiction while looking quantitative — worse than no table.

To get a real ranking:

```bash
python scripts/09_build_roadmap.py --write-effort-template
```

Fill in `effort_person_weeks`, then:

```bash
python scripts/09_build_roadmap.py --effort data/processed/effort_template.csv
```

---

## Confidence is measured, not felt

Standard RICE picks confidence at 100/80/50% by feel. Here it is derived from evidence quality, so it answers *how much do we know* rather than *how sure does someone feel today*:

| Component | Weight | What it measures |
|---|---:|---|
| Grounding | 0.30 | were the labels' quotes verbatim in the reviews |
| Sample | 0.25 | is the volume enough to be a pattern |
| Label confidence | 0.20 | what the enrichment model reported |
| Mechanism | 0.15 | did Phase 6 find a grounded root cause |
| Competitive | 0.10 | did a platform difference survive correction |

---

## Ranking

Reach is **reviews per month** over 3 observed months — not customers. People who write app-store reviews are a small, self-selecting, annoyed slice of users. It is a consistent relative signal, which is what RICE needs; multiplying it by a user base would be inventing a number.

| # | Area | Issue | Reach/mo | Impact | Confidence | RIC |
|---|---|---|---:|---|---:|---:|
| 1 | `customer_support` | `unhelpful_agent` | 318 | high (2.0) | 0.97 | **616.4** |
| 2 | `customer_support` | `no_response` | 235 | high (2.0) | 0.97 | **457.2** |
| 3 | `refunds` | `refund_not_received` | 116 | massive (3.0) | 0.98 | **339.4** |
| 4 | `order_lifecycle` | `unwanted_cancellation` | 146 | high (2.0) | 0.97 | **284.6** |
| 5 | `order_fulfilment` | `missing_items` | 126 | high (2.0) | 0.98 | **246.2** |
| 6 | `delivery_reliability` | `late_delivery` | 155 | medium (1.0) | 0.97 | **150.7** |
| 7 | `delivery_reliability` | `never_delivered` | 80 | high (2.0) | 0.93 | **148.1** |
| 8 | `product_quality` | `damaged_product` | 57 | high (2.0) | 0.87 | **99.3** |
| 9 | `returns_and_replacement` | `return_request_rejected` | 33 | high (2.0) | 0.81 | **53.1** |
| 10 | `refunds` | `partial_refund` | 6 | massive (3.0) | 0.74 | **13.3** |

---

## Opportunities

### Increase Human Customer Support Agent Staffing

`customer_support` / `unhelpful_agent`

**Change.** Hire and train 20 additional human customer support agents to reduce wait times and increase availability

**Addresses.** Inadequate staffing of human customer support agents

**Success.** `severe_share` should **decrease**.

**Risk if wrong.** Increased operational costs without corresponding improvement in customer satisfaction, potentially wasting resources

**Experiment.**

| | |
|---|---|
| Primary metric | `severe_share` |
| Guardrail | `negative_share` |
| Baseline | 53.0% |
| Target | 50.0% (3pp) |
| Sample needed | 4,356 per arm (8,712 total) |
| Duration | 6.5 months at review volume |
| Powered? | **no** |

> Underpowered at review volume: 8,712 reviews needed, 1,331 arrive per month.

---

### Implement Hybrid Support Model with Human Escalation

`customer_support` / `unhelpful_agent`

**Change.** Route customer support queries that are not resolved by AI bots within 2 interactions to a human customer support agent queue, ensuring timely human intervention for complex issues

**Addresses.** Over-reliance on automated systems (AI bots) for customer support

**Success.** `escalation_rate` should **decrease**.

**Risk if wrong.** Increased wait times for customers if human agents are not adequately prepared to handle escalated issues, leading to further frustration

**Experiment.**

| | |
|---|---|
| Primary metric | `escalation_rate` |
| Guardrail | `negative_share` |
| Baseline | 34.8% |
| Target | 31.8% (3pp) |
| Sample needed | 3,871 per arm (7,742 total) |
| Duration | 5.8 months at review volume |
| Powered? | yes |

---

### Route Complex Issues to Human Support Agents

`customer_support` / `no_response`

**Change.** Implement a system to route tickets that are not resolved by automated responses to a human support queue, ensuring customers receive personalized support

**Addresses.** Automated response system

**Success.** `negative_share` should **decrease**.

**Risk if wrong.** Increased wait times for customers if human support agents are not adequately staffed or trained

**Experiment.**

| | |
|---|---|
| Primary metric | `negative_share` |
| Guardrail | `praise_share` |
| Baseline | 76.4% |
| Target | 73.4% (3pp) |
| Sample needed | 3,280 per arm (6,560 total) |
| Duration | 4.9 months at review volume |
| Powered? | yes |

---

### Increase Support Team Capacity and Implement Priority Routing

`customer_support` / `no_response`

**Change.** Hire additional support staff and implement a priority routing system to ensure timely responses to customer queries, with high-priority issues addressed first

**Addresses.** Overwhelmed support team

**Success.** `escalation_rate` should **decrease**.

**Risk if wrong.** Wasted resources if the support team is not effectively utilized or if priority routing is not properly implemented

**Experiment.**

| | |
|---|---|
| Primary metric | `escalation_rate` |
| Guardrail | `negative_share` |
| Baseline | 34.8% |
| Target | 31.8% (3pp) |
| Sample needed | 3,871 per arm (7,742 total) |
| Duration | 5.8 months at review volume |
| Powered? | yes |

---

### Automate Refund Processing

`refunds` / `refund_not_received`

**Change.** Implement an automated refund processing system to reduce manual intervention and initiate refunds immediately after approval

**Addresses.** Refund processing is delayed or not initiated due to internal communication issues or inefficient systems

**Success.** `pain_point_volume` should **decrease**.

**Risk if wrong.** Customers may experience incorrect or duplicate refunds, leading to financial losses and damage to trust

**Experiment.**

| | |
|---|---|
| Primary metric | `pain_point_volume` |
| Guardrail | `negative_share` |
| Baseline | 7.6% |
| Target | 4.6% (3pp) |
| Sample needed | 1,001 per arm (2,002 total) |
| Duration | 1.5 months at review volume |
| Powered? | yes |

---

### Enhance Customer Support for Refund Issues

`refunds` / `refund_not_received`

**Change.** Provide customer support agents with specialized training and tools to resolve refund issues efficiently and effectively, and establish a clear escalation process for complex cases

**Addresses.** Customers are not receiving refunds due to a lack of effective customer support

**Success.** `escalation_rate` should **decrease**.

**Risk if wrong.** Customers may experience longer wait times or unhelpful support interactions, leading to increased frustration and churn

**Experiment.**

| | |
|---|---|
| Primary metric | `escalation_rate` |
| Guardrail | `negative_share` |
| Baseline | 34.8% |
| Target | 31.8% (3pp) |
| Sample needed | 3,871 per arm (7,742 total) |
| Duration | 5.8 months at review volume |
| Powered? | yes |

---

### Introduce Order Cancellation Option

`order_lifecycle` / `unwanted_cancellation`

**Change.** Add a clear and accessible order cancellation feature within the app, allowing customers to cancel their orders directly

**Addresses.** Lack of order cancellation option

**Success.** `pain_point_volume` should **decrease**.

**Risk if wrong.** Increased support contact volume due to confusion about the new cancellation feature

**Experiment.**

| | |
|---|---|
| Primary metric | `pain_point_volume` |
| Guardrail | `negative_share` |
| Baseline | 9.6% |
| Target | 6.6% (3pp) |
| Sample needed | 1,296 per arm (2,592 total) |
| Duration | 1.9 months at review volume |
| Powered? | yes |

---

### Enhance Customer Support Training

`order_lifecycle` / `unwanted_cancellation`

**Change.** Develop and implement a comprehensive training program for customer support agents to handle order cancellation requests effectively

**Addresses.** Inadequate customer support training

**Success.** `escalation_rate` should **decrease**.

**Risk if wrong.** Decreased customer satisfaction due to poorly trained support agents leading to further escalation

**Experiment.**

| | |
|---|---|
| Primary metric | `escalation_rate` |
| Guardrail | `negative_share` |
| Baseline | 34.8% |
| Target | 31.8% (3pp) |
| Sample needed | 3,871 per arm (7,742 total) |
| Duration | 5.8 months at review volume |
| Powered? | yes |

---

### Automate Refund Process

`refunds` / `partial_refund`

**Change.** Implement an automated refund processing system to reduce manual intervention and minimize delays

**Addresses.** Refund process is not automated and requires manual intervention, leading to delays and incomplete refunds

**Success.** `escalation_rate` should **decrease**.

**Risk if wrong.** Increased error rate in refund processing, potentially leading to more customer complaints and financial losses

**Experiment.**

| | |
|---|---|
| Primary metric | `escalation_rate` |
| Guardrail | `negative_share` |
| Baseline | 34.8% |
| Target | 31.8% (3pp) |
| Sample needed | 3,871 per arm (7,742 total) |
| Duration | 5.8 months at review volume |
| Powered? | yes |

---

### Clear Refund Policy Communication

`refunds` / `partial_refund`

**Change.** Add a clear and concise refund policy statement to the app's FAQ section and order confirmation emails

**Addresses.** Refund policy is not clearly communicated to customers, leading to confusion and frustration

**Success.** `pain_point_volume` should **decrease**.

**Risk if wrong.** Customer confusion and frustration due to inconsistent or misleading refund policy information, potentially leading to negative reviews and churn

**Experiment.**

| | |
|---|---|
| Primary metric | `pain_point_volume` |
| Guardrail | `negative_share` |
| Baseline | 0.4% |
| Target | 3.4% (3pp) |
| Sample needed | 323 per arm (646 total) |
| Duration | 0.5 months at review volume |
| Powered? | yes |

> Baseline of 0.4% is very low; rare-event effects need far more data than the default MDE assumes.

---

### Improve Inventory Management

`delivery_reliability` / `never_delivered`

**Change.** Implement a real-time inventory tracking system to ensure accurate inventory levels and prevent orders from being marked as 'out for delivery' when items are not available

**Addresses.** Inadequate inventory management leads to missing items and undelivered orders

**Success.** `pain_point_volume` should **decrease**.

**Risk if wrong.** Increased costs due to overstocking or wasted resources on incorrect inventory management

**Experiment.**

| | |
|---|---|
| Primary metric | `pain_point_volume` |
| Guardrail | `negative_share` |
| Baseline | 5.2% |
| Target | 2.2% (3pp) |
| Sample needed | 630 per arm (1,260 total) |
| Duration | 0.9 months at review volume |
| Powered? | yes |

---

### Enhance Delivery Communication and Transparency

`delivery_reliability` / `never_delivered`

**Change.** Introduce automated updates and notifications to customers about the status of their orders, including estimated delivery times and any potential delays

**Addresses.** Poor communication and lack of transparency in the delivery process lead to customer frustration

**Success.** `severe_share` should **decrease**.

**Risk if wrong.** Customer frustration and dissatisfaction due to information overload or irrelevant updates

**Experiment.**

| | |
|---|---|
| Primary metric | `severe_share` |
| Guardrail | `negative_share` |
| Baseline | 53.0% |
| Target | 50.0% (3pp) |
| Sample needed | 4,356 per arm (8,712 total) |
| Duration | 6.5 months at review volume |
| Powered? | **no** |

> Underpowered at review volume: 8,712 reviews needed, 1,331 arrive per month.

---

### Improve Inventory Management Accuracy

`order_fulfilment` / `missing_items`

**Change.** Implement a real-time inventory management system that updates stock levels automatically after each sale, and conduct regular audits to ensure accuracy

**Addresses.** Inadequate inventory management

**Success.** `pain_point_volume` should **decrease**.

**Risk if wrong.** Overstocking or understocking of items, leading to wasted resources or lost sales

**Experiment.**

| | |
|---|---|
| Primary metric | `pain_point_volume` |
| Guardrail | `negative_share` |
| Baseline | 8.3% |
| Target | 5.3% (3pp) |
| Sample needed | 1,101 per arm (2,202 total) |
| Duration | 1.6 months at review volume |
| Powered? | yes |

---

### Enhance Quality Control During Order Preparation

`order_fulfilment` / `missing_items`

**Change.** Introduce a mandatory double-check process for order preparation, where two staff members verify that all items are included in the order before it is shipped

**Addresses.** Insufficient quality control during order preparation

**Success.** `pain_point_volume` should **decrease**.

**Risk if wrong.** Increased labor costs and potential delays in order fulfillment if the double-check process is not efficient

**Experiment.**

| | |
|---|---|
| Primary metric | `pain_point_volume` |
| Guardrail | `negative_share` |
| Baseline | 8.3% |
| Target | 5.3% (3pp) |
| Sample needed | 1,101 per arm (2,202 total) |
| Duration | 1.6 months at review volume |
| Powered? | yes |

---

### Improve Estimated Delivery Times

`delivery_reliability` / `late_delivery`

**Change.** Update the app to display more realistic delivery times based on historical data and current traffic conditions

**Addresses.** Inaccurate estimated delivery times

**Success.** `negative_share` should **decrease**.

**Risk if wrong.** Customers may be deterred by longer estimated delivery times, potentially leading to a loss of sales

**Experiment.**

| | |
|---|---|
| Primary metric | `negative_share` |
| Guardrail | `praise_share` |
| Baseline | 76.4% |
| Target | 73.4% (3pp) |
| Sample needed | 3,280 per arm (6,560 total) |
| Duration | 4.9 months at review volume |
| Powered? | yes |

---

### Introduce Real-Time Tracking and Updates

`delivery_reliability` / `late_delivery`

**Change.** Implement a real-time tracking system that provides customers with accurate and up-to-date information on their delivery status

**Addresses.** Lack of real-time tracking and updates

**Success.** `severe_share` should **decrease**.

**Risk if wrong.** Technical issues with the tracking system may lead to increased customer frustration and dissatisfaction

**Experiment.**

| | |
|---|---|
| Primary metric | `severe_share` |
| Guardrail | `negative_share` |
| Baseline | 53.0% |
| Target | 50.0% (3pp) |
| Sample needed | 4,356 per arm (8,712 total) |
| Duration | 6.5 months at review volume |
| Powered? | **no** |

> Underpowered at review volume: 8,712 reviews needed, 1,331 arrive per month.

---

### Clear Return Policy Communication

`returns_and_replacement` / `return_request_rejected`

**Change.** Add a clear and concise return policy section to the website and include it in the order confirmation email

**Addresses.** Inconsistent or missing information about return policies leads to rejected return requests

**Success.** `pain_point_volume` should **decrease**.

**Risk if wrong.** Customers may still experience frustration if the return policy is not flexible or customer-friendly, leading to negative reviews and potential churn

**Experiment.**

| | |
|---|---|
| Primary metric | `pain_point_volume` |
| Guardrail | `negative_share` |
| Baseline | 2.2% |
| Target | 5.2% (3pp) |
| Sample needed | 616 per arm (1,232 total) |
| Duration | 0.9 months at review volume |
| Powered? | yes |

---

### Human Evaluation of Return Requests

`returns_and_replacement` / `return_request_rejected`

**Change.** Route return requests that are initially rejected by automated systems to a human customer support agent for review and evaluation

**Addresses.** Automated or unresponsive customer support systems cause return requests to be rejected without proper evaluation

**Success.** `escalation_rate` should **decrease**.

**Risk if wrong.** Increased workload on human customer support agents may lead to longer response times and decreased customer satisfaction if not properly staffed or trained

**Experiment.**

| | |
|---|---|
| Primary metric | `escalation_rate` |
| Guardrail | `negative_share` |
| Baseline | 34.8% |
| Target | 31.8% (3pp) |
| Sample needed | 3,871 per arm (7,742 total) |
| Duration | 5.8 months at review volume |
| Powered? | yes |

---

### Enhance Quality Control Measures

`product_quality` / `damaged_product`

**Change.** Implement additional inspection and testing procedures for products before delivery

**Addresses.** Inadequate quality control measures

**Success.** `pain_point_volume` should **decrease**.

**Risk if wrong.** Increased cost of inspection and testing without a corresponding decrease in damaged products

**Experiment.**

| | |
|---|---|
| Primary metric | `pain_point_volume` |
| Guardrail | `negative_share` |
| Baseline | 3.7% |
| Target | 0.7% (3pp) |
| Sample needed | 382 per arm (764 total) |
| Duration | 0.6 months at review volume |
| Powered? | yes |

---

### Improve Customer Support Process for Damaged Products

`product_quality` / `damaged_product`

**Change.** Provide specialized training for customer support agents on handling damaged product issues and equip them with necessary tools and resources

**Addresses.** Inefficient customer support process

**Success.** `escalation_rate` should **decrease**.

**Risk if wrong.** Wasted resources on unnecessary training and equipment without a corresponding improvement in customer satisfaction

**Experiment.**

| | |
|---|---|
| Primary metric | `escalation_rate` |
| Guardrail | `negative_share` |
| Baseline | 34.8% |
| Target | 31.8% (3pp) |
| Sample needed | 3,871 per arm (7,742 total) |
| Duration | 5.8 months at review volume |
| Powered? | yes |

---

## Caveats

**Sample sizes are in reviews, not users.** An experiment run on users has a different denominator and will usually reach power far sooner. These numbers say how long it would take to *see the effect in reviews*, which is the slowest possible instrument.

**Opportunities are model-generated.** Each is tied to a validated root-cause hypothesis and names a metric this pipeline measures, which makes it checkable — not correct.

**Impact is derived from severity, churn and escalation**, all of which are model labels over self-reported customer text. They describe how a review reads, not operational cost.

**An underpowered experiment cannot produce a null result.** Where `Powered? no` appears, a flat outcome means the test could not have detected the effect, not that there was none.

