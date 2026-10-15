---
loop: confit
date: 2026-10-15
master: 5e641eb
previous: loops/confit/reports/2026-10-15-short-reading.md
kind: short
trigger: staleness
kpis:
  training_round_trip: null
  engine_parity: "pass: 0 gated classes in 50,000 campaign seeds (31000000-31049999) on 5e641eb"
  binding_parity: null
  transformer_parity: null
  no_third_mode: null
  coverage_ladder: null
  serving_latency: null
  corpus_match: null
  corpus_total: null
  corpus_fail: null
  gated_campaign_classes: 0
  catalog_needs_open: 3
  catalog_needs_served_since_last: 0
needs_owner:
  - loops/confit/decisions/open/join-group-by-per-request.md
needs_from_other_loop: []
---

# Confit: short reading 2, 2026-10-15

This reading names master 5e641eb and follows the [first short reading of
2026-10-15](2026-10-15-short-reading.md). Since then, 15 PRs of the confit loop
merged, so the [reporting policy](../../reporting.md) asks for a short reading.
A campaign of 50,000 seeds (31000000 to 31049999) on 5e641eb found no gated class.

## What moved

**Confit serves more queries with an aggregate over the static rows that a
request matches.** A scalar subquery over a static table can now have WHERE
terms that read only the static table (#468), a first term that joins the
tables, such as `s.k = t.x + 1`, in which `t.x + 1` can trap (#470), an
expression over its aggregates (#471) and an aggregate over DISTINCT values
(#472). Confit also serves EXISTS and NOT EXISTS (#473), `x IN (SELECT ...)`
(#474) and comparisons with ANY and ALL, such as `x > ANY (SELECT ...)` (#476),
over a static table. A scalar subquery can read the one static row that a
request matches (#480) or that its ORDER BY and LIMIT keep (#481).

**Confit answers as the oracle does in more cases.** In `x IN (SELECT ...)`,
confit evaluates `x` only after a static row whose value is not NULL passes the
WHERE clause (#475). If two DECIMAL values need a common type with more than 38
digits, confit compares them as DECIMAL(38, s), with the larger scale s, as the
oracle does (#477). Confit now refuses an ambiguous column name in a join term
of a comma join, and the oracle cannot build such a query either (#479). In the
WHERE clause of a comma join, a string operation that cannot trap may now come
before the join term (#469). #478 fixes a defect of confit that stopped the
nightly campaign of 2026-10-15.

**The loop applied two rulings of the owner (#483).** The exclusion
`empty-static-trap-timing` removes a case from the comparison if confit traps
where an INNER or CROSS join reads an empty static table. Its witness adds a
row to the static table, and the oracle must then raise an error. The rule now
also covers an INNER join whose static rows each have a NULL key ([all-NULL-key
static](../decisions/closed/all-null-key-static.md)). The error may now be of
any kind ([error kind](../decisions/closed/empty-static-witness-trap-order.md)).
The rule excused 6 cases in the campaign of this reading.

## Needs the owner

- [JOIN with GROUP BY as an aggregate for each request](../decisions/open/join-group-by-per-request.md):
  with `GROUP BY` on a request column, two requests with the same value share one row and one sum.

## Next

The loop applies the two other rulings of 2026-10-15 and closes their records.
One excludes a case where the oracle runs out of time before a trap. The other
serves a DOUBLE `sum` and `avg` with a bound on the rounding error. Then the
loop serves more aggregates, such as `bool_and` and `count_if` ([PLANS](../PLANS.md)).
