---
loop: confit
date: 2026-10-15
master: 9d9fe20
previous: loops/confit/reports/2026-10-14-milestone.md
kind: short
trigger: cross-loop delivery
kpis:
  training_round_trip: null
  engine_parity: "pass: 0 gated classes in 50,000 campaign seeds (30000000-30049999) on 9d9fe20"
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
  catalog_needs_served_since_last: 2
needs_owner:
  - loops/confit/decisions/open/all-null-key-static.md
  - loops/confit/decisions/open/double-sum-bound.md
  - loops/confit/decisions/open/empty-static-witness-trap-order.md
  - loops/confit/decisions/open/join-group-by-per-request.md
  - loops/confit/decisions/open/oracle-timeout-before-a-trap.md
needs_from_other_loop: []
---

# Confit: short reading, 2026-10-15

This reading names master 9d9fe20 and follows the [milestone report of
2026-10-14](2026-10-14-milestone.md). Since that report, 14 PRs of the loop
merged. Two of them serve needs of the native loop, and this event starts the
reading. The loop measured again only the campaign and those needs.

## What moved

**Confit answers as the oracle does in more cases.** #438 fixes the five
classes of defects in the last report. #440 and #441 make confit evaluate
two NULL expressions for each row, as the oracle does. Confit now reads
operators mixed without parentheses as the oracle does (#464). It refuses a
comma join with a WHERE term that can trap before the join key (#451). A
campaign of 50,000 seeds on 9d9fe20 found no gated class.

**Confit serves the fourth kind of query of the owner's [ruled
order](../decisions/closed/next-query-classes.md), and a part of the
fifth.** It serves more than one join under `shape='many'` (#449). It serves
a scalar subquery with one aggregate over a static table (#456, #457). The
aggregate is `count`, `min`, `max`, or a `sum` or `avg` of integer or
DECIMAL values.

**Confit served two needs of the native loop.** With #459, confit computes a
small CASE that cannot trap without branches. So the catalog's PCA over 256
features with 64 components serves 64 rows in 4.9 to 5.2 ms, against 59 ms.
The other need was a slowdown that #456 caused, and #461 removes it. A
RandomTreesEmbedding of 5,982 fields built in 158 s before #461, and in 5.8
to 6.2 s after it. The PRs measured these numbers on release builds.

## Needs the owner

- [An INNER join whose static keys are all NULL](../decisions/open/all-null-key-static.md):
  for seeds 7242549 and 7829571, confit traps, and the oracle returns no rows.
- [The open parts of the float-reduction bound](../decisions/open/double-sum-bound.md):
  the oracle's sum of 1e16, 1 and -1e16 is 0 or 1, as the order changes.
- [Empty-static witness: must the padded trap match confit's category?](../decisions/open/empty-static-witness-trap-order.md):
  for seed 4313391, confit traps on row 1, and the oracle, run with one static row, traps on row 2.
- [JOIN with GROUP BY as an aggregate for each request](../decisions/open/join-group-by-per-request.md):
  with `GROUP BY t.k`, two requests with k = 1 get one row, with the sum 60.
- [The oracle times out where confit traps first](../decisions/open/oracle-timeout-before-a-trap.md):
  in 19 of its 35 cases, the oracle still runs out of time when each runs alone.

## Next

The loop continues with the fifth kind of query, in the order of
[PLANS](../PLANS.md). First, it serves the WHERE terms of a scalar subquery
that read only the static table and cannot trap. It also examines each
finding that the nightly campaign files in issue #435.
