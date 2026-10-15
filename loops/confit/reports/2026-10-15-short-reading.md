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
2026-10-14](2026-10-14-milestone.md). Since then, 14 PRs of the loop merged.
Two of them serve needs of the native loop, which builds the native catalog.
This cross-loop delivery starts a [report](../../reporting.md).

## What moved

**Confit answers as the oracle does in more cases.** #438 fixes the five
classes of defects that the last report found. With #440 and #441, confit
evaluates a regexp call over a NULL pattern and a NULL `struct_pack` field for
each row, as the oracle does. It parses an expression that mixes operators
without parentheses as the oracle does (#464). If a WHERE term of a comma join
can trap and comes before the join key, confit refuses the query (#451). A
campaign of 50,000 seeds on 9d9fe20 found no gated class.

**Confit serves the fourth kind of query in the order of a closed [decision
record](../decisions/closed/next-query-classes.md), and a part of the fifth.**
The fourth kind is more than one join under `shape='many'` (#449). The fifth
kind is an aggregate over the static rows that a request matches. Confit
serves it in a scalar subquery with one aggregate over a static table, such as
`count` or the `sum` of integers (#456, #457).

**Confit served two needs of the native loop.** With #459, the catalog entry
for PCA over 256 features with 64 components serves 64 rows in 4.9 to 5.2 ms,
against 59 ms before. #456 slowed the build of a struct that a query reads
field by field, and #461 removes this slowdown. Now the entry for
scikit-learn's RandomTreesEmbedding with 5,982 output fields builds in 5.8 to
6.2 s, against 158 s. Both PRs measured confit compiled for release.

## Needs the owner

- [An INNER join whose static keys are all NULL](../decisions/open/all-null-key-static.md):
  for seeds 7242549 and 7829571, confit traps, and the oracle returns no rows.
- [The open parts of the float-reduction bound](../decisions/open/double-sum-bound.md):
  confit refuses a DOUBLE `sum` or `avg`, and the oracle's sum of 1e16, 1 and -1e16 is 0 or 1, as the row order changes.
- [Empty-static witness: must the padded trap match confit's category?](../decisions/open/empty-static-witness-trap-order.md):
  for seed 4313391, confit traps with a conversion error on row 1. The oracle, with one padded static row, traps with an overflow on row 2.
- [JOIN with GROUP BY as an aggregate for each request](../decisions/open/join-group-by-per-request.md):
  with `GROUP BY t.k`, the oracle gives one row to two requests with k = 1, and its sum, 60, adds the matches of both.
- [The oracle times out where confit traps first](../decisions/open/oracle-timeout-before-a-trap.md):
  confit traps in all 35 cases of this record, and the oracle runs out of time in 19 when each runs alone.

## Next

Next, the loop serves a scalar subquery with WHERE terms that read only the
static table and cannot trap ([PLANS](../PLANS.md)). It also examines each
finding that the nightly campaign files in issue #435.
