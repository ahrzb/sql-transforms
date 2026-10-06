---
loop: confit
date: 2026-10-06
master: 5634d6a
previous: loops/confit/reports/2026-10-05-wave1-reading.md
kind: short
trigger: cross-loop delivery
kpis:
  training_round_trip: null
  engine_parity: null
  binding_parity: null
  transformer_parity: null
  no_third_mode: null
  coverage_ladder: null
  serving_latency: null
  corpus_match: null
  corpus_total: null
  corpus_fail: null
  gated_campaign_classes: null
  catalog_needs_open: 2
  catalog_needs_served_since_last: 4
needs_owner:
  - loops/confit/decisions/open/empty-static-witness-trap-order.md
  - loops/confit/decisions/open/oracle-timeout-before-a-trap.md
needs_from_other_loop: []
---

# Confit: short reading, 2026-10-06

This reading names master 5634d6a and follows the [reading of
2026-10-05](2026-10-05-wave1-reading.md). It does not measure the success
measures, the corpus or the campaign classes again, so their keys are null.
The last two readings listed two items under "Needs the owner", but neither
item had a decision record. The owner had ruled on the first item on
2026-09-27: the number of cases in a run of the training round trip (C1).
Confit triaged the second item itself (#398).

## What moved

**Confit served four needs of the native loop.** `greatest` and `least`
evaluate each argument once (#374). Confit now treats `round` over DOUBLE as
a value that cannot trap (#375). Under a CASE guard that excludes infinity,
it treats `sin`, `cos` and `tan` the same way. If only different CASE
branches read a value, confit no longer computes it on every row (#377). The
cube root `cbrt` calls the GNU C library, as DuckDB does (#390).

**Confit triaged the OPT_EMULATED findings of 2026-10-05 (#398).**
OPT_EMULATED is the verdict where confit agrees with the optimizer-on
reading but not with the oracle. Confit now refuses two of the four classes
by name: `nullif` over a value that can trap, and a join condition that
compares the two sides through an operand that can trap. The empty-static
ruling now excuses the third class. The fourth class needs the owner.

**In progress:** a SQL function (`SqlFunction`, a SQL expression with
parameters) that binds a value once. It serves the two open needs of the
native loop: the B-spline recurrence of SplineTransformer, and the leaf CASE
of each tree in RandomTreesEmbedding. A first change for the spline need
merged (#387). Confit now computes a shared value just before the first item
that reads it, not before all items.

## Needs the owner

- [Empty-static witness: trap order](../decisions/open/empty-static-witness-trap-order.md).
  For seed 4313391, confit traps on row 1 with a conversion error, and the
  witness traps in DuckDB on row 2 with an overflow.
- [The oracle times out where confit traps first](../decisions/open/oracle-timeout-before-a-trap.md).
  For seeds 4226438 and 946454, confit traps on an overflow, but DuckDB
  first builds a string of about 2 GiB and times out.

## Next

- Merge the SQL function that binds a value once, then tell the native loop.
- Triage the nightly campaign of 2026-10-06.
- Serve queries that return a struct column, the owner's third query class.
