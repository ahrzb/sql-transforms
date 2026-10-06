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
  catalog_needs_served_since_last: 5
needs_owner:
  - loops/confit/decisions/open/empty-static-witness-trap-order.md
  - loops/confit/decisions/open/oracle-timeout-before-a-trap.md
needs_from_other_loop: []
---

# Confit: short reading, 2026-10-06

This reading follows the [reading of 2026-10-05](2026-10-05-wave1-reading.md).
It names master commit 5634d6a. It does not measure the success measures
again, so their keys are null. The next reading measures them after the work
in progress merges.

This reading also corrects "Needs the owner". The last two readings listed
two items there that had no decision record. The owner ruled on the first
item, the case count of the C1 control, on 2026-09-27. Confit fixes in #398
closed the second item.

## What moved

**Confit served five needs of the native loop.** The SQL functions `greatest`
and `least` evaluate each argument once (#374). `round` over DOUBLE cannot
raise. `sin`, `cos` and `tan` cannot raise under a guard that excludes
infinity (#375). Confit no longer computes, on every row, a value that only
different CASE branches read (#377). It computes a shared value just before
the first item that reads it (#387). `cbrt` calls the GNU C library, as
DuckDB does (#390).

**Confit triaged the OPT_EMULATED findings of 2026-10-05 (#398).**
OPT_EMULATED is the verdict where confit matches the optimizer-on reading but
not the oracle. Two classes were confit defects: `nullif` over a value that
can trap, and a join condition between the two sides that can trap. Confit
now refuses both by name. Two other cases need the owner, and each has a
decision record.

**In progress:** a value bound once in a SQL function body. It serves the two
open needs of the native loop: the de Boor recurrence of SplineTransformer,
and the leaf tests of RandomTreesEmbedding.

## Needs the owner

- [Empty-static witness: trap order](../decisions/open/empty-static-witness-trap-order.md).
  The witness re-runs the case with one row in the empty static table, and
  DuckDB must trap with confit's error. For seed 4313391, the two engines
  trap on different rows first, so the errors differ.
- [The oracle times out where confit traps first](../decisions/open/oracle-timeout-before-a-trap.md).
  Seeds 4226438 and 946454: confit traps on an overflow, but DuckDB first
  builds a string of about 2 GiB and runs past its time budget.

## Next

- Finish the value bound once in a SQL function body, then tell the native loop.
- Triage the nightly campaign of 2026-10-06.
- Serve queries that return a struct column, the third query class in the
  owner's order.
