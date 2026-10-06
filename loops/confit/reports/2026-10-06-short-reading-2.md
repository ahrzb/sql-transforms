---
loop: confit
date: 2026-10-06
master: 4865d9f
previous: loops/confit/reports/2026-10-06-short-reading.md
kind: short
trigger: cross-loop delivery
kpis:
  training_round_trip: null
  engine_parity: "pass: 0 gated classes in 10,000 campaign seeds (1020000-1029999) on 4865d9f"
  binding_parity: null
  transformer_parity: null
  no_third_mode: null
  coverage_ladder: null
  serving_latency: null
  corpus_match: null
  corpus_total: null
  corpus_fail: null
  gated_campaign_classes: 0
  catalog_needs_open: 0
  catalog_needs_served_since_last: 2
needs_owner:
  - loops/confit/decisions/open/empty-static-witness-trap-order.md
  - loops/confit/decisions/open/oracle-timeout-before-a-trap.md
needs_from_other_loop: []
---

# Confit: second short reading, 2026-10-06

This reading names master 4865d9f and follows the [first short reading of
2026-10-06](2026-10-06-short-reading.md). It measures only the campaign
again, so the keys of the other success measures and the corpus are null.

## What moved

**Confit served the last two open needs of the native loop (#412).** The
native loop builds the native catalog. Both needs concern a SQL function
(`SqlFunction`), a function whose body is one SQL expression over its
parameters. Before, confit replaced each call with the body text, so a
value that the body read twice was in that text twice. Now confit binds
such a value once for each call. The oracle still runs the body as text,
so the definition does not change.

**Confit now builds the native loop's expressions at larger sizes.** These
measurements use a release build (compiled with optimizations) of 4865d9f:

- For the scikit-learn SplineTransformer at degree 5, 7 knots and 32
  features, confit builds the catalog entry's SQL function in 2.1 s.
  Before, confit refused it past the limit of 4,000,000 SQL tokens.
- For the scikit-learn RandomTreesEmbedding with 100 trees of depth 5
  (2,286 output fields), confit builds the form that the native loop asked
  for in 1.3 s. That form has one CASE expression for each tree. It serves
  a row in 118 µs (batches of 1,024 rows), and its output equals the
  twin's. The entry on master uses another form, which builds in 1.8 s and
  serves a row in 202 µs.

The SplineTransformer entry still refuses each step whose estimated build
time is above 7 s. Confit sent the new build times to the native loop.

**Parity held.** A campaign of 10,000 seeds on 4865d9f found no gated
class. The campaign does not generate SQL functions. Instead, 26 tests
compare SQL functions with the oracle (`tests/test_sql_function_lets.py`).

## Needs the owner

- [Empty-static witness: trap order](../decisions/open/empty-static-witness-trap-order.md).
  For seed 4313391, confit traps on row 1 with a conversion error. The
  witness is DuckDB's re-run of the case with one row in each empty static
  table. It traps on row 2 with an overflow.
- [The oracle times out where confit traps first](../decisions/open/oracle-timeout-before-a-trap.md).
  For seeds 4226438 and 946454, confit traps on an overflow, but DuckDB
  first builds a string of about 2 GiB and times out.

## Next

- Examine the findings of the nightly campaign of 2026-10-06.
- Serve queries whose output has a struct column. This is the third class
  in the owner's [ruled order](../decisions/closed/next-query-classes.md).
