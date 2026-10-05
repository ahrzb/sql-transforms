---
loop: confit
date: 2026-10-05
master: 35617d7
previous: loops/confit/reports/2026-10-05-short-reading.md
kind: short
trigger: batch
kpis:
  training_round_trip: "pass: C1 at the default depth 25 in the gate on 35617d7; the written depth 1,500 not re-run (finding c1-depth open)"
  engine_parity: "pass: 1 gated class in 20k campaign seeds (930000-949999), fixed (#364); 2 TIMEOUT:oracle seeds are the known lpad class"
  binding_parity: "pass: _serving_test.py green in the gate"
  transformer_parity: "pass: _transformers_test.py / _trees_test.py green in the gate"
  no_third_mode: "pass: corpus 0 FAIL; campaigns show no FAILED verdict"
  coverage_ladder: "pass: corpus match 543 (floor 539, +1 since the last reading)"
  serving_latency: "open: not re-measured for this reading"
  corpus_match: 543
  corpus_total: 678
  corpus_fail: 0
  gated_campaign_classes: 0
  catalog_needs_open: 1
  catalog_needs_served_since_last: 5
needs_owner:
  - "OPT_EMULATED: nightly 2026-10-05 seeds 4291817, 4313391"
  - "c1-depth: raise the C1 default depth to 1,500 or correct the text"
needs_from_other_loop: []
---

# Confit: wave 1 reading, 2026-10-05

Follows the [short reading](2026-10-05-short-reading.md) of this afternoon.
Measured on master 35617d7 (release build): corpus with
`scripts/corpus_counts.py`, gate with `scripts/gate.py`, timings with the
repros quoted in each PR.

## What moved

**Wave 1 merged: three workers, five rounds of review.** UBIGINT/HUGEINT on a
128-bit lane (#349), shared subexpressions within a call (#363), and an early
size refusal (#358). The reviews found two parity defects in #349, a merge
that would not compile and undid #362 in #363, and stale numbers in #358;
all were fixed before merge. The loop moved to `loops/confit/` (#370).

**Five catalog needs served since the last reading.** Siblings that cannot
raise are not evaluated (#362: Box-Cox at 64 features 3,448 → 52 µs per row,
as the catalog measured; `-x` spelled again); a repeated subexpression is
computed once (#363: Normalizer l2 at 32 features builds in 0.5 s, refused
after 18 s before); the early refusal comes in 2.5 s instead of 6 to 67 s
(#358, #371); and a HUGEINT past 38 digits inside a struct leaves as DuckDB
exports it (#364, a campaign find). Open: two CASE trees in one expression
(QuantileTransformer, 4.48 → 1.76 s at 2,000 quantiles, still about 3.8×
one tree; low priority).

**Parity held.** The corpus matches 543 of 678 with 0 FAIL (542 before; one
statement served by HUGEINT).

## Needs the owner

- **OPT_EMULATED, 2 classes** (nightly 2026-10-05, seeds 4291817 and
  4313391): confit matches optimizer-off DuckDB where the optimizer prunes a
  trapping expression. Accept them like DIVERGE_OPT, or emulate?
- **finding: c1-depth**: raise the C1 default depth to 1,500, or correct
  the text.

## Next

Inline while the account's usage warning stands: struct-valued outputs
(ruled class 3), then the follow-ups in PLANS.
