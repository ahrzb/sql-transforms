---
loop: confit
date: 2026-10-05
master: 5513891
previous: packages/confit/docs/reports/2026-09-26-goal-reading-n3.md
kind: short
trigger: staleness
kpis:
  training_round_trip: "pass: C1 at the default depth 25 in the gate on 5513891; the written depth 1,500 not re-run (finding c1-depth open)"
  engine_parity: "pass: 0 gated classes over 20k campaign seeds today; nightly 2026-10-05's DIVERGE_VALUE/DIVERGE_TRAP fixed (#330), no longer reproduce"
  binding_parity: "pass: _serving_test.py green in the gate"
  transformer_parity: "pass: _transformers_test.py / _trees_test.py green in the gate"
  no_third_mode: "pass: corpus 0 FAIL; campaigns show no FAILED verdict"
  coverage_ladder: "pass: corpus match 542 (floor 539, +3 since 2026-09-26)"
  serving_latency: "open: not re-measured for this reading"
  corpus_match: 542
  corpus_total: 678
  corpus_fail: 0
  gated_campaign_classes: 0
  catalog_needs_open: 1
  catalog_needs_served_since_last: 8
needs_owner:
  - "OPT_EMULATED: nightly 2026-10-05 seeds 4291817, 4313391"
  - "c1-depth: raise the C1 default depth to 1,500 or correct the text"
needs_from_other_loop: []
---

# Confit: short reading, 2026-10-05

Follows [goal reading N=3](2026-09-26-goal-reading-n3.md). Since then about 80
confit commits have merged, 29 of them today. Numbers below were taken on
master 5513891. Corpus counts come from
`scripts/corpus_counts.py`; campaigns used seeds 910000–929999 on a release
build; the gate (`scripts/gate.py`) ran green with 5,856 passed.

## What moved

**The native catalog's needs are served as they arrive: 8 since the last
reading.** Constant CASE results are not sibling traps and a Cranelift size
limit refuses by name (#336, #346); `greatest`/`least` fold flat (#337);
siblings keep only trap skeletons (#338); a call expands and binds once
(#339, #348); struct reads build linearly, `null_when` included (#350: 1,035
lanes 16.4 s → 0.8 s, 3,240 in 4.0 s); a dropped function frees its JIT
memory (#353: 2 mappings leaked per build, stalling near 32k builds).
One need is open: a subexpression shared within one call (worker T1).

**The served SQL grew:** DECIMAL expressions, row columns and join keys
(#334, #335); UTINYINT/USMALLINT/UINTEGER and depth 1000 (#341); `error()` as
a CASE result, list literals, struct field reads (#325–#327).
The mined corpus moved from 539 to 542 matches of 678, with 0 FAIL.
UBIGINT/HUGEINT (worker T3, #349) is in review: two blocking parity defects
were sent back.

**Parity held.** No gated class appeared in 20k campaign seeds today. The
2026-10-05 nightly's DIVERGE_VALUE and DIVERGE_TRAP classes were fixed in #330
and no longer reproduce on master. Its TIMEOUT class is the oracle (DuckDB)
timing out on a huge `lpad`, not confit. Its 4 metamorphic refuse→serve
findings (qualified struct columns) were not re-checked for this reading.

## Needs the owner

- **OPT_EMULATED, 2 classes** (nightly 2026-10-05, seeds 4291817 and 4313391).
  Confit matches optimizer-off DuckDB, and the optimizer prunes a trapping
  expression. Rule whether these classes are accepted, as DIVERGE_OPT is, or
  must be emulated.
- **finding: c1-depth** (open since 2026-09-26). The written C1 depth passes
  in 67.3 s, but the default is 25. Raise the default, or correct the text.

## Next

- **Wave 1 workers.** T1 (shared subexpressions within a call, the open
  catalog need), T2 (refuse an oversized program before compiling) and T3
  (HUGEINT, fixes requested).
- **Then the ruled query classes, in order.** The account is near its weekly
  usage limit, so the next wave may run inline.
