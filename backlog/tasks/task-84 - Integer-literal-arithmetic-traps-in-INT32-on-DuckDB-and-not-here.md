---
id: TASK-84
title: >-
  Integer-literal arithmetic traps in INT32 on DuckDB and succeeds in i64 here
status: To Do
assignee: []
created_date: '2026-08-11 13:00'
labels:
  - bug
  - parity
  - fuzz
dependencies:
  - TASK-79
type: bug
ordinal: 77000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
```text
SELECT (-6 * (- 2147483647)) AS s
  duckdb  Out of Range Error: Overflow in multiplication of INT32 (-6 * -2147483647)!
  ours    12884901882
```

DuckDB types both literals INTEGER and multiplies in 32 bits, trapping on
overflow; the engine computes in i64 and serves an answer where the oracle
errors. This is the RUNTIME face of TASK-79 — that ticket is about the Arrow
schema (int32 vs int64, values agree); this one is about trap behaviour, so
values do NOT agree and the divergence is not concat-shaped but answer-shaped.

Found by the fuzzer 2026-08-11: 16 `OutOfRangeException ... INT32/INT64`
DIVERGE_TRAP findings in the 20k campaign (seed 1547 is the sample above).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 An integer expression DuckDB traps on either traps here with the
      matching named error, or refuses at build — no served answer where the
      oracle errors
- [ ] #2 The decision is written into docs/known-limitations.md alongside the
      TASK-79 width note, since they are one design question
- [ ] #3 Executable pin(s) in test_known_divergences.py
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Any honest fix needs DuckDB's inferred integer width in the frontend — the
same "carry the declared width alongside Ty" work TASK-79's option 1 needs,
which is why this depends on it. A cheaper interim: refuse integer arithmetic
whose DuckDB-typed width is narrower than i64 when the bound could overflow
it — but computing "could overflow" is the same width analysis. Decide the
two tickets together.
<!-- SECTION:NOTES:END -->
