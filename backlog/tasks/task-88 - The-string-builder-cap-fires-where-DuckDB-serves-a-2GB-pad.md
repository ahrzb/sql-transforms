---
id: TASK-88
title: >-
  The string-builder cap fires where DuckDB serves a 2GB pad
status: Done
assignee: []
created_date: '2026-08-11 16:00'
labels:
  - bug
  - parity
  - fuzz
dependencies: []
documentation:
  - packages/confit/docs/known-limitations.md
type: bug
ordinal: 81000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
```text
SELECT c3 FROM __THIS__ WHERE (c3 <> lpad(c3, 2147483647, CAST(c0 AS VARCHAR)))
  duckdb  serves the row (it builds the 2GB string and compares)
  ours    ValueError: string builder result exceeds ...     [seed 30937]
```

The engine's internal string-builder cap traps on giant pad/repeat results
that DuckDB actually materializes and serves. Both campaigns also produced
the OTHER direction (DuckDB erroring "string builder result exceeds" or
grinding past the 45s timeout on similar shapes), so this is a boundary
mismatch, not a missing feature: the visible limit differs between engines
and between spellings.

4 findings round 2, 3 in round 1 (+2 TIMEOUTs each round are this class's
slow face).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A pad/repeat COUNT literal large enough to exceed the builder budget
      refuses at build by name — never a runtime trap on one engine and a
      served row on the other
- [ ] #2 The data-driven residual (a column-borne count exceeding the budget
      at row time) is documented in known-limitations.md with the cap stated
- [ ] #3 Pins: the refusal, and a large-but-under-budget count serving and
      matching
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refusal at build is the contract-sanctioned direction (DuckDB serves these,
slowly — matching it means gigabyte allocations in a serving engine, which
is the wrong trade for sub-5k-row serving; that judgement is recorded here
rather than silently embedded). The TASK-82 count machinery already parses
the literal spelling, so the budget check is a bound on the same number.

Closed by the 2026-08-13 grooming pass: fixed in 44676c9
(refuse_budget_breaking_count in the frontend, riding the TASK-82 count
machinery from cf6284c). Pinned by
test_a_budget_breaking_literal_count_refuses (lpad/rpad/repeat) and
test_a_large_but_bounded_count_still_serves_and_matches; the data-driven
residual is documented in known-limitations with the cap stated, per AC #2.
<!-- SECTION:NOTES:END -->
