---
id: TASK-121
title: >-
  An ambiguous struct-path column binds instead of refusing
status: To Do
assignee: []
created_date: '2026-08-17 00:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 106000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
```sql
SELECT v AS o FROM __THIS__ LEFT JOIN s0 ON (c0.f0 = s0.c0)
-- __THIS__.c0 is STRUCT(f0 BIGINT); s0.c0 is BIGINT
-- DuckDB: Binder Error: Ambiguous reference to column name "c0"
--         (use: "__THIS__.c0" or "s0.c0")
-- ours:   binds c0 to the struct and serves
```

`c0` names a column in BOTH scopes, so the reference is ambiguous before
anything looks at `.f0`. DuckDB refuses; we resolve it to the row table's
struct and answer.

**The bare-column case is already right.** `SELECT c0 ... ON (c0 = s0.c0)`
with a scalar `c0` in both scopes refuses with "ambiguous column 'c0'"
(measured). What is missing is the same check on the STRUCT-PATH route:
`compound`'s R3 arm (`bare_col_with_fields`) commits to the driving table's
struct column without asking whether the name also binds in a join scope.

Qualifying fixes it on both engines, which is the shape of the fix DuckDB
suggests in its own message:

```sql
ON (__THIS__.c0.f0 = s0.c0)   -- both serve
```

**This is 16 of the 28 findings in the 2026-08-17 4000-seed campaign**, the
largest single class the fuzzer sees by a wide margin, and it is
`DIVERGE_BUILD` — we build what DuckDB refuses, so the risk is a query that
answers here and cannot be run against DuckDB at all. Confirmed
optimizer-independent: identical under `PRAGMA disable_optimizer` and with the
optimizer on, so it is unambiguously ours.

Seeds: 217, 317, 776, 1231, 1577, 1636 and ten more.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 a bare struct-path whose HEAD name also binds in a join scope refuses
      with DuckDB's ambiguity wording, naming both candidate qualifications
- [ ] #2 the qualified spellings still bind — `__THIS__.c0.f0` reaches the
      struct, `s0.c0` reaches the static column (TASK-116 AC #3's rule)
- [ ] #3 the check sits where the bare-column one does, so the two cannot
      drift — one notion of "ambiguous", not two
- [ ] #4 a struct path whose head is unique still binds unqualified
- [ ] #5 the campaign's DIVERGE_BUILD ambiguity class is gone at 4000 seeds
<!-- AC:END -->
