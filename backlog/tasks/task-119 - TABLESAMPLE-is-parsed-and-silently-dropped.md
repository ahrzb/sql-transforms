---
id: TASK-119
title: >-
  TABLESAMPLE is parsed and silently dropped
status: To Do
assignee: []
created_date: '2026-08-16 02:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 104000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
```sql
SELECT a FROM __THIS__ TABLESAMPLE 3 ROWS   -- 20 input rows
-- DuckDB: 3 rows
-- ours:   20 rows, backend=cranelift, no refusal
```

And it builds under `shape="map"`, where we certify one output row per input
row - so the shape contract is satisfied by the very fact that the clause was
ignored.

TASK-69 closed the silently-dropped-clause class and its block in
test_known_divergences.py explains the fix as destructuring without `..`, so a
new clause breaks the build instead of being ignored. That holds for `Query`
and `Select`. It does not hold at `TableFactor::Table { name, alias, .. }`
(frontend.rs:565, 665, 862), which swallows `sample`.

So this is the same class the doctrine was built for, leaking through the one
struct the doctrine was not applied to. The fix is presumably to destructure
`TableFactor::Table` exhaustively too and refuse `sample` by name - but check
the other `..` patterns in the same sweep rather than fixing this one field.

Found 2026-08-16 by an adversarial classification pass over the divergence
file; reproduced by hand before filing. Pinned xfail-strict as
`test_tablesample_is_refused_not_dropped`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 TABLESAMPLE refuses by name on every shape, including the default
- [ ] #2 every remaining `..` in a sqlparser AST destructure in frontend.rs is
      audited, not just TableFactor::Table - the doctrine is the deliverable,
      not this clause
- [ ] #3 whatever else that audit turns up is refused or ticketed, with the
      list recorded
- [ ] #4 the xfail-strict pin flips and its reason line is deleted
<!-- AC:END -->
