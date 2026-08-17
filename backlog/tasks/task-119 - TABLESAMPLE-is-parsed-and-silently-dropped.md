---
id: TASK-119
title: >-
  TABLESAMPLE is parsed and silently dropped
status: Done
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
test_open_divergences.py explains the fix as destructuring without `..`, so a
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
- [x] #1 TABLESAMPLE refuses by name on every shape, including the default
- [x] #2 every remaining `..` in a sqlparser AST destructure in frontend.rs is
      audited, not just TableFactor::Table - the doctrine is the deliverable,
      not this clause
- [x] #3 whatever else that audit turns up is refused or ticketed, with the
      list recorded
- [x] #4 the xfail-strict pin flips and its reason line is deleted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed as the CLASS, like TASK-69: `plain_table` in frontend.rs is now the
only `TableFactor::Table` pattern in the file, it destructures exhaustively
(name, alias, args, with_hints, version, with_ordinality, partitions,
json_path, sample, index_hints), and all three relation positions -- driving
relation, JOIN, comma-join -- call it. A field added to sqlparser breaks the
build at one site instead of changing answers at three.

The AC #2 audit of every remaining `..` in a sqlparser destructure found one
more real pair: `SqlExpr::Cast`'s `array` and `format`, both silently dropped.
Refused by name. Everything else drops formatting-only fields (`Case`'s
attached tokens, `Substring`'s `special`/`shorthand`) or sits in an arm that
already refuses the whole node.

Pin moved to known_divergences/test_dropped_clauses.py, next to TASK-69's --
same class, same file. Measured refusals now cover TABLESAMPLE at all three
relation positions plus WITH ORDINALITY and the two CAST modifiers, with a
control that the bare spellings still build.
<!-- SECTION:NOTES:END -->
