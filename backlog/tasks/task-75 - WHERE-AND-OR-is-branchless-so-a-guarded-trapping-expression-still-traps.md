---
id: TASK-75
title: >-
  WHERE AND/OR is branchless so a guarded trapping expression still traps
status: To Do
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - lowering
  - crash
dependencies: []
documentation:
  - packages/confit/tests/test_known_divergences.py
type: bug
ordinal: 68000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`fn kleene` is documented as "branchless Kleene AND/OR from flag algebra" and
emits BOTH operands unconditionally. `fn case`, immediately below it, DOES
branch - which is why the same trapping call inside a never-taken CASE arm is
correctly skipped.

```sql
SELECT k FROM __THIS__ WHERE k = 0 AND tree_predict('m', mid, struct_pack(x := x)) > 0
  -> ValueError: predict: no model with id 999      (k = 0 is false for every row)
  duckdb: []

SELECT k FROM __THIS__ WHERE k = 0 AND 9223372036854775807 + k > 0
  duckdb: []      engine: overflow trap
```

The second form makes DuckDB the oracle, so this is not tree-specific: any
trapping expression behind a false guard kills a whole request DuckDB would have
answered with an empty result. Guarding a partial function behind a predicate is
the normal way to write this in SQL.

Found by the 2026-08-08 adversarial sweep (6 finders over distinct surfaces,
then two independent refute-by-default verifiers per finding; 18 raw, 12
verified, 9 confirmed, 2 disputed, 1 refuted).

Pinned xfail-strict, so it cannot silently start or stop failing. Full context
for every finding is in the module docstring of
`packages/confit/tests/test_known_divergences.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A trapping expression behind a guard that excludes every row does not
      trap, matching DuckDB, for AND and for OR
- [ ] #2 Covers a native trap (overflow, div-by-zero) and a `tree_predict`
      unknown-model trap
- [ ] #3 Kleene NULL semantics are unchanged - the branchless form was chosen
      for a reason and three-valued logic must not regress
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The tension is real: branchless flag algebra is what makes Kleene NULL
semantics cheap and correct. Short-circuiting only matters when an operand can
TRAP, so one option is to branch only when the right operand contains a trapping
op (the frontend already computes a `total`-style property for residuals - see
TASK-74 - and could reuse it) and stay branchless otherwise.
<!-- SECTION:NOTES:END -->
