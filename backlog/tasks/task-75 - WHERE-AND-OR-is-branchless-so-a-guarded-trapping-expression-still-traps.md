---
id: TASK-75
title: >-
  WHERE AND/OR is branchless so a guarded trapping expression still traps
status: Done
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - lowering
  - crash
dependencies: []
documentation:
  - packages/confit/tests/known_divergences/test_short_circuit.py
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
`packages/confit/tests/known_divergences/test_short_circuit.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A trapping expression behind a guard that excludes every row does not
      trap, matching DuckDB, for AND and for OR
- [x] #2 Covers a native trap (overflow, div-by-zero) and a `tree_predict`
      unknown-model trap
- [x] #3 Kleene NULL semantics are unchanged - the branchless form was chosen
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

## Resolution (2026-08-08)

Both forms kept, selected per call site. `FB::kleene` stays branchless — the
flag algebra that makes three-valued NULL semantics free — whenever the right
operand cannot trap, which covers ordinary predicates like `a > 1 AND b < 2`
entirely. When it can trap, `FB::kleene_shortcut` branches:

- a definite FALSE decides an AND, a definite TRUE decides an OR: the result
  is that constant and the right operand is never evaluated;
- a NULL left decides nothing, so the right operand still runs, and still
  traps, exactly as DuckDB does;
- otherwise the ordinary flag algebra applies, unchanged and shared.

"Can this trap" is `plan::may_trap`, the same predicate the JOIN ON residual
rule uses (TASK-74). One definition; they cannot drift.

The left lane rides the branch as a live-stack entry, because values never
cross blocks except as branch args — the same discipline TASK-66 and TASK-73
were about.

### Mutation-checked, not just green

- Forcing the branchless path (`if false && may_trap(b)`) fails 9 tests across
  both backends.
- Letting a NULL left decide (dropping the flag guard on `decides`) turns
  `NULL AND TRUE` into FALSE and fails the three-valued-logic test — so AC #3
  is guarded on the new path specifically, not only on the old one.

### A bug this fix introduced, and what caught it

The first cut of `kleene_shortcut` always carried a flag param with the
result. That passed the whole suite in RELEASE and panicked in DEBUG on
`BETWEEN int x double` and `IN` with NaN:

```text
lower.rs:198: non-nullable column with a flag lane
```

The null-lane discipline says an SExpr with `nullable == false` lowers to a
bare payload register with no flag anywhere, and `emit_stores` asserts it —
with `debug_assert!`, which release compiles out. So a release-only test cycle
could not see it. The branch now carries a flag param only when the result is
nullable, exactly as `FB::case` decides from its own `res_nullable`.

**Run the Python suite against a debug build as well as a release one.** The
invariants that hold this lowering together are debug assertions; a green
release bar says nothing about them.
