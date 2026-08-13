# The Join node — general joins for the dialect logical plan (TASK-104)

**Date:** 2026-08-13. **Status:** approved design (brainstormed with
AmirHossein; scope and node shape are his calls, marked **[D]**).
**Parent:** `2026-08-13-dialect-logical-plan-design.md` (laws L1–L4,
decisions D1–D7). This spec supersedes the parent's `Join` sketch — see
"Spec change" at the end.

## Scope [D]

The general case, in one slice: INNER / LEFT / RIGHT / FULL / CROSS,
`ON <any boolean expr>`, `USING(...)`, `NATURAL`, comma-joins
(`FROM t1, t2`), and chained joins (`a JOIN b JOIN c`). Corpus stake:
~90 of the 678 statements contain a join form.

Refused by name (no corpus family, or DuckDB-exotic): SEMI / ANTI / ASOF /
CROSS APPLY / OUTER APPLY / STRAIGHT_JOIN / GLOBAL / positional joins,
table functions in FROM, derived-table subqueries in FROM (their own
ticket when a corpus family demands them).

## Node [D: ON as expression, not a JoinKey list]

```rust
pub enum JoinKind { Inner, Left, Right, Full, Cross }

Rel::Join {
    left:  Box<Rel>,
    right: Box<Rel>,
    kind:  JoinKind,
    on:    Option<Expr>,   // bound over left ++ right; None iff Cross
}
```

* Output schema is `left ++ right` for **every** kind. Nullability is not
  tracked in v0 (named coarseness inherited from the parent spec) — so
  RIGHT/FULL introducing NULLs changes nothing the plan carries.
* D3 (nothing implicit) holds without a `null_safe` field: `=` and
  `IS NOT DISTINCT FROM` are distinct expression nodes (`bin eq` vs
  `isnotdistinct`), each already spelled per dialect by the printers.
* Verifier rules: `on.is_some()` XOR `kind == Cross`; the predicate types
  BOOLEAN; every Col ordinal/name/type checks against the combined schema
  (existing machinery, no new expression surface).
* Canonical text: `(join inner|left|right|full|cross rel rel expr?)`.

## Frontend binder

* The FROM clause folds **left-nested**: comma-separated relations are
  CROSS joins; each join clause wraps the accumulated left side.
  `FROM a, b JOIN c ON ...` ⇒ `Join(Join(a, b, Cross), c, Inner, on)`.
* Scope is a list of `(qualifier, ordinal range)` sources. Unqualified
  names resolve over all in-scope columns — ambiguity is a `Bind` error
  (DuckDB rejects it too, USING-merged names excepted); qualified names
  resolve within their source's range. A duplicate qualifier is `Bind`
  (DuckDB: "duplicate alias"). ON for a chained join sees every source
  joined so far (DuckDB semantics).
* **USING is a binder concern, not a plan node.** `USING(c)` desugars to
  plain `=` on the two `c`s; name resolution follows DuckDB:
  * merged column `c` resolves to: INNER/LEFT → left's `c`; RIGHT →
    right's `c`; FULL → `CASE WHEN l.c IS NOT NULL THEN l.c ELSE r.c END`;
  * `*` expansion dedups the merged column (DuckDB's observed order —
    live-verified by the L2 gate, statement by statement);
  * qualified `t2.c` still reaches the underlying column.
* NATURAL = USING(common column names, left order); no common columns
  follows DuckDB's behavior, measured before shipped.
* WHERE and SELECT items bind over the full scope; lateral-alias and
  auto-naming rules are unchanged.

## Printers [D: approach A — qualified ordinal refs]

Joined tables routinely share column names (self-joins are the corpus's
main join shape), and the current printer addresses columns by NAME,
refusing duplicates. The shared fix in `printer.rs`:

* The expression-printer input becomes a per-ordinal table of
  pre-rendered SQL refs (`ColRef`), built once per query shape.
* Every join side gets a deterministic alias:

  ```sql
  SELECT "j0"."id" AS "id", "j1"."id" AS "id"
  FROM "test" AS "j0" INNER JOIN "test" AS "j1" ON ("j0"."id" = "j1"."id")
  ```

  so cross-side duplicates are simply unambiguous. Within one side, a
  duplicate name (a dup-name Project as a join input) still refuses at
  reference time, same words as today.
* Name-addressed pass-throughs (Filter over a subquery) keep the existing
  dup-name refusal; nothing weakens.
* The frontend∘printer fixpoint survives: reparsing the printed SQL binds
  the aliases back to the identical ordinals, so `parse(print(p)) == p`
  on SQL-derived plans stays exact.
* ON prints through each dialect's existing expression printer —
  per-dialect null-safe spelling and float forcings come for free. Join
  keywords are identical across DuckDB/Spark/BigQuery. USING never
  prints; L2 (bit-exact oracle invisibility) is the proof the desugar
  preserves semantics.

Rejected alternatives: positional-rename subqueries (breaks the SQL
fixpoint — the renames bind as extra Projects); keeping name-addressing
and refusing dup-name joins (refuses the corpus's main join shape).

## Gates

* Rust unit tests: join round-trip through plan text, SQL fixpoint,
  refusals (exotic kinds, ambiguity as Bind), USING/NATURAL/star cases,
  FULL-USING merge expr, chained and comma joins.
* L2 corpus gate: floor 235 → the new measured count (recorded in the
  commit message). Three-outcome accounting: no statement changes class
  downward, zero wrong answers.
* L3 Spark gate: floor 213 → new measured count.
* Any corpus statement whose reprint diverges becomes an xfail-strict pin
  and a finding — never an inline guess (the engine-bug process).

## Spec change to the parent

The parent spec's node sketch `Join(input, input, kind, [JoinKey])` with
`JoinKey.null_safe` is replaced by `on: Option<Expr>` [D]; its
mandatory-field table row for join equality now reads "explicit via
expression node kind (Eq vs IsDistinct)". The parent file is edited in
the same commit as this spec.
