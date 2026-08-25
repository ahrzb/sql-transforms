# CASE WHEN support — design

**Author:** Ritchie (codegen dev)
**Date:** 2026-07-19
**Status:** approved for planning

## Summary

Add SQL `CASE` support to the serving engines' `infer` path. Both forms:

- **Searched:** `CASE WHEN cond THEN r [WHEN cond THEN r ...] [ELSE d] END`
- **Simple:** `CASE x WHEN v THEN r [WHEN v THEN r ...] [ELSE d] END`

The feature splits across two engines and ships in two sequenced phases,
tracked as two Backlog tickets:

1. **Phase 1 — codegen** (`sql_transform/_codegen/`). Implemented first.
2. **Phase 2 — native** (`src/expr_build.rs`, `src/expr.rs`). Implemented right
   after Phase 1, by the same dev.

The DataFusion `transform` path — the oracle — already supports `CASE` today, so
no work is needed there (see "What already works").

## Goals

- `infer` (codegen, then native) produces the same rows as DataFusion for `CASE`.
- Both CASE forms, with and without `ELSE`.
- Window aggregates nested inside a CASE (`CASE WHEN AVG(x) OVER () > 0 THEN ...`)
  keep working end to end (they already freeze + rewrite correctly; this is a
  parity assertion, not new code).

## Non-goals

- No changes to the fit / state-freeze / rewrite path — CASE already flows
  through it untouched.
- No exotic cross-type branch coercion beyond what `_common_base` already does
  for `COALESCE` (int/float unify to float; a single shared base stays itself;
  anything else is `OTHER`). Mixed incompatible branches (e.g. `THEN 'a' ELSE 1`)
  inherit COALESCE's existing behavior and are not specially handled.
- No new boolean-condition validation for CASE beyond the codebase's existing
  runtime approach (a non-boolean WHEN condition is treated as non-matching via
  `rt.truthy`, consistent with how `Filter` predicates already behave). Strict
  "WHEN must be boolean" validation, if ever wanted, is a cross-cutting concern
  for `Filter` + `CASE` together, out of scope here.

## What already works (verified)

- `parse_and_validate` (`_sql.py`) accepts a CASE in the projection — it only
  rejects JOIN/WHERE/GROUP BY/HAVING/ORDER BY/LIMIT clauses and enforces
  `FROM __THIS__`. A CASE scalar expression passes.
- `find_window_aggregates` walks `find_all(exp.Window)`, so a window agg inside a
  CASE is discovered regardless of nesting; `rewrite_sql` replaces the Window node
  in place. So state freezing + rewrite already handle window-agg-inside-CASE, and
  codegen then sees a CASE over a plain state-column reference.
- Therefore `SQLTransform.transform` (DataFusion) already evaluates CASE
  correctly — the oracle is ready. Only the `infer` engines have the gap.

## sqlglot AST (verified 2026-07-19)

`sqlglot.parse_one("... CASE ... END ...")` yields an `exp.Case`:

- `.args["this"]` — the operand. `None` for searched CASE; the operand expression
  (e.g. a `Column`) for simple CASE.
- `.args["ifs"]` — a list of `exp.If`, one per WHEN arm. Each `If` has:
  - `.this` — the condition (searched) or the comparison value (simple).
  - `.args["true"]` — the THEN result.
- `.args["default"]` — the ELSE result, or `None` if no ELSE.

Simple form is normalized to searched at convert time: each arm's condition
becomes `operand = value`.

## Phase 1 — codegen (implemented first)

Five touch points, all in `sql_transform/_codegen/`.

### 1. IR node (`plan.py`)

```python
@dataclass
class Case:
    arms: list       # list[tuple[cond, result]]
    default: Any     # result expr; Literal(None) when no ELSE
    has_else: bool   # whether an explicit ELSE was written
```

`default` is always a real expression node (`Literal(None)` when no ELSE), so
emission and validation stay uniform (no ELSE == ELSE NULL). `has_else` is
carried separately so type inference can tell "explicit ELSE" from "implicit
NULL" without identity-comparing a `Literal(None)` sentinel.

### 2. `_convert_expr` (`plan.py`)

Handle `exp.Case` before the generic fallthrough:

- Read `operand = e.args.get("this")`.
- For each `exp.If` in `e.args["ifs"]`:
  - `value_or_cond = _convert_expr(if_.this)`
  - `cond = value_or_cond` if searched (operand is None), else
    `BinaryOp("eq", _convert_expr(operand), value_or_cond)`.
  - `result = _convert_expr(if_.args["true"])`.
- `has_else = e.args.get("default") is not None`; `default =
  _convert_expr(e.args["default"])` if `has_else` else `Literal(None)`.
- Return `Case(arms, default, has_else)`.

NULL semantics fall out of the existing runtime: a NULL operand or NULL WHEN
value makes `rt.eq` return `None`, and `rt.truthy(None)` is `False`, so the arm
doesn't match — matching SQL.

### 3. `infer_type` (`plan.py`)

```python
if isinstance(e, Case):
    branch_types = [infer_type(r, schemas) for _, r in e.arms]
    branch_types.append(infer_type(e.default, schemas))
    base = _common_base(branch_types)
    nullable = (not e.has_else) or any(t.nullable for t in branch_types)
    return FieldType(base, nullable)
```

- Reuses `_common_base` (the exact helper COALESCE uses) — int/float mix → float.
- Nullable if there's no explicit ELSE (`e.has_else` is False, so an unmatched
  row yields NULL) or any branch is nullable.

Conditions are not type-checked at build (consistent with the codebase deferring
predicate type errors to runtime `_tribool`/`truthy`).

### 4. `_validate_expr` (`plan.py`)

Recurse into every condition and result:

```python
elif isinstance(e, Case):
    for cond, result in e.arms:
        _validate_expr(cond, ...)
        _validate_expr(result, ...)
    _validate_expr(e.default, ...)
```

### 5. Emission (`engine.py` `_emit_expr`)

Emit a **nested Python conditional expression** — the load-bearing decision:

```python
if isinstance(e, cp.Case):
    out = _emit_expr(e.default, env)
    for cond, result in reversed(e.arms):
        out = (
            f"({_emit_expr(result, env)} if rt.truthy({_emit_expr(cond, env)}) "
            f"else {out})"
        )
    return out
```

Produces `(r1 if rt.truthy(c1) else (r2 if rt.truthy(c2) else default))`. This:

- **Short-circuits** exactly like SQL CASE — Python evaluates only the taken
  branch, so `CASE WHEN x > 0 THEN 1/x ELSE 0 END` does not raise on `x = 0`,
  matching the oracle. Eager evaluation (e.g. a `rt.case([...])` runtime helper)
  would wrongly raise.
- Reuses `rt.truthy` (matches only on `Bool(true)`, three-valued) and `rt.eq`
  (for the simple form). **No new `runtime.py` helper.**

### Harness bridge (`tests/differential.py`)

During the Phase 1→2 window, codegen has CASE and native does not. `check()`
only skip-catches codegen's `UnsupportedInCodegen`; native would raise a generic
error and a CASE case would **error** on the native backend rather than skip.

Add a `codegen_only: bool = False` parameter to `check()`: when set and the
active backend is `"native"`, `pytest.skip("native does not implement CASE yet")`.
Same loud-skip philosophy the harness already uses for codegen deferrals. This
flag is a **temporary bridge**, removed in Phase 2.

## Phase 2 — native (implemented right after Phase 1)

Add CASE to the native Rust engine so both backends support it and the harness
asymmetry disappears.

- `src/expr_build.rs` `convert_expr`: handle `SqlExpr::Case { operand, conditions,
  results, else_result }` (sqlparser shape — validate exact field names during
  implementation), producing a native `Expr::Case` (new variant) analogous to the
  codegen IR. Simple form normalized to searched via `Expr::BinaryOp { op: Eq }`,
  same as codegen.
- `src/expr.rs`: add the `Expr::Case` variant and its evaluation — left-to-right,
  short-circuiting, matching only on `Value::Bool(true)` (mirror `rt.truthy`),
  falling to `else`/`Value::Null`. Type inference in `src/types.rs` (or wherever
  `infer_type` lives) mirrors the codegen `_common_base`-over-branches logic.
- **Remove the `codegen_only` flag** from `check()` and its CASE call sites, so
  every CASE case runs on both backends through the normal parametrized `check()`.

Native follows the same DataFusion-oracle parity bar. If native disagrees with the
oracle on some CASE edge that codegen matches, that's the standard native-parity
bug process (xfail-strict differential + PM ticket), not an inline fix.

## Testing

New `tests/test_diff_case.py`, through the harness (`codegen_only=True` in Phase 1;
the flag is dropped in Phase 2 so the same cases then run on both backends):

- Searched: multi-arm, with ELSE, without ELSE (unmatched → NULL).
- Simple: `CASE x WHEN v THEN ...`, including a value that matches none → NULL.
- Result coercion: int/float branches unify to float (parity with the oracle).
- NULL condition / NULL operand skips the arm.
- Short-circuit avoids error: `CASE WHEN x > 0 THEN 1/x ELSE 0 END` with `x = 0`
  returns 0, does not raise.
- Window agg inside CASE end to end (fit → freeze → infer parity).
- Nested CASE.

`tests/test_codegen_coverage.py`: update so CASE is no longer in the codegen
"deferred surface" set.

## Task breakdown

One cohesive feature, two engine phases:

- **Ticket 1 (codegen)** — Phase 1 above. ~40 lines across `plan.py`/`engine.py`
  + the harness flag + tests. Implemented first.
- **Ticket 2 (native)** — Phase 2 above. Implemented immediately after, by the
  same dev; drops the harness bridge.

Splitting searched vs simple would be artificial (simple is a few lines of
normalization on top of searched), so it is not split further.
