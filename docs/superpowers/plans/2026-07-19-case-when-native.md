# CASE WHEN (native) Implementation Plan — TASK-27

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SQL `CASE` (searched + simple forms) to the native Rust `InferFn` engine so it matches DataFusion (the oracle), then retire the temporary `codegen_only` harness bridge so every CASE case runs on BOTH backends.

**Architecture:** Mirror the already-merged codegen CASE semantics in Rust: a new `Expr::Case` variant, `SqlExpr::Case` conversion (simple form normalized to `operand = value`), short-circuiting evaluation (only the taken arm's result is evaluated), common-supertype result typing via the existing `common_base`, and validation/transformer-resolution recursion. Then drop `codegen_only` from the differential harness so the existing `tests/test_diff_case.py` cases run against the oracle on native too, and add the window-agg-inside-CASE integration test that TASK-30 deferred (now reachable, since `SQLTransform.infer_batch` is native-backed).

**Tech Stack:** Rust (pyo3, sqlparser 0.62), maturin (build), Python 3, pytest, DataFusion (oracle). Build the native extension with `uv run maturin develop`; run Rust unit tests with `cargo test`.

## Global Constraints

- v0, no backward-compat — breaking changes directly, no shims.
- DataFusion is the oracle. Where an engine disagrees with DataFusion, the engine is wrong.
- Mirror the codegen CASE semantics exactly (already merged, DataFusion-parity): both forms; simple form normalized to `operand = value`; short-circuit (only the taken arm evaluates); result base = common supertype of THEN branches + ELSE (via existing `common_base`); nullable if there is no explicit ELSE OR any branch is nullable.
- Work stays in `src/` (Rust) and `tests/` (Python). Do NOT touch the codegen engine (`sql_transform/_codegen/`) or the fit/state/rewrite path.
- After any Rust change, rebuild with `uv run maturin develop` before running pytest, or the Python side imports the stale `_interpreter` and tests lie.
- Do NOT commit the built `sql_transform/_interpreter.pyd` / `.pdb` (gitignored — leave them out of every `git add`).
- The `_run_datafusion` oracle helper in `tests/differential.py` (builds the result table from the batches' own physical schema) is correct and must be left as-is.

## Reachability context

`SQLTransform.infer_batch` is hardcoded to the native `InferFn` (`sql_transform/__init__.py:92`). Once native supports CASE, CASE becomes usable through the public API, and the window-agg-inside-CASE composition (fit → freeze → infer) becomes testable through `SQLTransform` — that integration test is Task 2 (it was explicitly deferred from TASK-30 because no CASE-capable engine sat behind `infer_batch` yet).

## Native `Expr` match sites (every exhaustive match over `Expr` needs a `Case` arm)

Adding a variant to `Expr` makes the compiler flag every exhaustive match. The four that must gain a `Case` arm:
- `src/expr.rs` `eval` — exhaustive (compiler-forced). The main evaluation arm.
- `src/types.rs` `infer_type` — exhaustive (compiler-forced).
- `src/plan.rs` `validate_expr` — exhaustive (compiler-forced).
- `src/lib.rs` `resolve_transformers` — **has a catch-all `other => Ok(other)`**, so the compiler will NOT force this one. Omitting a `Case` arm here silently drops transformer resolution inside CASE branches (a `my_transformer(...)` call nested in a THEN/ELSE would never be rewritten to `Expr::Transform`). Add it explicitly.

`column_qualifier` (`plan.rs`) matches only `Expr::Column`+`_ => None`; a CASE is correctly "not a column qualifier," so it needs no change.

## sqlparser 0.62 AST (verified)

`SqlExpr::Case { operand: Option<Box<Expr>>, conditions: Vec<CaseWhen>, else_result: Option<Box<Expr>>, .. }`, where `CaseWhen { condition: Expr, result: Expr }`. Simple CASE sets `operand`; searched CASE leaves it `None`.

## File Structure

- **Modify** `src/expr.rs` — add `Expr::Case { arms, default }` variant; evaluate it in `eval`.
- **Modify** `src/expr_build.rs` — convert `SqlExpr::Case` → `Expr::Case`.
- **Modify** `src/types.rs` — type `Expr::Case` in `infer_type`.
- **Modify** `src/plan.rs` — recurse `Expr::Case` in `validate_expr`.
- **Modify** `src/lib.rs` — recurse `Expr::Case` in `resolve_transformers`.
- **Modify** `tests/differential.py` — remove the `codegen_only` param + guard from `check()`.
- **Modify** `tests/test_diff_case.py` — remove `codegen_only=True` from all call sites; update the module docstring.
- **Create** `tests/test_case_window_integration.py` — window-agg-inside-CASE parity through `SQLTransform`.

---

### Task 1: Native CASE — flip the differential tests onto native (RED), implement in Rust (GREEN)

**Files:**
- Modify: `src/expr.rs` (`Expr` enum ~line 250; `eval` ~line 316)
- Modify: `src/expr_build.rs` (`convert_expr` ~line 10, before the `_ =>` fallthrough)
- Modify: `src/types.rs` (`infer_type` ~line 38)
- Modify: `src/plan.rs` (`validate_expr` ~line 1052)
- Modify: `src/lib.rs` (`resolve_transformers` ~line 125, before `other => Ok(other)`)
- Modify: `tests/differential.py` (`check()` ~line 216)
- Modify: `tests/test_diff_case.py` (docstring + 10 call sites)

**Interfaces:**
- Produces: `Expr::Case { arms: Vec<(Expr, Expr)>, default: Option<Box<Expr>> }` (native IR); `check(query, tables, expect=None)` (no `codegen_only`).
- Consumes: existing `as_tribool`, `common_base`, `convert_expr`, `resolve_transformers`, `validate_expr`, `infer_type`.

- [ ] **Step 1: Remove the `codegen_only` bridge (flip CASE onto native)**

In `tests/differential.py`, revert `check()` to have no `codegen_only` — delete the param and the guard, keep everything else (including the `_run_datafusion` fix):

```python
def check(
    query: str,
    tables: dict[str, Table],
    expect: list[dict] | None = None,
) -> None:
    """Run `query` through DataFusion (oracle) AND the active backend engine over
    the same typed tables; assert their output rows match (order-insensitive,
    float-tolerant, NULL-aware). If `expect` is given, also assert
    output == expect."""
    oracle = _run_datafusion(query, tables)
```

In `tests/test_diff_case.py`: delete the `codegen_only=True,` line from all 10 `check(...)` calls, and replace the module docstring's "codegen-only until native..." framing with:

```python
"""Differential coverage of SQL CASE (searched + simple forms).

Runs against the DataFusion oracle on BOTH backends (native + codegen). Emission
must short-circuit (only the taken branch evaluates), which the short-circuit
test below pins.
"""
```

- [ ] **Step 2: Run the CASE tests to verify they now fail on native (RED)**

Run: `uv run pytest tests/test_diff_case.py -q`
Expected: FAIL — the `codegen` backend still passes, but the `native` backend now runs CASE and raises `Unsupported expression: CASE ...` (from `convert_expr`'s fallthrough in the *current* built `_interpreter`), so every CASE test fails on the `[native]` parametrization. This confirms the tests now exercise native.

- [ ] **Step 3: Add the `Expr::Case` variant**

In `src/expr.rs`, add to the `pub enum Expr` (after the `Transform { ... }` variant, before the closing `}`):

```rust
    /// SQL CASE. `arms` are (condition, result) pairs evaluated left to right;
    /// the first arm whose condition is `Bool(true)` wins. `default` is the ELSE
    /// result, or `None` when there is no ELSE (an unmatched row yields NULL).
    Case {
        arms: Vec<(Expr, Expr)>,
        default: Option<Box<Expr>>,
    },
```

- [ ] **Step 4: Evaluate `Expr::Case` in `eval`**

In `src/expr.rs` `eval`, add an arm (anywhere before the closing `}` of the `match expr`):

```rust
        Expr::Case { arms, default } => {
            // Short-circuit: evaluate conditions left to right, stop at the first
            // Bool(true), and evaluate ONLY that arm's result. A non-matching
            // arm's result is never touched, so CASE WHEN x>0 THEN 1/x ELSE 0 END
            // does not divide by zero at x=0 -- matching the oracle.
            for (cond, result) in arms {
                if let Some(true) = as_tribool(&eval(cond, row)?)? {
                    return eval(result, row);
                }
            }
            match default {
                Some(d) => eval(d, row),
                None => Ok(Value::Null),
            }
        }
```

(`as_tribool` treats `Bool(false)`/`Null` as non-matching and errors on a non-boolean condition, consistent with `Not`/`logic`; DataFusion rejects non-boolean WHEN conditions too, so this is not tested.)

- [ ] **Step 5: Convert `SqlExpr::Case` in `convert_expr`**

In `src/expr_build.rs` `convert_expr`, add an arm before the `_ => Err(...)` fallthrough:

```rust
        SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            let mut arms = Vec::with_capacity(conditions.len());
            for when in conditions {
                // Simple form: `CASE <operand> WHEN <v> ...` normalizes each arm's
                // condition to `operand = v`, matching DataFusion/codegen. A NULL
                // operand/value makes the `=` NULL, so the arm doesn't match.
                let cond = match operand {
                    Some(op) => Expr::BinaryOp {
                        op: BinOp::Eq,
                        left: Box::new(convert_expr(op)?),
                        right: Box::new(convert_expr(&when.condition)?),
                    },
                    None => convert_expr(&when.condition)?,
                };
                arms.push((cond, convert_expr(&when.result)?));
            }
            let default = match else_result {
                Some(e) => Some(Box::new(convert_expr(e)?)),
                None => None,
            };
            Ok(Expr::Case { arms, default })
        }
```

- [ ] **Step 6: Type `Expr::Case` in `infer_type`**

In `src/types.rs` `infer_type`, add an arm (mirrors the merged codegen logic — the ELSE type joins the base pool ONLY when present; nullability keys off whether an explicit ELSE exists):

```rust
        Expr::Case { arms, default } => {
            let mut branch_types: Vec<FieldType> = arms
                .iter()
                .map(|(_, result)| infer_type(result, schemas))
                .collect::<Result<_, _>>()?;
            let has_else = default.is_some();
            if let Some(d) = default {
                branch_types.push(infer_type(d, schemas)?);
            }
            // No explicit ELSE => an unmatched row yields NULL, so nullable
            // regardless of the branch types.
            let nullable = !has_else || branch_types.iter().any(|t| t.nullable);
            Ok(FieldType {
                base: common_base(&branch_types),
                nullable,
            })
        }
```

- [ ] **Step 7: Validate `Expr::Case` in `validate_expr`**

In `src/plan.rs` `validate_expr`, add an arm (recurses into every condition, result, and the default so columns inside CASE branches are checked and collected into `used_columns`):

```rust
        Expr::Case { arms, default } => {
            for (cond, result) in arms {
                validate_expr(
                    cond, resolved, row_schemas, static_schemas, effective_schemas,
                    used_columns,
                )?;
                validate_expr(
                    result, resolved, row_schemas, static_schemas, effective_schemas,
                    used_columns,
                )?;
            }
            if let Some(d) = default {
                validate_expr(
                    d, resolved, row_schemas, static_schemas, effective_schemas,
                    used_columns,
                )?;
            }
            Ok(())
        }
```

- [ ] **Step 8: Recurse `Expr::Case` in `resolve_transformers`**

In `src/lib.rs` `resolve_transformers`, add an arm BEFORE the `other => Ok(other)` catch-all (this one is NOT compiler-forced — without it, a transformer call inside a CASE branch is silently left unresolved):

```rust
        Expr::Case { arms, default } => {
            let mut new_arms = Vec::with_capacity(arms.len());
            for (cond, result) in arms {
                new_arms.push((
                    resolve_transformers(cond, resolved)?,
                    resolve_transformers(result, resolved)?,
                ));
            }
            let default = match default {
                Some(d) => Some(Box::new(resolve_transformers(*d, resolved)?)),
                None => None,
            };
            Ok(Expr::Case { arms: new_arms, default })
        }
```

- [ ] **Step 9: Rebuild the native extension and run cargo tests**

Run: `uv run maturin develop`
Expected: compiles cleanly (no warnings about the new variant — every match now has a `Case` arm).

Run: `cargo test`
Expected: PASS — existing Rust unit tests still green.

- [ ] **Step 10: Run the CASE differential tests to verify GREEN on both backends**

Run: `uv run pytest tests/test_diff_case.py -q`
Expected: PASS — all CASE tests pass on BOTH `native` and `codegen` (no skips now). The two typing regression tests (`test_case_no_else_result_stays_int_in_arithmetic`, `test_case_else_nullable_column_keeps_result_nullable`) now also pass on native, confirming native's `infer_type` matches.

- [ ] **Step 11: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — no regressions. The CASE tests that previously showed as skipped on native now run and pass, so the skip count drops accordingly and the pass count rises; no failures/errors, `1 xfailed` unchanged.

- [ ] **Step 12: Commit (do NOT stage the built .pyd/.pdb)**

```bash
git add src/expr.rs src/expr_build.rs src/types.rs src/plan.rs src/lib.rs \
  tests/differential.py tests/test_diff_case.py
git commit -m "feat(native): CASE WHEN (searched + simple) — TASK-27

Add Expr::Case (arms + optional ELSE), SqlExpr::Case conversion (simple
form normalized to operand = value), short-circuiting evaluation,
common-supertype result typing, and validation/transformer-resolution
recursion. Retire the temporary codegen_only harness bridge so every
CASE case in tests/test_diff_case.py runs against the oracle on both
backends. Mirrors the merged codegen semantics (TASK-30).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Window-agg-inside-CASE integration test (deferred from TASK-30)

**Files:**
- Create: `tests/test_case_window_integration.py`

**Interfaces:**
- Consumes: `SQLTransform` (public API, native-backed `infer_batch`), `differential._rows_equal`.

- [ ] **Step 1: Write the integration test**

Create `tests/test_case_window_integration.py`:

```python
"""CASE composing with window-aggregate freezing (fit -> freeze -> infer).

A window aggregate inside a CASE is frozen at fit and looked up per row at
infer, exactly like any other window agg; CASE just wraps the resulting
state-column reference. This pins that the two compose (transform ==
infer_batch) through SQLTransform's native infer path -- reachable now that
native supports CASE (TASK-27). Deferred here from TASK-30, whose codegen CASE
was not reachable through SQLTransform.infer_batch (which is native-backed).
"""

import pyarrow as pa
from differential import _rows_equal

from sql_transform import SQLTransform


def test_case_over_window_agg_transform_infer_parity():
    train = pa.table({"g": ["a", "a", "b"], "x": [1.0, 3.0, 10.0]})
    sql = (
        "SELECT CASE WHEN x > AVG(x) OVER (PARTITION BY g) THEN 'above' "
        "ELSE 'below' END AS c FROM __THIS__"
    )
    t = SQLTransform(sql).fit(train)
    transform_out = t.transform(train).to_pylist()
    infer_out = [r.model_dump() for r in t.infer_batch(train.to_pylist())]
    assert _rows_equal(transform_out, infer_out), (transform_out, infer_out)
    # g='a': AVG=2 -> x=1 below, x=3 above; g='b': AVG=10 -> x=10 below.
    assert sorted(r["c"] for r in transform_out) == ["above", "below", "below"]
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/test_case_window_integration.py -q`
Expected: PASS — the CASE wraps a frozen `AVG(...) OVER (PARTITION BY g)`; `transform` (DataFusion) and `infer_batch` (native) agree, and the `c` values match the hand-computed expectation. (No Rust change in this task, so no maturin rebuild needed — Task 1 already rebuilt.)

- [ ] **Step 3: Full suite + cargo, then commit**

Run: `uv run pytest -q`
Expected: PASS — one more test than after Task 1, no regressions.

Run: `cargo test`
Expected: PASS.

```bash
git add tests/test_case_window_integration.py
git commit -m "test(native): window-agg-inside-CASE fit/infer parity — TASK-27

Integration test deferred from TASK-30: a window aggregate inside a CASE
freezes at fit and composes with CASE through SQLTransform's native infer
path, now that native supports CASE. transform == infer_batch.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage (Phase 2):** `Expr::Case` variant + short-circuit eval (Steps 3-4) ✓; `SqlExpr::Case` conversion, both forms (Step 5) ✓; `common_base` typing + no-ELSE/ELSE nullability mirroring the fixed codegen logic (Step 6) ✓; validation recursion (Step 7) ✓; transformer-resolution recursion, the non-compiler-forced one (Step 8) ✓; remove `codegen_only` so CASE runs on both backends (Step 1) ✓; window-agg-inside-CASE integration test (Task 2) ✓. No codegen/fit-path changes ✓.
- **Placeholder scan:** none — every step has concrete code and exact run/expected lines.
- **Type consistency:** `Expr::Case { arms: Vec<(Expr, Expr)>, default: Option<Box<Expr>> }` defined in Step 3 is destructured identically in Steps 4/6/7/8; `check(query, tables, expect=None)` (Step 1) matches the existing call sites once `codegen_only=True` is removed.
- **Build discipline:** every Rust change is followed by `uv run maturin develop` before pytest (Step 9); the gitignored `.pyd`/`.pdb` are excluded from `git add` (Step 12).
