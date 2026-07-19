# CASE WHEN (codegen) Implementation Plan — TASK-30

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the codegen serving engine to evaluate SQL `CASE` (searched + simple forms) with DataFusion-parity, matching the oracle row-for-row.

**Architecture:** Add a `Case` IR node to the codegen front-end (`plan.py`), convert `exp.Case` into it (simple form normalized to `operand = value`), infer its result type via the existing `_common_base` supertype helper, validate its sub-expressions, and emit it as a short-circuiting nested Python conditional in `engine.py`. Prove it against the DataFusion oracle through the existing differential harness, with a temporary `codegen_only` skip so the native backend (which lacks CASE until TASK-27) skips loudly rather than erroring.

**Tech Stack:** Python 3, sqlglot (front-end AST), pyarrow, pydantic, pytest, DataFusion (oracle).

## Global Constraints

- v0, no backward-compat concerns — make breaking changes directly, no shims.
- DataFusion is the oracle (decision-1). Where an engine disagrees with DataFusion, the engine is wrong.
- No new runtime helpers if an existing one suffices: CASE reuses `rt.truthy` and `rt.eq`; no `runtime.py` change.
- All work stays inside `sql_transform/_codegen/` and `tests/` — the fit/state/rewrite path is not touched (CASE already flows through it; verified in the spec).
- Vocabulary: the backends are `native` and `codegen` (never "rust").
- Emission MUST short-circuit — only the taken branch may evaluate — to match the oracle (`CASE WHEN x>0 THEN 1/x ELSE 0 END` must not raise on `x=0`).
- Run tests with `uv run pytest`.

## Reachability note (why there is no SQLTransform-level test here)

`SQLTransform.infer_batch` is hardcoded to the native `InferFn` (`sql_transform/__init__.py:92`); the pluggable-backend design is paused. So after this task, codegen CASE is exercised through the codegen engine directly and the differential harness, **not** through the public `SQLTransform.infer_batch` API. The window-agg-inside-CASE *integration* test (fit → freeze → infer parity) therefore belongs to TASK-27 (native), where `SQLTransform.infer_batch` will actually route CASE through a CASE-capable engine. It is intentionally not in this plan.

## File Structure

- **Modify** `sql_transform/_codegen/plan.py` — add `Case` dataclass; handle `exp.Case` in `_convert_expr`; type it in `infer_type`; recurse it in `_validate_expr`.
- **Modify** `sql_transform/_codegen/engine.py` — emit `cp.Case` in `_emit_expr`.
- **Modify** `tests/differential.py` — add `codegen_only` param to `check()`.
- **Create** `tests/test_diff_case.py` — differential coverage of both CASE forms + edges.
- **Modify** `tests/test_codegen_coverage.py` — move CASE into the committed-surface set.

---

### Task 1: Codegen CASE (searched + simple), proven against the oracle

**Files:**
- Modify: `sql_transform/_codegen/plan.py` (IR node after `Cast` ~line 190; `_convert_expr` after the `exp.Not` branch ~line 315; `infer_type` before the `Func` branch ~line 720; `_validate_expr` before the `Func` branch ~line 686)
- Modify: `sql_transform/_codegen/engine.py` (`_emit_expr` before the final `raise` ~line 98)
- Modify: `tests/differential.py` (`check()` signature ~line 216)
- Create: `tests/test_diff_case.py`
- Modify: `tests/test_codegen_coverage.py` (`_COMMITTED` list ~line 16)

**Interfaces:**
- Consumes: `plan.Literal`, `plan.BinaryOp`, `plan._convert_expr`, `plan._common_base`, `plan.infer_type`, `plan.FieldType`, `runtime.truthy`, `runtime.eq` (all existing).
- Produces: `plan.Case(arms: list[tuple[cond, result]], default, has_else: bool)` IR node; `check(query, tables, expect=None, codegen_only=False)`.

- [ ] **Step 1: Add the `codegen_only` skip to the harness**

In `tests/differential.py`, change the `check` signature and add the guard as its first line:

```python
def check(
    query: str,
    tables: dict[str, Table],
    expect: list[dict] | None = None,
    codegen_only: bool = False,
) -> None:
    """Run `query` through DataFusion (oracle) AND the active backend engine over
    the same typed tables; assert their output rows match (order-insensitive,
    float-tolerant, NULL-aware). If `expect` is given, also assert
    output == expect. `codegen_only=True` skips the native backend for surface
    codegen supports but native does not yet (e.g. CASE, until TASK-27)."""
    if codegen_only and _backend == "native":
        pytest.skip("native does not implement this surface yet (codegen_only)")
    oracle = _run_datafusion(query, tables)
```

(Leave the rest of `check` unchanged.)

- [ ] **Step 2: Run the full suite — flag is inert, nothing regresses**

Run: `uv run pytest -q`
Expected: PASS — `446 passed, 16 skipped, 1 xfailed` (unchanged; the new param has no callers yet).

- [ ] **Step 3: Write the failing differential tests for CASE**

Create `tests/test_diff_case.py`:

```python
"""Differential coverage of SQL CASE (searched + simple forms).

CASE is codegen-only until the native engine gains it (TASK-27), so these run
against the DataFusion oracle on the codegen backend and skip on native via
codegen_only=True. Emission must short-circuit (only the taken branch evaluates),
which the short-circuit test below pins.
"""

from differential import check, rows


def test_case_searched_with_else():
    check(
        "SELECT CASE WHEN x > 0 THEN 'pos' WHEN x < 0 THEN 'neg' ELSE 'zero' END "
        "AS c FROM t",
        {"t": rows({"x": "int"}, [{"x": 5}, {"x": -3}, {"x": 0}])},
        expect=[{"c": "pos"}, {"c": "neg"}, {"c": "zero"}],
        codegen_only=True,
    )


def test_case_searched_no_else_unmatched_is_null():
    check(
        "SELECT CASE WHEN x > 0 THEN 1 END AS c FROM t",
        {"t": rows({"x": "int"}, [{"x": 5}, {"x": -1}])},
        expect=[{"c": 1}, {"c": None}],
        codegen_only=True,
    )


def test_case_simple_form():
    check(
        "SELECT CASE g WHEN 1 THEN 'a' WHEN 2 THEN 'b' ELSE 'z' END AS c FROM t",
        {"t": rows({"g": "int"}, [{"g": 1}, {"g": 2}, {"g": 9}])},
        expect=[{"c": "a"}, {"c": "b"}, {"c": "z"}],
        codegen_only=True,
    )


def test_case_simple_null_operand_falls_through():
    # NULL operand matches no WHEN value (NULL = v is NULL, not true) -> ELSE.
    check(
        "SELECT CASE g WHEN 1 THEN 'a' ELSE 'z' END AS c FROM t",
        {"t": rows({"g": "int?"}, [{"g": None}])},
        expect=[{"c": "z"}],
        codegen_only=True,
    )


def test_case_result_int_float_coerces_to_float():
    # Mixed int/float branches unify to float, matching the oracle (COALESCE rule).
    check(
        "SELECT CASE WHEN x > 0 THEN 1 ELSE 2.5 END AS c FROM t",
        {"t": rows({"x": "int"}, [{"x": 5}, {"x": -1}])},
        expect=[{"c": 1.0}, {"c": 2.5}],
        codegen_only=True,
    )


def test_case_short_circuits_avoiding_error():
    # The non-taken THEN (1 / x with x=0) must NOT be evaluated -- CASE is lazy.
    # If emission were eager this would raise "division by zero".
    check(
        "SELECT CASE WHEN x > 0 THEN 1 / x ELSE 0 END AS c FROM t",
        {"t": rows({"x": "int"}, [{"x": 0}])},
        expect=[{"c": 0}],
        codegen_only=True,
    )


def test_case_null_condition_skips_arm():
    # A NULL WHEN condition doesn't match (three-valued) -> next arm / ELSE.
    check(
        "SELECT CASE WHEN b THEN 'yes' ELSE 'no' END AS c FROM t",
        {"t": rows({"b": "bool?"}, [{"b": None}, {"b": True}])},
        expect=[{"c": "no"}, {"c": "yes"}],
        codegen_only=True,
    )


def test_case_nested():
    check(
        "SELECT CASE WHEN x > 0 THEN "
        "CASE WHEN x > 10 THEN 'big' ELSE 'small' END "
        "ELSE 'neg' END AS c FROM t",
        {"t": rows({"x": "int"}, [{"x": 20}, {"x": 5}, {"x": -1}])},
        expect=[{"c": "big"}, {"c": "small"}, {"c": "neg"}],
        codegen_only=True,
    )
```

- [ ] **Step 4: Run the new tests to verify they fail (RED)**

Run: `uv run pytest tests/test_diff_case.py -q`
Expected: FAIL — on the codegen backend, `CodegenFn` construction raises `ValueError: Unsupported expression: CASE ...` from `_convert_expr`'s generic fallthrough (not caught by `check`'s `UnsupportedInCodegen` handler, so it surfaces as an error). Native runs are skipped by `codegen_only`.

- [ ] **Step 5: Add the `Case` IR node**

In `sql_transform/_codegen/plan.py`, after the `Cast` dataclass (~line 190):

```python
@dataclass
class Case:
    arms: list       # list[tuple[cond, result]]
    default: Any     # result expr; Literal(None) when no ELSE
    has_else: bool   # whether an explicit ELSE was written
```

- [ ] **Step 6: Handle `exp.Case` in `_convert_expr`**

In `plan.py` `_convert_expr`, immediately after the `exp.Not` branch (the `return Not(...)` line ~315), before the `_BINOPS` loop:

```python
    if isinstance(e, exp.Case):
        operand = e.args.get("this")  # simple form: CASE <operand> WHEN ...
        base = _convert_expr(operand) if operand is not None else None
        arms = []
        for if_ in e.args["ifs"]:
            lhs = _convert_expr(if_.this)
            cond = lhs if base is None else BinaryOp("eq", base, lhs)
            arms.append((cond, _convert_expr(if_.args["true"])))
        default_expr = e.args.get("default")
        has_else = default_expr is not None
        default = _convert_expr(default_expr) if has_else else Literal(None)
        return Case(arms, default, has_else)
```

- [ ] **Step 7: Type `Case` in `infer_type`**

In `plan.py` `infer_type`, before the `if isinstance(e, Func):` branch (~line 720):

```python
    if isinstance(e, Case):
        branch_types = [infer_type(r, schemas) for _, r in e.arms]
        branch_types.append(infer_type(e.default, schemas))
        base = _common_base(branch_types)
        # No explicit ELSE means an unmatched row yields NULL, so the result is
        # nullable regardless of the branch types.
        nullable = (not e.has_else) or any(t.nullable for t in branch_types)
        return FieldType(base, nullable)
```

- [ ] **Step 8: Recurse `Case` in `_validate_expr`**

In `plan.py` `_validate_expr`, add a branch before the `elif isinstance(e, Func):` branch (~line 686):

```python
    elif isinstance(e, Case):
        for cond, result in e.arms:
            _validate_expr(cond, resolved, row_schemas, static_schemas, used)
            _validate_expr(result, resolved, row_schemas, static_schemas, used)
        _validate_expr(e.default, resolved, row_schemas, static_schemas, used)
```

- [ ] **Step 9: Emit `cp.Case` as a short-circuiting nested conditional**

In `sql_transform/_codegen/engine.py` `_emit_expr`, before the final `raise UnsupportedInCodegen(...)` (~line 98):

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

This produces `(r1 if rt.truthy(c1) else (r2 if rt.truthy(c2) else default))` — Python evaluates only the taken branch, matching SQL CASE's laziness; `rt.truthy` matches only on `Bool(true)` (three-valued), and the simple form's `rt.eq` returns `None` for a NULL operand/value so that arm doesn't match.

- [ ] **Step 10: Run the CASE tests to verify they pass (GREEN)**

Run: `uv run pytest tests/test_diff_case.py -q`
Expected: PASS — all 8 tests pass on codegen; native runs show as skipped.

- [ ] **Step 11: Move CASE into the committed-surface guard**

In `tests/test_codegen_coverage.py`, add to the `_COMMITTED` list (after the `NULLIF` entry, ~line 49):

```python
    (
        "SELECT CASE WHEN a > 0 THEN 1 ELSE 0 END AS x FROM t",
        {"t": rows({"a": "int"}, [{"a": 1}])},
    ),
    (
        "SELECT CASE a WHEN 1 THEN 'one' ELSE 'other' END AS x FROM t",
        {"t": rows({"a": "int"}, [{"a": 1}])},
    ),
```

- [ ] **Step 12: Run the coverage guard and the full suite**

Run: `uv run pytest tests/test_codegen_coverage.py -q`
Expected: PASS — CASE is now committed surface, never deferred.

Run: `uv run pytest -q`
Expected: PASS — previous total plus 8 new CASE tests running on codegen and skipping on native, plus 2 new committed-surface entries. No regressions, no new failures/xfails.

- [ ] **Step 13: Commit**

```bash
git add sql_transform/_codegen/plan.py sql_transform/_codegen/engine.py \
  tests/differential.py tests/test_diff_case.py tests/test_codegen_coverage.py
git commit -m "feat(codegen): CASE WHEN (searched + simple) — TASK-30

Add a Case IR node, exp.Case conversion (simple form normalized to
operand = value), common-supertype result typing, validation recursion,
and short-circuiting nested-conditional emission. Proven against the
DataFusion oracle; native skips via a temporary codegen_only flag until
TASK-27 lands CASE natively.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Follow-on (not this task)

**TASK-27 (native CASE)** — separate plan, written when Phase 1 lands. It adds `SqlExpr::Case` to `src/expr_build.rs` + eval in `src/expr.rs`, then removes the `codegen_only` flag so every CASE case in `tests/test_diff_case.py` runs on both backends, and adds the window-agg-inside-CASE integration test through `SQLTransform` (reachable once native — the engine `SQLTransform.infer_batch` uses — supports CASE).

## Self-Review

- **Spec coverage:** Both forms (Steps 6 core; searched + simple tests Step 3) ✓. Short-circuit emission (Step 9 + short-circuit test) ✓. `_common_base` result typing + no-ELSE nullability (Step 7 + coercion/no-else tests) ✓. NULL condition / NULL operand semantics (tests) ✓. `codegen_only` harness bridge (Step 1) ✓. Coverage guard update (Step 11) ✓. Window-agg-inside-CASE + native — explicitly deferred to TASK-27 with reason (reachability note) ✓. No fit/state/rewrite changes ✓.
- **Placeholder scan:** none — every step has concrete code and exact run/expected lines.
- **Type consistency:** `Case(arms, default, has_else)` defined in Step 5 is used identically in Steps 6–9; `arms` is `list[(cond, result)]` throughout; `check(..., codegen_only=False)` defined Step 1, used in Step 3.
