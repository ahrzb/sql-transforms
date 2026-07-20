# Codegen deferred surface — Phase A (scalar operators) Implementation Plan — TASK-29

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the codegen engine evaluate unary-minus-on-a-non-literal and the `||` operator, retiring their 2 differential-suite skips (16 → 14).

**Architecture:** Both are scalar and reuse codegen's existing `BinaryOp` infra. Unary minus on a non-literal lowers to `0 - x` (mirrors native `expr_build.rs`), reusing `rt.sub`. `||` becomes a `BinaryOp("dpipe", …)` emitting to a new NULL-propagating concat runtime helper (mirrors native `expr.rs` `concat_op`). No new IR relational surface.

**Tech Stack:** Python 3, sqlglot, pyarrow, pydantic, pytest, DataFusion (oracle), maturin (native build for the differential harness's native backend).

## Global Constraints

- Native is the reference; DataFusion is the oracle. Where an engine disagrees with DataFusion, the engine is wrong.
- Test bar (no test = not done): each shape flips from skipped to green on the codegen backend in the differential suite AND moves into `_COMMITTED` in `tests/test_codegen_coverage.py`.
- v0, no backward-compat; direct changes, no shims.
- Scope stays in `sql_transform/_codegen/` and `tests/`. No native-engine or fit/state/rewrite changes.
- The differential harness's native backend imports the built `_interpreter`; ensure it's built with `uv run maturin develop` before running the suite. Run tests with `uv run pytest`.

## Pre-req: native build + baseline

Before Task 1, ensure the native extension is built so the differential harness's `native` backend works, and record the baseline skip count:

```
uv run maturin develop
uv run pytest -q            # record the "N skipped" number — Phase A should drop it by the unary-minus + || codegen skips
```

## File Structure

- **Modify** `sql_transform/_codegen/plan.py` — `_convert_expr` (`exp.Neg` non-literal ~line 317; `exp.DPipe` ~line 326); `infer_type` `BinaryOp` arm (~line 700) for the `dpipe` op.
- **Modify** `sql_transform/_codegen/engine.py` — `_OPS` map (~line 29) add `dpipe`.
- **Modify** `sql_transform/_codegen/runtime.py` — add `dpipe` helper (near `concat`, ~line 392).
- **Modify** `tests/test_codegen_coverage.py` — `_COMMITTED` list (~line 16) + the stale NB comment about unary-minus/`||`.

The existing differential tests `tests/test_diff_rust_bugs.py::test_unary_minus` and `::test_string_concat_operator` already cover both shapes (they skip on codegen today, pass on native); they need no change — implementing the shapes flips them green on codegen.

---

### Task 1: unary minus on a non-literal

**Files:**
- Modify: `sql_transform/_codegen/plan.py` (`_convert_expr` `exp.Neg` branch)
- Modify: `tests/test_codegen_coverage.py` (`_COMMITTED` + NB comment)

**Interfaces:**
- Consumes: existing `BinaryOp`, `Literal`, `_convert_expr`, `infer_type` `sub` rule, `engine._OPS["sub"]`.
- Produces: codegen accepts `-x` for a non-literal `x` (as `BinaryOp("sub", Literal(0), x)`).

- [ ] **Step 1: Commit the shape to the coverage guard (RED)**

In `tests/test_codegen_coverage.py`, add to the `_COMMITTED` list (e.g. after the `NULLIF`/`CAST` block) an entry for unary minus on a non-literal:

```python
    ("SELECT -a AS x FROM t", {"t": rows({"a": "int"}, [{"a": 5}])}),
```

And edit the existing NB comment (which currently says unary minus AND `||` are deliberately absent) so it no longer claims unary minus is deferred — leave only `||` mentioned for now:

```python
    # NB: the `||` operator is deliberately absent here until Phase A Task 2 lands
    # it -- see the codegen deferred-surface spec.
```

- [ ] **Step 2: Run the coverage guard to verify RED**

Run: `uv run pytest tests/test_codegen_coverage.py::test_committed_surface_is_never_deferred -q`
Expected: FAIL — the new `-a` entry raises `UnsupportedInCodegen` ("unary minus on a non-literal is not supported in codegen yet"), which `test_committed_surface_is_never_deferred` reports as a failure.

- [ ] **Step 3: Implement — lower unary minus to `0 - x`**

In `sql_transform/_codegen/plan.py` `_convert_expr`, replace the `exp.Neg` branch's `raise UnsupportedInCodegen(...)` for the non-literal case with a lowering to `BinaryOp("sub", Literal(0), inner)`:

```python
    if isinstance(e, exp.Neg):
        inner = _convert_expr(e.this)
        if isinstance(inner, Literal) and type(inner.value) in (int, float):
            return Literal(-inner.value)
        # Unary minus on a non-literal: lower to 0 - x, mirroring native
        # (expr_build.rs), reusing Sub's int->int / float->float promotion and
        # NULL propagation. infer_type/emit already handle "sub".
        return BinaryOp("sub", Literal(0), inner)
```

(Delete the two-line `raise UnsupportedInCodegen("unary minus on a non-literal ...")` that was there.)

- [ ] **Step 4: Run coverage guard + the differential test to verify GREEN**

Run: `uv run pytest tests/test_codegen_coverage.py -q`
Expected: PASS — `-a` is now committed surface.

Run: `uv run pytest tests/test_diff_rust_bugs.py::test_unary_minus -q`
Expected: PASS on both backends — `-a` (int and float) now evaluates on codegen and matches the DataFusion oracle (native already passed). No skip.

- [ ] **Step 5: Commit**

```bash
git add sql_transform/_codegen/plan.py tests/test_codegen_coverage.py
git commit -m "feat(codegen): unary minus on a non-literal (lower to 0 - x) — TASK-29 Phase A"
```

---

### Task 2: the `||` operator

**Files:**
- Modify: `sql_transform/_codegen/runtime.py` (add `dpipe`)
- Modify: `sql_transform/_codegen/plan.py` (`_convert_expr` `exp.DPipe`; `infer_type` `BinaryOp`)
- Modify: `sql_transform/_codegen/engine.py` (`_OPS`)
- Modify: `tests/test_codegen_coverage.py` (`_COMMITTED` + remove NB comment)

**Interfaces:**
- Consumes: `runtime.display`, existing `BinaryOp` IR, `engine._emit_expr` `BinaryOp` path, `plan.infer_type` `BinaryOp` arm.
- Produces: codegen accepts `a || b` as `BinaryOp("dpipe", a, b)`, typed `STR`, evaluated NULL-propagating.

- [ ] **Step 1: Commit the shape to the coverage guard (RED)**

In `tests/test_codegen_coverage.py`, add to `_COMMITTED`:

```python
    ("SELECT a || '!' AS x FROM t", {"t": rows({"a": "str"}, [{"a": "hi"}])}),
```

And DELETE the NB comment about `||` being deferred (both operators are committed after this task).

- [ ] **Step 2: Run the coverage guard to verify RED**

Run: `uv run pytest tests/test_codegen_coverage.py::test_committed_surface_is_never_deferred -q`
Expected: FAIL — the `a || '!'` entry raises `UnsupportedInCodegen` ("the || operator is not supported in codegen yet").

- [ ] **Step 3: Add the NULL-propagating concat runtime helper**

In `sql_transform/_codegen/runtime.py`, add near `concat` (~line 392):

```python
def dpipe(l: Any, r: Any) -> Any:  # noqa: E741
    """The `||` operator: NULL-propagating string concat -- any NULL operand
    yields NULL (unlike concat(), which skips NULLs). Mirrors expr.rs concat_op:
    non-NULL operands are rendered via display() and joined."""
    if l is None or r is None:
        return None
    return display(l) + display(r)
```

- [ ] **Step 4: Convert `exp.DPipe` and type it**

In `sql_transform/_codegen/plan.py` `_convert_expr`, replace the `exp.DPipe` branch's `raise UnsupportedInCodegen(...)` with a lowering to a `dpipe` `BinaryOp` (binary; `a||b||c` nests as `DPipe(DPipe(a,b), c)`, so `.this`/`.expression` recursion handles it):

```python
    if isinstance(e, exp.DPipe):
        # The `||` operator: NULL-propagating string concat (mirrors native
        # concat_op). Binary -- a||b||c nests as DPipe(DPipe(a,b), c).
        return BinaryOp("dpipe", _convert_expr(e.this), _convert_expr(e.expression))
```

In `sql_transform/_codegen/plan.py` `infer_type`, in the `BinaryOp` arm, add a `dpipe` case that types as `STR` BEFORE the container check and the `BOOL` fallthrough (place it right after the arithmetic-ops `if` block):

```python
        if e.op == "dpipe":
            return FieldType(STR, nullable)
```

- [ ] **Step 5: Emit `dpipe` to the runtime helper**

In `sql_transform/_codegen/engine.py`, add to the `_OPS` map:

```python
    "dpipe": "rt.dpipe",
```

- [ ] **Step 6: Run coverage guard + the differential test to verify GREEN**

Run: `uv run pytest tests/test_codegen_coverage.py -q`
Expected: PASS — `a || '!'` is committed surface.

Run: `uv run pytest tests/test_diff_rust_bugs.py::test_string_concat_operator -q`
Expected: PASS on both backends — `a || '!'` (string), `a || NULL` (→ NULL), and `a || 5` (int coerced via display → "hi5") all evaluate on codegen and match the DataFusion oracle. No skip.

- [ ] **Step 7: Run the full suite and confirm the skip delta**

Run: `uv run pytest -q`
Expected: PASS — no regressions; the "N skipped" count is lower than the pre-req baseline by the unary-minus + `||` codegen skips (Phase A retires 2 shapes). Note the before/after skip numbers for the merge report.

- [ ] **Step 8: Commit**

```bash
git add sql_transform/_codegen/runtime.py sql_transform/_codegen/plan.py \
  sql_transform/_codegen/engine.py tests/test_codegen_coverage.py
git commit -m "feat(codegen): || operator (NULL-propagating concat) — TASK-29 Phase A"
```

---

## Self-Review

- **Spec coverage (Phase A):** unary-minus-on-non-literal (Task 1) ✓; `||` (Task 2) ✓. Both mirror native and are proven by the existing differential tests flipping green on codegen + `_COMMITTED` entries ✓. No native/fit-path changes ✓.
- **Placeholder scan:** none — every step has concrete code and exact run/expected lines.
- **Type consistency:** the `dpipe` op string is identical across `_convert_expr` (Task 2 Step 4), `infer_type` (Step 4), and `engine._OPS` (Step 5); `rt.dpipe` name matches the runtime helper (Step 3).
- **Note:** unary minus needs no `infer_type`/emit change because it reuses the existing `sub` `BinaryOp` path; only the conversion changes.
