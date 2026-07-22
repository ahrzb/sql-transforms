# Codegen deferred surface — Phase B (containers) Implementation Plan — TASK-29

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the codegen engine handle SQL struct/list surface — typed-column passthrough, construction (`named_struct`/`struct`/`array`), field access, and struct/list comparison — retiring the 9 container-related differential skips (14 → 5, leaving only UNNEST for Phase C).

**Architecture:** Codegen already has the container *type* machinery — `StructBase`/`ListBase` field types, `field_type_to_python` (pydantic submodels / `list[...]`), `_to_native` (unwraps struct/list column values to plain dicts/lists), and `compatible`. Phase B adds the container *expressions*: three new IR nodes (`StructExpr`, `ListExpr`, `FieldAccess`) threaded through `_convert_expr` → `infer_type` → `_validate_expr` → `_emit_expr`, plus type-tagged deep equality in the runtime for container comparison, and removal of the passthrough raise. Container values are plain Python dicts (structs, key order significant) and lists at runtime.

**Tech Stack:** Python 3, sqlglot, pyarrow, pydantic, pytest, DataFusion (oracle), maturin (native build for the differential harness's native backend).

## Global Constraints

- Native is the reference; DataFusion is the oracle. Where an engine disagrees with DataFusion, the engine is wrong.
- Test bar (no test = not done): each shape flips from skipped to green on the codegen backend in the differential suite AND moves from `_DEFERRED` to `_COMMITTED` in `tests/test_codegen_coverage.py`.
- Struct values are Python dicts with **significant key order** (mirrors native `Value::Struct`'s ordered `(name, value)` list); list values are Python lists. Scalar equality inside containers is **type-tagged** (`Int(1) != Float(1.0)`), mirroring native `Value`'s `PartialEq` via the existing `runtime.val_eq`.
- `struct(a, b, ...)` names fields positionally `c0, c1, …` (mirrors native `expr_build.rs`); `named_struct('k', v, …)` uses explicit string-literal keys; `array(...)`/`make_array(...)` build a list.
- Scope: `sql_transform/_codegen/` and `tests/` only. No native-engine or fit/state/rewrite changes.
- Build the native extension with `uv run maturin develop` before running the suite (differential harness's native backend). Run tests with `uv run pytest`. Do NOT run `cargo test` (pyo3 Windows DLL error, unrelated to code).
- **Phase-A carryover (fold into Task 4):** `infer_type`'s `BinaryOp` arm currently types `dpipe` and arithmetic ops as their result base without checking operands aren't containers, so `struct || 'x'` would reach `rt.dpipe`/`display` and render `"<object>"` instead of deferring. Guard container operands there.

## Pre-req: native build + baseline

```
uv run maturin develop
uv run pytest -q            # baseline: 492 passed / 14 skipped
```

## The 9 container skips (Fermi inventory) and where each is retired

- struct/list-typed column passthrough ×3 → **Task 1**
- struct/list construction (named_struct/struct/array) + `named_struct()` ×2 → **Task 2**
- struct field access ×3 → **Task 3**
- struct/list comparison ×1 → **Task 4**

## Deferral sites being implemented (current `plan.py`/`engine.py`)

- `plan.py` `_convert_expr`: `exp.Dot` (~296), 3-part `exp.Column` w/ `db` (~304), `exp.Struct`/`exp.Array` (~351), `_DEFERRED_FUNCS` = `("named_struct", "struct", "unnest", "make_array")` (~382).
- `plan.py` `_validate_expr`: 2-part struct-qualified column (~683).
- `plan.py` `infer_type`: `BinaryOp` container-operand raise (~745).
- `engine.py` `CodegenFn.__init__`: container-output raise (~242).
- `engine.py` `_emit_expr`: fallthrough raise (~107).

## File Structure

- **Modify** `sql_transform/_codegen/plan.py` — new IR dataclasses `StructExpr`/`ListExpr`/`FieldAccess`; `_convert_expr`, `infer_type`, `_validate_expr` arms; adjust `_DEFERRED_FUNCS`.
- **Modify** `sql_transform/_codegen/engine.py` — `_emit_expr` arms; remove the container-output raise.
- **Modify** `sql_transform/_codegen/runtime.py` — `getfield` helper; container-aware `eq`/`neq`.
- **Modify** `tests/test_codegen_coverage.py` — move shapes `_DEFERRED` → `_COMMITTED`.

---

### Task 1: struct/list-typed column passthrough

Retires `SELECT s` / `SELECT l` where the column is a struct/list — the value flows through unchanged. The type/marshalling/output-model machinery already handles this; only the explicit guard blocks it.

**Files:** Modify `sql_transform/_codegen/engine.py`, `tests/test_codegen_coverage.py`

**Interfaces:**
- Consumes: existing `cp.is_container`, `field_type_to_python`, `_to_native`, `Column` emit (`var[name]`).
- Produces: codegen accepts a struct/list column in the projection (output-model field is a pydantic submodel / `list[...]`).

- [ ] **Step 1: Move the two passthrough shapes to `_COMMITTED` (RED)**

In `tests/test_codegen_coverage.py`, move these two entries OUT of `_DEFERRED` and INTO `_COMMITTED`:

```python
    ("SELECT s AS x FROM t", {"t": rows({"s": "struct{x:int}"}, [{"s": {"x": 1}}])}),
    ("SELECT l AS x FROM t", {"t": rows({"l": "list[int]"}, [{"l": [1]}])}),
```

- [ ] **Step 2: Run the coverage guard to verify RED**

Run: `uv run pytest tests/test_codegen_coverage.py -q`
Expected: FAIL — `test_committed_surface_is_never_deferred` fails on the two new entries; `CodegenFn.__init__` raises `UnsupportedInCodegen("column '…' is a struct/list, which codegen does not support yet")`.

- [ ] **Step 3: Remove the container-output raise**

In `sql_transform/_codegen/engine.py` `CodegenFn.__init__` (~line 240), delete the block that rejects container output:

```python
        inferred = [(alias, cp.infer_type(e, schemas)) for alias, e in plan.projection]
        # (DELETE these lines:)
        # for alias, ft in inferred:
        #     if cp.is_container(ft.base):
        #         raise UnsupportedInCodegen(
        #             f"column '{alias}' is a struct/list, which codegen does not "
        #             "support yet"
        #         )
```

Leave the rest of `__init__` unchanged — `field_type_to_python` already builds the submodel/list output type, and `_to_native` already unwraps the incoming value to a dict/list.

- [ ] **Step 4: Verify GREEN + differential parity**

Run: `uv run pytest tests/test_codegen_coverage.py -q`
Expected: PASS — `s`/`l` are committed surface.

Run: `uv run pytest tests/test_diff_types.py -q`
Expected: PASS on both backends with fewer skips — struct/list column round-trips now run on codegen and match the DataFusion oracle. (If a struct/list passthrough case lives in a different `test_diff_*` file, it likewise unskips; the full-suite skip count in Step 5 is the real check.)

- [ ] **Step 5: Full suite + commit**

Run: `uv run pytest -q`
Expected: PASS — skip count drops from 14 by the struct/list passthrough cases; no failures.

```bash
git add sql_transform/_codegen/engine.py tests/test_codegen_coverage.py
git commit -m "feat(codegen): struct/list-typed column passthrough — TASK-29 Phase B"
```

---

### Task 2: struct/list construction (`named_struct`, `struct`, `array`/`make_array`)

**Files:** Modify `sql_transform/_codegen/plan.py`, `sql_transform/_codegen/engine.py`, `tests/test_codegen_coverage.py`

**Interfaces:**
- Consumes: `_convert_expr`, `infer_type`, `FieldType`, `StructBase`, `ListBase`, `_emit_expr`.
- Produces: IR nodes `StructExpr(fields: list[tuple[str, expr]])`, `ListExpr(items: list[expr])`; codegen builds a Python dict / list at runtime.

- [ ] **Step 1: Move construction shapes to `_COMMITTED` (RED)**

In `tests/test_codegen_coverage.py`, add to `_COMMITTED`:

```python
    ("SELECT named_struct('a', x, 'b', y) AS s FROM t",
     {"t": rows({"x": "int", "y": "int"}, [{"x": 1, "y": 2}])}),
    ("SELECT array(x, y) AS l FROM t",
     {"t": rows({"x": "int", "y": "int"}, [{"x": 1, "y": 2}])}),
```

- [ ] **Step 2: Run coverage guard → RED**

Run: `uv run pytest tests/test_codegen_coverage.py -q`
Expected: FAIL — `named_struct()`/`array()` raise `UnsupportedInCodegen`.

- [ ] **Step 3: Add the IR nodes**

In `sql_transform/_codegen/plan.py`, after the `Cast` dataclass, add:

```python
@dataclass
class StructExpr:
    fields: list  # list[tuple[str, expr]] -- key order significant


@dataclass
class ListExpr:
    items: list  # list[expr]
```

- [ ] **Step 4: Convert construction in `_convert_expr`**

In `plan.py` `_convert_expr`, replace the `exp.Struct`/`exp.Array` raise (~351) with construction, and handle the `named_struct`/`struct`/`make_array` anonymous functions (remove them from `_DEFERRED_FUNCS`, which becomes `("unnest",)`):

```python
    if isinstance(e, exp.Struct):
        # sqlglot parses named_struct('k', v, ...) and struct(a, b, ...) authored
        # via STRUCT(...) into exp.Struct with exp.PropertyEQ / bare exprs.
        return _convert_struct(e.expressions)
    if isinstance(e, exp.Array):
        return ListExpr([_convert_expr(a) for a in e.expressions])
```

Add the `named_struct`/`struct`/`make_array` anonymous-function forms to the anonymous branch (they arrive as `exp.Anonymous` with `.expressions`), BEFORE the generic `Func` return:

```python
    if isinstance(e, exp.Anonymous):
        name = e.name.lower()
        if name == "named_struct":
            return _named_struct(e.expressions)
        if name == "struct":
            return StructExpr([(f"c{i}", _convert_expr(a))
                               for i, a in enumerate(e.expressions)])
        if name in ("array", "make_array"):
            return ListExpr([_convert_expr(a) for a in e.expressions])
        if name in _DEFERRED_FUNCS:
            raise UnsupportedInCodegen(f"{name}() is not supported in codegen yet")
        return Func(name, [_convert_expr(a) for a in e.expressions])
```

And change `_DEFERRED_FUNCS = ("named_struct", "struct", "unnest", "make_array")` to:

```python
_DEFERRED_FUNCS = ("unnest",)
```

Add these helpers near `_convert_expr` (module level):

```python
def _named_struct(args: list) -> "StructExpr":
    if len(args) % 2 != 0:
        raise ValueError("named_struct expects (key, value) pairs")
    fields = []
    for i in range(0, len(args), 2):
        key = args[i]
        if not (isinstance(key, exp.Literal) and key.is_string):
            raise ValueError("named_struct field names must be string literals")
        fields.append((key.this, _convert_expr(args[i + 1])))
    return StructExpr(fields)


def _convert_struct(exprs: list) -> "StructExpr":
    # STRUCT(...) authored form: sqlglot wraps each field as exp.PropertyEQ
    # (aliased 'name := value') or a bare expr (positional c0, c1, ...).
    fields = []
    for i, item in enumerate(exprs):
        if isinstance(item, exp.PropertyEQ):
            fields.append((item.this.name, _convert_expr(item.expression)))
        elif isinstance(item, exp.Alias):
            fields.append((item.alias, _convert_expr(item.this)))
        else:
            fields.append((f"c{i}", _convert_expr(item)))
    return StructExpr(fields)
```

- [ ] **Step 5: Type construction in `infer_type`**

In `plan.py` `infer_type`, add arms (before the `Func` arm):

```python
    if isinstance(e, StructExpr):
        fields = tuple((name, infer_type(v, schemas)) for name, v in e.fields)
        return FieldType(StructBase(fields), False)
    if isinstance(e, ListExpr):
        elem_types = [infer_type(x, schemas) for x in e.items]
        # Native unifies: identical element types collapse to that type; an empty
        # or mixed list is unresolvable.
        elem = elem_types[0] if elem_types and all(t == elem_types[0] for t in elem_types) \
            else FieldType(OTHER, True)
        return FieldType(ListBase(elem), False)
```

- [ ] **Step 6: Validate construction in `_validate_expr`**

In `plan.py` `_validate_expr`, add arms so column refs inside constructions are validated (place with the other recursive arms):

```python
    elif isinstance(e, StructExpr):
        for _, v in e.fields:
            _validate_expr(v, resolved, row_schemas, static_schemas, used)
    elif isinstance(e, ListExpr):
        for x in e.items:
            _validate_expr(x, resolved, row_schemas, static_schemas, used)
```

- [ ] **Step 7: Emit construction in `_emit_expr`**

In `sql_transform/_codegen/engine.py` `_emit_expr`, add arms (before the final raise):

```python
    if isinstance(e, cp.StructExpr):
        items = ", ".join(f"{name!r}: {_emit_expr(v, env)}" for name, v in e.fields)
        return f"{{{items}}}"
    if isinstance(e, cp.ListExpr):
        return f"[{', '.join(_emit_expr(x, env) for x in e.items)}]"
```

- [ ] **Step 8: Verify GREEN + parity**

Run: `uv run pytest tests/test_codegen_coverage.py -q`
Expected: PASS.

Run: `uv run pytest tests/test_diff_types.py tests/test_diff_expressions.py -q`
Expected: PASS on both backends; construction cases match the oracle.

- [ ] **Step 9: Full suite + commit**

Run: `uv run pytest -q`
Expected: PASS — skip count lower; no failures.

```bash
git add sql_transform/_codegen/plan.py sql_transform/_codegen/engine.py tests/test_codegen_coverage.py
git commit -m "feat(codegen): struct/list construction (named_struct/struct/array) — TASK-29 Phase B"
```

---

### Task 3: struct field access (`s.field`, nested)

**Files:** Modify `sql_transform/_codegen/plan.py`, `sql_transform/_codegen/engine.py`, `sql_transform/_codegen/runtime.py`, `tests/test_codegen_coverage.py`

**Interfaces:**
- Consumes: `_convert_expr`, `infer_type`, `_validate_expr`, `_emit_expr`, `StructBase`.
- Produces: IR node `FieldAccess(base: expr, field: str)`; runtime `getfield(v, name)`.

sqlglot shapes (verified in Phase A exploration; RE-VERIFY the exact attribute names during implementation with a one-off `sqlglot.parse_one` before writing the branch): `s.f` on a struct column parses as a 2-part `exp.Column` (`.table='s'`, `.name='f'`) — distinguished from a table-qualified column only once schemas resolve; `s.a.b` parses as a 3-part `exp.Column` carrying `db`; and an explicit `exp.Dot` can also occur. All three route to `FieldAccess`.

- [ ] **Step 1: Move field-access shape to `_COMMITTED` (RED)**

In `tests/test_codegen_coverage.py`, add to `_COMMITTED`:

```python
    ("SELECT s.x AS v FROM t", {"t": rows({"s": "struct{x:int}"}, [{"s": {"x": 1}}])}),
```

- [ ] **Step 2: Run coverage guard → RED**

Run: `uv run pytest tests/test_codegen_coverage.py -q`
Expected: FAIL — `s.x` raises `UnsupportedInCodegen("struct field access …")`.

- [ ] **Step 3: Add the `FieldAccess` IR node**

In `plan.py`, after `ListExpr`:

```python
@dataclass
class FieldAccess:
    base: Any
    field: str
```

- [ ] **Step 4: Convert field access in `_convert_expr`**

Replace the `exp.Dot` raise (~296) and the 3-part `exp.Column` raise (~304). `exp.Dot` → `FieldAccess(convert(e.this), e.expression.name)`. For a 3-part column `s.a.b` (has `db`), layer `FieldAccess` over the base column (mirrors native `expr_build.rs`). The 2-part struct-vs-table ambiguity stays deferred at CONVERT time (it's resolved in `_validate_expr` where schemas exist — Step 6):

```python
    if isinstance(e, exp.Dot):
        return FieldAccess(_convert_expr(e.this), e.expression.name)
    if isinstance(e, exp.Column):
        db = e.args.get("db")
        catalog = e.args.get("catalog")
        if catalog:
            raise UnsupportedInCodegen("4-part identifiers are not supported")
        if db:
            # 3-part s.a.b: column `db.this`(=s) then fields a, b. `db`/`this`
            # are the outer/inner names; `name` is the leaf field.
            base = Column(table=None, name=_fold(db))
            return FieldAccess(FieldAccess(base, e.args["this"].name), e.name)
        return Column(table=e.table or None, name=_fold(e.this))
```

(RE-VERIFY the 3-part part names — `db`/`this`/`name` ordering — with a live `sqlglot.parse_one("SELECT s.a.b")` dump before finalizing; adjust the two field names if the dump differs.)

- [ ] **Step 5: Type + validate field access**

In `plan.py` `infer_type`, add an arm:

```python
    if isinstance(e, FieldAccess):
        base_ty = infer_type(e.base, schemas)
        if not isinstance(base_ty.base, StructBase):
            raise ValueError(f"cannot access field {e.field!r} on a non-struct")
        for name, ft in base_ty.base.fields:
            if name == e.field:
                return FieldType(ft.base, ft.nullable or base_ty.nullable)
        raise ValueError(f"unknown struct field: {e.field}")
```

In `_validate_expr`, add:

```python
    elif isinstance(e, FieldAccess):
        _validate_expr(e.base, resolved, row_schemas, static_schemas, used)
```

The 2-part struct-column case: the existing `_validate_expr` code at ~683 currently RAISES `UnsupportedInCodegen` when a table-qualified name turns out to be a container column. Change that raise to REWRITE the node into a `FieldAccess` and re-validate (mirrors native `plan.rs`, which rewrites `Column{table:Some}` → `FieldAccess`). Because `_validate_expr` receives the `Column` inside a parent expression, do the rewrite by returning a replacement the caller substitutes — simplest: detect the container case in `_validate_expr` for a `Column` with a `table` that resolves to a column (not a relation) whose type is a struct, and raise a clear `ValueError` if the struct lacks the field, else record usage of the real container column. RE-READ the ~660–690 block during implementation and mirror native's rewrite precisely; if the in-place rewrite is awkward in Python's structure, handle the 2-part form in `_convert_expr` instead by leaving it as a `Column` and resolving in validation — pick whichever keeps `FieldAccess` the single representation.

- [ ] **Step 6: Add the runtime helper + emit**

In `sql_transform/_codegen/runtime.py`:

```python
def getfield(v: Any, name: str) -> Any:
    """Struct field access: NULL struct -> NULL; a missing field is an error
    (mirrors native FieldAccess in expr.rs)."""
    if v is None:
        return None
    if isinstance(v, dict) and name in v:
        return v[name]
    raise ValueError(f"cannot access field {name!r} on a {type_name(v)} value")
```

In `engine.py` `_emit_expr`:

```python
    if isinstance(e, cp.FieldAccess):
        return f"rt.getfield({_emit_expr(e.base, env)}, {e.field!r})"
```

- [ ] **Step 7: Verify GREEN + parity + full suite + commit**

Run: `uv run pytest tests/test_codegen_coverage.py tests/test_diff_types.py -q`
Expected: PASS on both backends.

Run: `uv run pytest -q`
Expected: PASS — skip count lower.

```bash
git add sql_transform/_codegen/plan.py sql_transform/_codegen/engine.py sql_transform/_codegen/runtime.py tests/test_codegen_coverage.py
git commit -m "feat(codegen): struct field access — TASK-29 Phase B"
```

---

### Task 4: struct/list comparison + container-operand guard

**Files:** Modify `sql_transform/_codegen/plan.py`, `sql_transform/_codegen/runtime.py`, `tests/test_codegen_coverage.py`

**Interfaces:**
- Consumes: `runtime.eq`/`neq`/`val_eq`, `infer_type` `BinaryOp` arm.
- Produces: container-aware `runtime.eq`/`neq`; `infer_type` allows `eq`/`neq` on containers (BOOL) and still defers other ops on containers.

- [ ] **Step 1: Move comparison shape to `_COMMITTED` (RED)**

In `tests/test_codegen_coverage.py`, move `("SELECT (s = s) AS x FROM t", …)` from `_DEFERRED` to `_COMMITTED`.

- [ ] **Step 2: Run coverage guard → RED**

Run: `uv run pytest tests/test_codegen_coverage.py -q`
Expected: FAIL — `infer_type`'s `BinaryOp` arm raises `UnsupportedInCodegen("struct/list comparison …")`.

- [ ] **Step 3: Container-aware equality in the runtime**

In `sql_transform/_codegen/runtime.py`, add a type-tagged deep-equality helper and route `eq`/`neq` through it when either operand is a container (scalars keep the existing `_cmp` path):

```python
def _veq(a: Any, b: Any) -> bool:
    """Type-tagged structural equality mirroring native Value::PartialEq:
    dicts (structs) compare by key order + values, lists elementwise, scalars
    by variant tag + value (Int(1) != Float(1.0))."""
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a.keys()) == list(b.keys()) and all(_veq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_veq(x, y) for x, y in zip(a, b))
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return False
    return val_eq(a, b)
```

Change `eq`/`neq` to short-circuit containers:

```python
def eq(l, r):  # noqa: E741
    if l is None or r is None:
        return None
    if isinstance(l, (dict, list)) or isinstance(r, (dict, list)):
        return _veq(l, r)
    return _cmp(l, r) == 0


def neq(l, r):  # noqa: E741
    if l is None or r is None:
        return None
    if isinstance(l, (dict, list)) or isinstance(r, (dict, list)):
        return not _veq(l, r)
    return _cmp(l, r) != 0
```

- [ ] **Step 4: Type container comparison (and guard other container ops)**

In `plan.py` `infer_type`'s `BinaryOp` arm, replace the blanket container raise so `eq`/`neq` on containers types as BOOL, and everything else on a container (arithmetic, `dpipe`, ordering) still defers — this also closes the Phase-A carryover (`struct || 'x'`):

```python
        if is_container(left.base) or is_container(right.base):
            if e.op in ("eq", "neq"):
                return FieldType(BOOL, nullable)
            raise UnsupportedInCodegen(
                "only equality is supported on struct/list values in codegen yet"
            )
```

(Keep the existing arithmetic arm ABOVE this check so a numeric op on two scalars is unaffected; the container guard must sit before the `dpipe`/BOOL fallthrough so `dpipe` on a container reaches the raise, not `STR`.)

- [ ] **Step 5: Verify GREEN + parity**

Run: `uv run pytest tests/test_codegen_coverage.py -q`
Expected: PASS — `(s = s)` committed; a struct/list in arithmetic or `||` still deferred (add a `_DEFERRED` entry `("SELECT s || 'x' AS r FROM t", {"t": rows({"s": "struct{x:int}"}, [{"s": {"x": 1}}])})` to pin that guard).

Run: `uv run pytest tests/test_diff_types.py tests/test_diff_composition.py -q`
Expected: PASS on both backends; struct/list equality matches the oracle.

- [ ] **Step 6: Full suite + commit**

Run: `uv run pytest -q`
Expected: PASS — remaining skips should be only the 5 UNNEST cases (Phase C). Note the before/after skip counts for the merge report.

```bash
git add sql_transform/_codegen/plan.py sql_transform/_codegen/runtime.py tests/test_codegen_coverage.py
git commit -m "feat(codegen): struct/list equality + container-operand guard — TASK-29 Phase B"
```

---

## Self-Review

- **Spec coverage (Phase B):** passthrough (Task 1) ✓; construction incl. `named_struct` (Task 2) ✓; field access incl. nested/3-part (Task 3) ✓; comparison (Task 4) ✓; Phase-A container-operand carryover folded into Task 4 ✓. UNNEST explicitly left for Phase C ✓. No native/fit-path changes ✓.
- **Placeholder scan:** two steps (Task 3 Steps 4–5) instruct RE-VERIFYING exact sqlglot attribute names with a live parse before writing — this is a genuine validate-don't-assume gate, not a placeholder; the surrounding code is complete. The 2-part-column rewrite (Task 3 Step 5) names the exact native mirror and two acceptable implementations; the implementer picks one and keeps `FieldAccess` the single representation.
- **Type consistency:** `StructExpr(fields)`, `ListExpr(items)`, `FieldAccess(base, field)` are defined once (Tasks 2–3) and referenced identically in `infer_type`/`_validate_expr`/`_emit_expr`; `rt.getfield` name matches between runtime (Task 3 Step 6) and emit; `_veq` name matches between definition and `eq`/`neq` (Task 4 Step 3).
- **Ordering risk flagged:** Task 4 Step 4 explicitly requires the container guard sit after arithmetic and before the `dpipe`/BOOL fallthrough — the one placement bug that would mistype `struct || x`.
