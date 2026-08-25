# Codegen deferred surface — Phase C (UNNEST) Implementation Plan — TASK-29

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the codegen engine handle `unnest()` — row multiplication for a list argument, per-field column expansion for a struct argument — retiring the last 5 differential skips and closing TASK-29.

**Architecture:** `unnest()` is the only SQL surface in this ticket that is *relational* rather than scalar, so it cannot be handled in `_emit_expr` like every prior phase. Both forms are recognized at **validation time** (`validate_columns` in `plan.py`), which is the first point where the argument's type is known and therefore the first point where list-vs-struct can be told apart. A list argument rewrites the plan: the input rel is wrapped in a new `Unnest` node and the projection item becomes a plain column reference to the element the node binds. A struct argument rewrites the projection: one `FieldAccess` column per struct field. Both mirror native (`src/plan.rs`) exactly.

**Tech Stack:** Python 3.14, sqlglot, pyarrow, pydantic, pytest, DataFusion (oracle), maturin (native build for the differential harness's native backend).

## Global Constraints

- Native is the reference; DataFusion is the oracle. Where an engine disagrees with DataFusion, the engine is wrong.
- Test bar (no test = not done): each shape flips from skipped to green on the codegen backend in the differential suite AND moves from `_DEFERRED` to `_COMMITTED` in `tests/test_codegen_coverage.py`.
- Scope: `sql_transform/_codegen/` and `tests/` only. No native-engine or fit/state/rewrite changes. A native parity bug found along the way is an xfail-strict test + a note for the PM, never an inline fix.
- v0, no backward compat. Direct changes, no shims.
- Build the native extension before running the suite: `uv run maturin develop`. Run tests with `uv run pytest`. Do NOT run `cargo test` (pyo3 Windows DLL error, unrelated to code).
- **Baseline to verify before starting:** `uv run pytest -q` → `542 passed, 5 skipped, 5 xfailed`. The 5 skips are the 5 unnest cases; every step below reports against that number.

## Measured facts (verified 2026-07-24 — do not re-derive, do not assume otherwise)

1. **sqlglot parses `unnest(x)` as its own `exp.Unnest` class**, NOT `exp.Anonymous`, with the arguments in `.expressions` (a list). The `_DEFERRED_FUNCS` name-matching branch in `_convert_expr` never sees it. Dump used:
   ```
   SELECT unnest(l) AS x FROM t  ->  Alias(this=Unnest(expressions=[Column(this=Identifier(this=l))]), alias=Identifier(this=x))
   SELECT unnest(s) FROM t       ->  Unnest(expressions=[Column(this=Identifier(this=s))])
   ```
2. **A bare `unnest(...)` with no `AS` arrives in the SELECT list as a top-level `exp.Unnest`**, so it never reaches `_build_projection`'s `exp.Alias` branch. `_build_projection` already has a dedicated `exp.Unnest` branch for this (it currently discards the item).
3. **`_validate_rel` has no fallthrough arm.** A rel node with no `isinstance` arm is silently skipped — and so is its entire subtree, meaning a `Filter` or `Join` underneath would never be validated. The new node MUST get an arm.
4. **`referenced_tables` has the same shape** and needs the new node in its passthrough tuple, or `infer()` raises "Unknown table in FROM clause".
5. **`optimize()` runs BEFORE `validate_columns`** (see `CodegenFn.__init__`), so `_optimize_rel` never sees an `Unnest` node. No arm is needed there.
6. **`_emit_expr` for a `Column` emits `env[table][name]`** — a subscript into the dict held by the local bound for that table. So if the Unnest loop binds each element as a **one-key dict** under a synthetic qualifier, `Column(table=UNNEST_KEY, name=alias)` compiles with zero changes to `_emit_expr`.
7. **`tests/test_diff_types.py::test_multi_unnest_rejected` does NOT cover codegen.** It builds a native `InferFn` directly rather than going through `check()`, so it runs the native engine under both backend parameters. Codegen's one-unnest-per-query guard needs its own unit test — see Task 1, Step 9.

## File Structure

- **Modify** `sql_transform/_codegen/plan.py` — `UNNEST_KEY` constant; `Unnest` rel dataclass; `_convert_expr`'s `exp.Unnest` arm; `_build_projection`'s `exp.Unnest` arm; the expansion dispatch inside `validate_columns`; `_validate_rel` and `referenced_tables` arms; `_unnest_arg` and `_unnest_display_name` helpers.
- **Modify** `sql_transform/_codegen/engine.py` — `_emit_rel` gains an `Unnest` arm.
- **Modify** `sql_transform/_codegen/runtime.py` — `unnest_rows` helper.
- **Modify** `sql_transform/_codegen/plan_test.py` — replace the stale `test_build_plan_defers_unnest`.
- **Modify** `sql_transform/_codegen/engine_test.py` — codegen-only guards the differential harness cannot reach.
- **Modify** `tests/test_codegen_coverage.py` — move unnest shapes `_DEFERRED` → `_COMMITTED`.

---

### Task 1: `unnest(list)` — row multiplication (retires 3 skips)

Retires `test_unnest_list_expands_rows`, `test_unnest_list_all_dropped_yields_no_rows`, `test_unnest_list_preserves_other_columns`. Also lands the shared plumbing (conversion, dispatch) that Task 2 extends, and keeps `unnest(struct)` deferring cleanly so the suite stays green between the two tasks.

**Files:**
- Modify: `sql_transform/_codegen/plan.py`, `sql_transform/_codegen/engine.py`, `sql_transform/_codegen/runtime.py`, `sql_transform/_codegen/plan_test.py`, `sql_transform/_codegen/engine_test.py`, `tests/test_codegen_coverage.py`

**Interfaces:**
- Consumes: `Func`, `Column`, `FieldType`, `ListBase`, `StructBase`, `infer_type`, `_validate_expr`, `_emit_rel`, `_emit_expr`, `rt.type_name`.
- Produces:
  - `plan.UNNEST_KEY: str` — the synthetic qualifier (`"\0unnest"`).
  - `plan.Unnest(input, list_expr, output_col: str)` — rel node.
  - `plan._unnest_arg(e) -> Any | None` — the single argument of an `unnest(...)` item, else `None`.
  - `runtime.unnest_rows(v, name: str) -> list[dict]` — one `{name: element}` binding per list element.

- [ ] **Step 1: Move the list-unnest shape to `_COMMITTED` (RED)**

In `tests/test_codegen_coverage.py`, DELETE this entry from `_DEFERRED`:

```python
    ("SELECT unnest(l) AS x FROM t", {"t": rows({"l": "list[int]"}, [{"l": [1]}])}),
```

and add it to the end of `_COMMITTED` (just after the `(s = s)` entry), with its comment:

```python
    # unnest(list) multiplies rows: the projection item becomes a reference to
    # the column the Unnest rel node binds, one output row per element.
    ("SELECT unnest(l) AS x FROM t", {"t": rows({"l": "list[int]"}, [{"l": [1]}])}),
```

Leave the `s || 'x'` entry in `_DEFERRED` — it is a deliberate guard test, not an unimplemented feature.

- [ ] **Step 2: Run the coverage guard to verify RED**

Run: `uv run pytest tests/test_codegen_coverage.py -q`

Expected: FAIL — `test_committed_surface_is_never_deferred[SELECT unnest(l) AS x FROM t-...]` fails with `committed surface must not be deferred: 'SELECT unnest(l) AS x FROM t' raised unnest() is not supported in codegen yet`.

- [ ] **Step 3: Keep `unnest` as a `Func` instead of raising**

In `sql_transform/_codegen/plan.py`, REPLACE the `exp.Unnest` arm in `_convert_expr` (currently a raise) with:

```python
    if isinstance(e, exp.Unnest):
        # Measured: UNNEST(...) parses as its own exp.Unnest class (args in
        # .expressions), not exp.Anonymous. Converted to a plain Func node so
        # validate_columns can recognize and REPLACE it -- a surviving
        # unnest() Func never reaches infer_type or emit.
        if len(e.expressions) != 1:
            raise ValueError("unnest() takes exactly one argument")
        return Func("unnest", [_convert_expr(e.expressions[0])])
```

In the same file, REPLACE the body of `_build_projection`'s `exp.Unnest` branch (currently `_convert_expr(item)`, which discards the item) with:

```python
        elif isinstance(item, exp.Unnest):
            # A bare `unnest(...)` (no AS) never reaches the exp.Alias branch
            # above. Placeholder name, mirroring native projection_name: a
            # struct unnest discards it (its columns are named per field) and a
            # list unnest only keeps it when the user wrote no alias.
            out.append(("unnest", _convert_expr(item)))
```

`unnest` was `_DEFERRED_FUNCS`'s only member, and per measured fact #1 that branch never fired for it, so the constant is now dead. DELETE both — the definition:

```python
_DEFERRED_FUNCS = ("unnest",)
```

and its only use, inside `_convert_expr`'s `exp.Anonymous` arm:

```python
        if name in _DEFERRED_FUNCS:
            raise UnsupportedInCodegen(f"{name}() is not supported in codegen yet")
```

The `exp.Anonymous` arm keeps its `named_struct` and `make_array` branches and its final `return Func(name, [...])`, which is now what an unknown function name reaches. An unknown function is still rejected — `_emit_expr` raises `ValueError(f"Unknown function: {name}")` for any name not in `_BUILTINS` — so nothing silently compiles.

- [ ] **Step 4: Add the `UNNEST_KEY` constant and the `Unnest` rel node**

In `plan.py`, immediately BEFORE the `Plan` dataclass (i.e. after `LookupJoin`), add:

```python
@dataclass
class Unnest:
    """Row-multiplying `unnest(list)`: evaluates `list_expr` per input row and
    emits one row per element, bound under UNNEST_KEY as `output_col`."""

    input: Any
    list_expr: Any
    output_col: str


# Synthetic qualifier the Unnest node binds its emitted element column under.
# NUL keeps it unspellable as a real table alias (mirrors native UNNEST_KEY).
UNNEST_KEY = "\0unnest"
```

- [ ] **Step 5: Add the `_unnest_arg` helper**

In `plan.py`, immediately BEFORE `def _resolve_tables(`, add:

```python
def _unnest_arg(e: Any) -> Any:
    """The single argument of an `unnest(...)` projection item, else None."""
    if isinstance(e, Func) and e.name == "unnest":
        return e.args[0]
    return None
```

- [ ] **Step 6: Expand the projection in `validate_columns`**

In `plan.py`'s `validate_columns`, REPLACE this block:

```python
    used: dict = {}
    plan.projection[:] = [
        (alias, _validate_expr(e, resolved, row_schemas, static_schemas, used))
        for alias, e in plan.projection
    ]
    _validate_rel(plan.input, resolved, row_schemas, static_schemas, used)
```

with:

```python
    used: dict = {}
    expanded: list = []
    unnest_seen = False
    for alias, e in plan.projection:
        e = _validate_expr(e, resolved, row_schemas, static_schemas, used)
        arg = _unnest_arg(e)
        if arg is None:
            expanded.append((alias, e))
            continue
        arg_type = infer_type(arg, effective_schemas)
        if isinstance(arg_type.base, ListBase):
            # unnest(list) MULTIPLIES rows: wrap the input rel and leave a plain
            # reference to the column the Unnest node binds.
            if unnest_seen:
                raise ValueError("Only one unnest(list) per query is supported")
            unnest_seen = True
            plan.input = Unnest(plan.input, arg, alias)
            effective_schemas.setdefault(UNNEST_KEY, {})[alias] = arg_type.base.elem
            expanded.append((alias, Column(table=UNNEST_KEY, name=alias)))
        elif isinstance(arg_type.base, StructBase):
            raise UnsupportedInCodegen(
                "unnest() on a struct is not supported in codegen yet"
            )
        else:
            raise ValueError("unnest() expects a struct or list argument")
    plan.projection[:] = expanded
    _validate_rel(plan.input, resolved, row_schemas, static_schemas, used)
```

Note the two different exception types, and keep them straight: the struct case raises `UnsupportedInCodegen` so it stays a *skip* until Task 2 replaces that branch; a scalar argument raises `ValueError` because it is a genuine user error, not a deferral.

- [ ] **Step 7: Teach the two rel walkers about `Unnest`**

Still in `plan.py`. In `_validate_rel`, REPLACE:

```python
    elif isinstance(node, SubqueryAlias):
        _validate_rel(node.input, resolved, row_schemas, static_schemas, used)
```

with:

```python
    elif isinstance(node, (SubqueryAlias, Unnest)):
        # Unnest.list_expr came from the projection and was validated there.
        _validate_rel(node.input, resolved, row_schemas, static_schemas, used)
```

In `referenced_tables`, REPLACE:

```python
    if isinstance(node, (SubqueryAlias, Filter, LookupJoin)):
```

with:

```python
    if isinstance(node, (SubqueryAlias, Filter, LookupJoin, Unnest)):
```

Both are required (measured facts #3 and #4). Skipping the first silently stops validating everything below the Unnest; skipping the second makes `infer()` raise "Unknown table in FROM clause".

- [ ] **Step 8: Add the runtime helper and the emit arm**

In `sql_transform/_codegen/runtime.py`, add immediately BEFORE `def _fmt_float(`:

```python
def unnest_rows(v: Any, name: str) -> list:
    """One binding per list element for the Unnest loop. An empty list yields
    zero rows, and so does a NULL list -- the input row disappears rather than
    producing a NULL (matches DataFusion / native execute_rel)."""
    if v is None:
        return []
    if not isinstance(v, list):
        raise ValueError(f"unnest() expected a list, got a {type_name(v)} value")
    return [{name: item} for item in v]
```

In `sql_transform/_codegen/engine.py`'s `_emit_rel`, add this arm immediately BEFORE `elif isinstance(node, cp.LookupJoin):`:

```python
    elif isinstance(node, cp.Unnest):

        def unnested(inner: dict, i: int) -> None:
            v = em.var("_u")
            em.line(
                i,
                f"for {v} in rt.unnest_rows("
                f"{_emit_expr(node.list_expr, inner)}, {node.output_col!r}):",
            )
            body({**inner, cp.UNNEST_KEY: v}, i + 1)

        _emit_rel(node.input, env, ind, em, unnested)
```

Binding each element as a one-key dict is what makes `Column(UNNEST_KEY, alias)` emit correctly with no change to `_emit_expr` (measured fact #6).

- [ ] **Step 9: Replace the stale unit test and add the codegen-only guards**

`sql_transform/_codegen/plan_test.py` currently asserts that building an unnest plan raises. That is now false. REPLACE:

```python
def test_build_plan_defers_unnest():
    with pytest.raises(cp.UnsupportedInCodegen):
        cp.build_plan("SELECT unnest(a) AS x FROM t")
```

with:

```python
def test_build_plan_keeps_unnest_as_a_func():
    # TASK-29 Phase C: unnest survives conversion as a plain Func so
    # validate_columns can expand it once the argument's type is known
    # (row-multiplying for a list, per-field columns for a struct).
    plan = cp.build_plan("SELECT unnest(a) AS x FROM t")
    assert plan.projection == [("x", cp.Func("unnest", [cp.Column(None, "a")]))]
    # A bare unnest (no AS) still gets the placeholder projection name.
    assert cp.build_plan("SELECT unnest(a) FROM t").projection[0][0] == "unnest"
```

In `sql_transform/_codegen/engine_test.py`, add immediately BEFORE `def test_generated_source_is_available_for_debugging():`:

```python
class TwoLists(BaseModel):
    a: list[int]
    b: list[int]


def test_multi_unnest_rejected():
    # Two unnest(list) calls is a cross-product cardinality change we don't
    # support (mirrors native). The differential harness can't cover this --
    # its multi-unnest test builds a native InferFn directly on both backends.
    with pytest.raises(ValueError, match="Only one unnest"):
        CodegenFn("SELECT unnest(a) AS x, unnest(b) AS y FROM t", {"t": TwoLists}, {})


def test_unnest_of_a_scalar_is_rejected():
    with pytest.raises(ValueError, match="struct or list"):
        CodegenFn("SELECT unnest(a) AS x FROM t", {"t": Row}, {})
```

`Row` is the module-level model already defined at the top of `engine_test.py` (`a: int`), so `unnest(a)` there is an unnest of a scalar.

- [ ] **Step 10: Verify GREEN on the unit tests and the coverage guard**

Run: `uv run pytest tests/test_codegen_coverage.py sql_transform/_codegen/ -q`

Expected: PASS, no failures. The coverage guard accepts `unnest(l)` as committed surface, and the two new `engine_test.py` guards pass.

- [ ] **Step 11: Verify the 3 list skips flipped to green against the oracle**

Run: `uv run pytest tests/test_diff_types.py -v -k unnest`

Expected: `test_unnest_list_expands_rows[codegen]`, `test_unnest_list_all_dropped_yields_no_rows[codegen]` and `test_unnest_list_preserves_other_columns[codegen]` all **PASSED** (they were SKIPPED). The two `test_unnest_struct_*[codegen]` cases are still SKIPPED — that is Task 2. All `[native]` cases stay PASSED.

Reading the `[codegen]` result as proof matters here: `tests/conftest.py` parametrizes the harness modules over both backends, and a green `[codegen]` ID is only meaningful because the `_backend` fixture is `autouse` (a past bug ran native twice while reporting codegen).

- [ ] **Step 12: Full suite**

Run: `uv run pytest -q`

Expected: `2 skipped` (down from 5), no failures. The 2 remaining skips are the struct-unnest cases.

- [ ] **Step 13: Commit**

```bash
git add sql_transform/_codegen/plan.py sql_transform/_codegen/engine.py sql_transform/_codegen/runtime.py sql_transform/_codegen/plan_test.py sql_transform/_codegen/engine_test.py tests/test_codegen_coverage.py
git commit -m "feat(codegen): unnest(list) row multiplication — TASK-29 Phase C (Task 1)"
```

---

### Task 2: `unnest(struct)` — per-field column expansion (retires 2 skips)

Retires `test_unnest_struct_expands_columns` and `test_unnest_struct_column_expands_columns`. Cardinality is unchanged; the single projection item becomes one column per struct field. The whole difficulty is the column NAMES — DataFusion derives them from its own rendering of the argument expression, the differential harness compares column names, and native already mirrors this in `unnest_display_name` (`src/plan.rs`).

**Files:**
- Modify: `sql_transform/_codegen/plan.py`, `tests/test_codegen_coverage.py`

**Interfaces:**
- Consumes: `Column`, `FieldAccess`, `StructExpr`, `StructBase`, and the `elif isinstance(arg_type.base, StructBase):` branch left by Task 1, Step 6.
- Produces: `plan._unnest_display_name(e) -> str` — DataFusion's rendering of an unnest argument.

- [ ] **Step 1: Move the struct-unnest shape to `_COMMITTED` (RED)**

In `tests/test_codegen_coverage.py`, add to the end of `_COMMITTED`:

```python
    (
        "SELECT unnest(s) FROM t",
        {"t": rows({"s": "struct{x:int}"}, [{"s": {"x": 1}}])},
    ),
```

- [ ] **Step 2: Run the coverage guard to verify RED**

Run: `uv run pytest tests/test_codegen_coverage.py -q`

Expected: FAIL — `committed surface must not be deferred: 'SELECT unnest(s) FROM t' raised unnest() on a struct is not supported in codegen yet` (the `UnsupportedInCodegen` Task 1 Step 6 left in place).

- [ ] **Step 3: Add the display-name helper**

In `sql_transform/_codegen/plan.py`, immediately AFTER `_unnest_arg`, add:

```python
def _unnest_display_name(e: Any) -> str:
    """Render `unnest()`'s argument the way DataFusion's logical-plan Display
    does -- that is what it derives the expanded column names from, so matching
    it exactly is what makes the differential tests agree column-for-column.
    Mirrors native unnest_display_name; columns are already qualified with their
    effective table by _validate_expr.

    ponytail: only the node shapes reachable as an unnest() arg today (columns,
    field access, named_struct/struct construction). Struct construction always
    renders as named_struct(...) -- StructExpr can't tell the two authored forms
    apart, and DataFusion names a struct()-built unnest differently. Widen if
    that combination ever needs differential coverage."""
    if isinstance(e, Column):
        return f"{e.table}.{e.name}"
    if isinstance(e, FieldAccess):
        return f"{_unnest_display_name(e.base)}.{e.field}"
    if isinstance(e, StructExpr):
        inner = ",".join(
            f'Utf8("{key}"),{_unnest_display_name(value)}' for key, value in e.fields
        )
        return f"named_struct({inner})"
    raise ValueError("unnest() argument is too complex to name")
```

Two naming rules this encodes, both taken from native and both load-bearing for the differential tests: a column is rendered **qualified** (`t.s`, never `s`), and a constructed struct is rendered `named_struct(Utf8("x"),t.a,Utf8("y"),t.b)` — the literal key wrapped in `Utf8("...")`, comma-separated with no spaces. `_validate_expr` has already set `Column.table` to the effective table name by the time this runs, so no schema lookup is needed (native does look up, because its `Column.table` can still be `None` at that point).

- [ ] **Step 4: Replace the deferral with the expansion**

In `validate_columns`, REPLACE the branch Task 1 left:

```python
        elif isinstance(arg_type.base, StructBase):
            raise UnsupportedInCodegen(
                "unnest() on a struct is not supported in codegen yet"
            )
```

with:

```python
        elif isinstance(arg_type.base, StructBase):
            # unnest(struct) expands into ONE COLUMN PER FIELD, named
            # "<arg display>.<field>" -- DataFusion ignores the SELECT alias here.
            display = _unnest_display_name(arg)
            expanded.extend(
                (f"{display}.{name}", FieldAccess(arg, name))
                for name, _ in arg_type.base.fields
            )
```

`alias` is deliberately unused in this branch: DataFusion discards any `AS` on a struct unnest, so mirroring it means discarding it too.

- [ ] **Step 5: Verify GREEN on the coverage guard**

Run: `uv run pytest tests/test_codegen_coverage.py -q`

Expected: PASS — `_DEFERRED` now holds only the `s || 'x'` container-operand guard.

- [ ] **Step 6: Verify the 2 struct skips flipped to green against the oracle**

Run: `uv run pytest tests/test_diff_types.py -v -k unnest`

Expected: all 12 IDs PASSED, no SKIPPED. In particular `test_unnest_struct_expands_columns[codegen]` and `test_unnest_struct_column_expands_columns[codegen]`.

If either fails, read the assertion message before touching the implementation: a **row-key set mismatch** (`{'t.s.x'} != {'s.x'}` or similar) is a display-name bug in Step 3, not a value bug. Compare against the oracle's column names in the failure output and fix `_unnest_display_name` to match.

- [ ] **Step 7: Full suite**

Run: `uv run pytest -q`

Expected: `0 skipped`, no failures. This is the number to report at merge: the differential skip set went 5 → 0 across Phase C, and 16 → 0 across TASK-29.

- [ ] **Step 8: Lint**

Run: `uv run pre-commit run --all-files`

Expected: PASS (ruff-check and ruff-format both clean). This is what CI runs, so a local pass is the real gate.

- [ ] **Step 9: Commit**

```bash
git add sql_transform/_codegen/plan.py tests/test_codegen_coverage.py
git commit -m "feat(codegen): unnest(struct) per-field column expansion — TASK-29 Phase C (Task 2)"
```

---

## Self-Review

- **Spec coverage.** Phase C section of `2026-07-20-codegen-deferred-surface-design.md`, item by item: `Unnest` plan node → Task 1 Step 4 ✓; build-time rewrite of the projection item to a synthetic emitted column → Task 1 Step 6 ✓; `\0unnest` reserved key → Task 1 Step 4 ✓; at-most-one-unnest build error → Task 1 Step 6, tested Step 9 ✓; `_emit_rel` `for` loop over the list → Task 1 Step 8 ✓; NULL/non-list handling pinned against the oracle → Task 1 Step 8 (`unnest_rows`: NULL → zero rows, non-list → error) with the oracle check at Step 11 ✓; emitted column types as the list element type → Task 1 Step 6 (`effective_schemas[UNNEST_KEY][alias] = arg_type.base.elem`) ✓. C2 struct expansion + DataFusion column naming → Task 2 ✓. Testing bar (differential green + `_DEFERRED`→`_COMMITTED` + skip delta reported) → Task 1 Steps 1/11/12 and Task 2 Steps 1/6/7 ✓.
- **Placeholder scan.** No TBD/TODO, no "add error handling", no "similar to Task N". Every code step carries the literal code to write; the two rel-walker edits are given as explicit before/after rather than a description.
- **Type consistency.** `Unnest(input, list_expr, output_col)` is defined once (Task 1 Step 4) and constructed with exactly those three positional fields (Step 6), read with `node.input` / `node.list_expr` / `node.output_col` (Steps 7–8). `UNNEST_KEY` is defined in `plan.py` and referenced as `cp.UNNEST_KEY` from `engine.py` — the module is imported as `cp` there. `rt.unnest_rows(value, name)` matches its definition's `(v, name)` arity. `_unnest_arg` is defined in Task 1 and reused unchanged in Task 2. `_unnest_display_name` is defined and used only in Task 2.
- **Ordering risk flagged.** Task 1 Step 6 must raise `UnsupportedInCodegen` (not `ValueError`) for the struct case. If that is written as a `ValueError`, the two struct differential cases turn from SKIPPED into FAILED between the two commits, and Task 1 cannot be merged independently.
- **Dead code removed, not deferred.** `_DEFERRED_FUNCS = ("unnest",)` becomes dead once Task 1 Step 3 converts unnest (measured fact #1: it never reaches the `exp.Anonymous` branch), so Step 3 deletes the constant and its single use. AmirHossein's call, 2026-07-24, overriding this plan's first draft, which left it in place.
