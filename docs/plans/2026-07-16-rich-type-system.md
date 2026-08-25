# Rich (recursive) type system + UNNEST — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the `InferFn` interpreter a recursive, schema-driven type layer carrying **struct** and **list** values, plus **`UNNEST`** (struct→columns, list→rows), all bit-identical to DataFusion via the differential harness.

**Architecture:** Make `Value` (`src/expr.rs`) and `Base`/`FieldType` (`src/types.rs`) recursive (container variants wrapping inner types); drive Python↔`Value` marshalling off the declared schema; dispatch ops per type with a clean-error default; expand `unnest(struct)` at plan-build (projection) and `unnest(list)` via a new cardinality-changing `RelNode::Unnest`.

**Tech Stack:** Rust + pyo3 0.29 (`src/`, built via `uv run maturin develop`), sqlparser 0.62, Python 3.14, DataFusion (`datafusion` pkg) as the differential oracle, pytest, the differential harness `tests/differential.py`.

**Spec:** [docs/superpowers/specs/2026-07-16-rich-type-system-design.md](../specs/2026-07-16-rich-type-system-design.md)

## Global Constraints

- **DataFusion is the oracle.** Every new SQL capability is proven with a `tests/differential.py`-style case asserting `transform` (DataFusion) == `infer` (`InferFn`). Verified semantics to match: `named_struct('x',1,'y',2)` builds a struct; `unnest(struct)`→fields as columns (same cardinality); `unnest(list)`→one row per element, **empty list→0 rows, NULL list→0 rows**; struct deep-equality + struct-as-join-key work; `(expr).field` is **rejected** by DataFusion (do NOT support it) — field access only on an aliased struct column `s.x`.
- **`unnest(x)` parses as `SqlExpr::Function{name:"unnest", args:[x]}`** in SELECT position (no dedicated AST node); struct-vs-list dispatch is **type-driven at plan-build**.
- **Rust rebuild after any `src/` change:** `uv run maturin develop`. Run tests: `uv run pytest`. Rust unit tests: `cargo test`.
- **v0, no back-compat.** Change `Value`/`Base`/`Schema` shapes in place. Keep the opaque `Value::Object` for genuinely-unknown values.
- **No over-quoting / scalar regressions:** every `match Base` / `match Value` site gets a new arm; the existing scalar suite (current baseline **159 passed, 1 xfailed**) must stay green after each task.
- Ponytail (if active): reuse existing helpers, smallest correct diff, no speculative abstraction; the recursion is the point, don't gold-plate the op surface (arithmetic/CAST on containers = clean error, matching DataFusion).

## File Structure

- `src/expr.rs` — `Value` (+`Struct`,`List`), `Clone`/`Hash`/`PartialEq`, `eval` (construct, field access, list index), `compare_values`.
- `src/types.rs` — `Base` (+`Struct`,`List`), `infer_type`, `compatible`, `resolve_column_type`.
- `src/expr_build.rs` — parse `named_struct`/`struct`/`[…]`/`s.x`; detect `unnest`.
- `src/plan.rs` — `RelNode::Unnest`, `build_projection` (unnest-struct expansion), `build_plan`/`build_from` (unnest-list operator), `execute`/`execute_rel`, `validate_*`.
- `src/schema.rs` — `field_type_to_python`, `annotation_to_field_type`, `from_arrow_table`, output-model synthesis — all recursive.
- `src/lib.rs` — Python→`Value` input marshalling + `Value`→Python output, recursive.
- `tests/differential.py` — extend `_arrow`/`rows`/`static` to accept `struct{…}` / `list[…]` type specs.
- `tests/test_diff_types.py` (new) — differential parity for all new capabilities.

---

### Task 1: Recursive `Value` + `Base` spine (compile-clean, unit-tested)

Add the container variants and make the whole crate compile — every `match` over `Value`/`Base` gets its arm. No new SQL surface yet; proven by Rust unit tests for equality/hash.

**Files:** Modify `src/expr.rs`, `src/types.rs`, `src/schema.rs` (match arms). Test: `cargo test` (inline `#[cfg(test)]`).

**Interfaces — Produces:**
- `Value::Struct(Vec<(String, Value)>)`, `Value::List(Vec<Value>)` (ordered).
- `Base::Struct(Vec<(String, FieldType)>)`, `Base::List(Box<FieldType>)`.

- [ ] **Step 1: Failing Rust unit test** — in `src/expr.rs` `#[cfg(test)]`:
```rust
#[test]
fn struct_and_list_value_equality() {
    let a = Value::Struct(vec![("x".into(), Value::Int(1))]);
    let b = Value::Struct(vec![("x".into(), Value::Int(1))]);
    let c = Value::List(vec![Value::Int(1), Value::Int(2)]);
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(c.clone(), c);
}
```
- [ ] **Step 2: Run, expect compile error** — `cargo test struct_and_list_value_equality` → FAIL (no `Value::Struct`).
- [ ] **Step 3: Add the variants + trait impls.** In `src/expr.rs:4` add `Struct(Vec<(String, Value)>)` and `List(Vec<Value>)` to `Value`. Extend the hand-written `Clone` (`:16`), `PartialEq` (`:32`), and `Hash` (if present near `:44`) with structural arms (recurse over the field/element `Value`s). In `type_name`/display (`:85`,`:97`) add `"struct"`/`"list"` and a `Debug`-style render. In `src/types.rs:4` add `Base::Struct(Vec<(String, FieldType)>)` and `Base::List(Box<FieldType>)`. Add arms to every `match Base`: `compatible()` (`types.rs:154` — struct compatible iff same field names+compatible types; list iff compatible element), and `field_type_to_python` (`schema.rs:174` — struct → the value `typing.Any` placeholder for now (Task 3 makes it a nested model); list → `list[...]`). Any exhaustive `match` the compiler flags gets an arm.
- [ ] **Step 4: Run** — `cargo test struct_and_list_value_equality` → PASS; `cargo build` clean.
- [ ] **Step 5: Full suite unaffected** — `uv run maturin develop && uv run pytest -q` → still 159 passed, 1 xfailed (no SQL surface changed).
- [ ] **Step 6: Commit** — `git add src/expr.rs src/types.rs src/schema.rs && git commit -m "feat: recursive Value/Base spine (struct+list variants)"`

---

### Task 2: Struct + list construction expressions

Parse and evaluate `named_struct('x',e,…)`, `struct(e,…)` (auto-names `c0,c1`), and `[e,…]`.

**Files:** Modify `src/expr_build.rs`, `src/expr.rs` (`Expr`+`eval`), `src/types.rs` (`infer_type`). Test: `tests/test_diff_types.py` (new).

**Interfaces — Consumes** Task 1 variants. **Produces** `Expr::Struct(Vec<(String, Expr)>)`, `Expr::List(Vec<Expr>)`.

- [ ] **Step 1: Failing differential test** — create `tests/test_diff_types.py`:
```python
from differential import check
def test_struct_construct():
    check("SELECT named_struct('x', a, 'y', b) AS s FROM t",
          {"t": __import__("differential").rows({"a": "int", "b": "int"}, [{"a": 1, "b": 2}])})
def test_list_construct():
    check("SELECT [a, b, a] AS l FROM t",
          {"t": __import__("differential").rows({"a": "int", "b": "int"}, [{"a": 1, "b": 2}])})
```
(These will also need harness struct/list output support from Task 3 to *compare* — if the comparison can't represent the struct yet, assert `_run_infer` returns the right Python shape directly here and defer full `check()` equality to Task 3. Prefer: land Task 3's harness support first if the reviewer flags ordering — see Task 3.)
- [ ] **Step 2: Run, expect fail** — `uv run pytest tests/test_diff_types.py::test_struct_construct -x` → FAIL (`convert_expr` Err on the function / array node).
- [ ] **Step 3: Parse + eval + type.** `src/expr_build.rs`: in `convert_function`, match `name.eq_ignore_ascii_case("named_struct")` → pair args as (literal-string key, value expr) → `Expr::Struct`; `"struct"` → auto-name `c{i}`. Handle `SqlExpr::Array` → `Expr::List`. `src/expr.rs`: add `Expr::Struct`/`Expr::List` variants; `eval` builds `Value::Struct`/`Value::List` by evaluating children. `src/types.rs` `infer_type`: `Expr::Struct` → `Base::Struct(field types)`; `Expr::List` → `Base::List(unified element type)` (element types must unify or → `Base::Other`).
- [ ] **Step 4: Rebuild + run** — `uv run maturin develop && uv run pytest tests/test_diff_types.py -x` → PASS (pending Task 3 harness support; see note).
- [ ] **Step 5: Commit** — `git add src/expr_build.rs src/expr.rs src/types.rs tests/test_diff_types.py && git commit -m "feat: struct/list construction expressions"`

---

### Task 3: Schema-driven marshalling (Python ↔ struct/list, in and out)

Read Python `dict`/`list` inputs into `Value::Struct`/`List` per the declared schema, and emit `Value::Struct`→nested pydantic model / `Value::List`→`list`. Extend the differential harness to declare/compare struct+list columns.

**Files:** Modify `src/lib.rs` (input read + output build), `src/schema.rs` (`annotation_to_field_type`, `from_arrow_table`, output-model synthesis, `field_type_to_python`). Test: `tests/differential.py` (harness), `tests/test_diff_types.py`.

**Interfaces — Consumes** Task 1/2. **Produces** harness helpers accepting `"struct{x:int,y:int}"` / `"list[int]"` type specs.

- [ ] **Step 1: Failing round-trip test** — append to `tests/test_diff_types.py`:
```python
def test_struct_input_roundtrip():
    check("SELECT s FROM t",
          {"t": __import__("differential").rows({"s": "struct{x:int,y:int}"},
                                                [{"s": {"x": 1, "y": 2}}])})
def test_list_input_roundtrip():
    check("SELECT l FROM t",
          {"t": __import__("differential").rows({"l": "list[int]"}, [{"l": [1, 2, 3]}])})
```
- [ ] **Step 2: Run, expect fail** — FAIL (harness `_arrow_field` can't parse `struct{…}`; and `InferFn` output/synthesis can't represent the struct).
- [ ] **Step 3: Harness support** — in `tests/differential.py`, extend `_arrow_field` to parse `struct{f:spec,…}` → `pa.struct([...])` (recursive) and `list[spec]` → `pa.list_(...)`. Ensure `_val_equal`/`_rows_equal` compare dict/list values structurally (dicts field-wise, lists element-wise, reusing `_val_equal`).
- [ ] **Step 4: Rust marshalling** — `src/schema.rs`: `annotation_to_field_type` (`:100`) resolves `list[X]`→`Base::List`, a nested pydantic `BaseModel` annotation → `Base::Struct` (introspect its fields recursively). `from_arrow_table` (`:37`) walks `pa.StructType`/`ListType` children instead of prefix-matching. `field_type_to_python` (`:174`): `Base::Struct` → a nested `create_model` sub-model; `Base::List` → `list[inner]`. Output-model synthesis (`lib.rs:27`) recurses. Input read (`lib.rs`): a Python `dict`→`Value::Struct` / `list`→`Value::List` per the field's declared `Base` (recursively).
- [ ] **Step 5: Rebuild + run** — `uv run maturin develop && uv run pytest tests/test_diff_types.py tests/differential.py -q` → PASS; full suite green.
- [ ] **Step 6: Commit** — `git add tests/differential.py src/lib.rs src/schema.rs tests/test_diff_types.py && git commit -m "feat: schema-driven struct/list marshalling (python in/out)"`

---

### Task 4: Struct field access `s.x`

Resolve dotted identifiers as struct field access when the qualifier isn't a relation alias.

**Files:** Modify `src/expr_build.rs` (loosen `parts.len()==2` guard `:15`), `src/expr.rs` (`Expr::FieldAccess`+eval), `src/types.rs` (`infer_type`), `src/plan.rs` (`validate_expr` `:743` — distinguish qualifier). Test: `tests/test_diff_types.py`.

**Interfaces — Produces** `Expr::FieldAccess{base: Box<Expr>, field: String}`.

- [ ] **Step 1: Failing test** — append:
```python
def test_struct_field_access():
    check("SELECT s.x AS fx FROM t",
          {"t": __import__("differential").rows({"s": "struct{x:int,y:int}"},
                                                [{"s": {"x": 5, "y": 9}}])})
```
- [ ] **Step 2: Run, expect fail** — FAIL (`s.x` resolves as `table.column`, no table `s`).
- [ ] **Step 3: Implement.** `expr_build.rs`: for `CompoundIdentifier(parts)`, if `parts[0]` is a known relation alias (pass the alias set in, or resolve later) keep `Expr::Column`; else build nested `Expr::FieldAccess` (leftmost is a `Column`, each subsequent part a field). **Precedence rule (open item #4): relation alias wins, else struct field.** `expr.rs` `eval`: `FieldAccess` evaluates base → `Value::Struct`, returns the named field (missing field → clean error; base not a struct → clean error). `infer_type`: look the field up in the base's `Base::Struct` schema. `plan.rs` `validate_expr`: a `FieldAccess` validates its base + that the field exists in the base's struct type.
- [ ] **Step 4: Rebuild + run** — PASS; full suite green.
- [ ] **Step 5: Commit** — `git commit -am "feat: struct field access s.x"`

---

### Task 5: `unnest(struct)` → columns (projection expansion)

Detect `unnest(<struct>)` in a SELECT item and expand it, at plan-build, into one field-access projection per struct field.

**Files:** Modify `src/plan.rs` (`build_projection` `:247`). Test: `tests/test_diff_types.py`.

**Interfaces — Consumes** Task 4 `FieldAccess`, Task 1 `Base::Struct`.

- [ ] **Step 1: Failing test** — append:
```python
def test_unnest_struct_expands_columns():
    check("SELECT unnest(named_struct('x', a, 'y', b)) FROM t",
          {"t": __import__("differential").rows({"a": "int", "b": "int"}, [{"a": 1, "b": 2}])})
```
(Match DataFusion's output column names — verify via the oracle; `check()` compares by value/shape.)
- [ ] **Step 2: Run, expect fail** — FAIL (`unnest` is an unknown function producing one opaque column, ≠ DataFusion's two).
- [ ] **Step 3: Implement.** In `build_projection`, when a `SelectItem`'s expr is `Function{name:"unnest", args:[arg]}` and `infer_type(arg)` is `Base::Struct(fields)`: replace that single item with one `(col_name, FieldAccess{base: arg, field})` per field, naming columns to match DataFusion (`<arg_sql>.<field>` unless the item was aliased). If the arg's type isn't a struct, leave for Task 6 (list) / error.
- [ ] **Step 4: Rebuild + run** — PASS; full suite green.
- [ ] **Step 5: Commit** — `git commit -am "feat: unnest(struct) projection expansion to columns"`

---

### Task 6: `unnest(list)` → rows (`RelNode::Unnest`, cardinality change)

The novel piece: one input row with an N-element list becomes N output rows.

**Files:** Modify `src/plan.rs` (`RelNode` `:33`, `build_plan`, `execute_rel` `:448`, `resolve_tables`/`validate_rel`). Test: `tests/test_diff_types.py`.

**Interfaces — Produces** `RelNode::Unnest { input: Box<RelNode>, list_expr: Expr, output_col: String }`.

- [ ] **Step 1: Failing test** — append:
```python
def test_unnest_list_expands_rows():
    check("SELECT id, unnest(vals) AS v FROM t",
          {"t": __import__("differential").rows(
              {"id": "int", "vals": "list[int]"},
              [{"id": 1, "vals": [10, 20, 30]}, {"id": 2, "vals": []}, {"id": 3, "vals": None}])},
          expect=[{"id": 1, "v": 10}, {"id": 1, "v": 20}, {"id": 1, "v": 30}])
    # empty list (id=2) and NULL list (id=3) both -> zero rows
```
- [ ] **Step 2: Run, expect fail** — FAIL (no row multiplication).
- [ ] **Step 3: Implement.** Add `RelNode::Unnest{input, list_expr, output_col}`. In `build_plan`: when a projection contains `Function{name:"unnest",args:[arg]}` with `infer_type(arg)==Base::List`, wrap the input rel in `RelNode::Unnest{input, list_expr:arg, output_col:<alias>}` and replace the select item with `Column{table:None, name:<alias>}`. In `execute_rel` add the `Unnest` arm: for each input `Row`, `eval(list_expr)` → `Value::List(items)` (or `Null`/empty → **skip the row**); for each item, clone the row and insert `output_col → item` (under a synthetic table key consistent with `resolve_column`), pushing one output row per item. `resolve_tables`/`validate_rel` (`:581`,`:656`): thread through `Unnest` (recurse into `input`; the `output_col` becomes an available unqualified column typed as the list's element `Base`).
- [ ] **Step 4: Rebuild + run** — `uv run pytest tests/test_diff_types.py::test_unnest_list_expands_rows -x` → PASS; full suite green.
- [ ] **Step 5: Commit** — `git commit -am "feat: unnest(list) row expansion via RelNode::Unnest"`

---

## Notes for the implementer

- **Task ordering caveat:** Task 2's `check()` equality needs Task 3's harness struct/list comparison. If a reviewer flags it, either (a) in Task 2 assert on `_run_infer`'s raw Python output shape and defer full `check()` to Task 3, or (b) swap Tasks 2↔3's harness step first. Don't leave Task 2 asserting nothing.
- **The `match Base`/`match Value` churn (Task 1) is the silent-regression risk** — after adding variants, `cargo build` will flag every non-exhaustive match; add a real arm to each (clean error for unsupported ops on containers, matching DataFusion), don't `_ => unreachable!()`.
- **Do not support `(expr).field`** — DataFusion rejects it; the harness would diverge. Field access is `CompoundIdentifier` dotted only.
- **Deferred (not this plan):** temporal/decimal/map/dictionary/binary types (fast-follows on this spine), `SELECT *`/general wildcard, multi-`unnest` cross-product semantics.
