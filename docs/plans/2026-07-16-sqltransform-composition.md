# SQLTransform Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a `SQLTransform` reference another **fitted** `SQLTransform` inside its SQL via a PEP 750 t-string (`{scaler.transform}(age)`), inlining the referenced transform's frozen expression so the composite fits/transforms/infers as one fused transform.

**Architecture:** Composition is a fit-time front-end (`_compose.py`) that runs before the existing find-windows → build-state → rewrite pipeline and reduces the composite to a plain SELECT plus name-scoped state tables. From there the existing pipeline runs unchanged. Both engines run the same rewritten SQL over the same merged frozen state.

**Tech Stack:** Python 3.14 (t-strings), sqlglot (AST), DataFusion (`datafusion`, fit-time state + batch), Rust `InferFn` (inference), pyarrow, Pydantic v2, pytest, the differential harness (`tests/differential.py`).

**Spec:** [docs/superpowers/specs/2026-07-16-sqltransform-composition-design.md](../specs/2026-07-16-sqltransform-composition-design.md)

## Global Constraints

- **Python floor is 3.14** — `t"..."` (PEP 750) is available natively; `from string.templatelib import Template, Interpolation`.
- **v0, no back-compat** — change `WindowAgg`, `state_key`, `build_state_tables`, `rewrite_sql`, `__init__` signatures in place; no shims.
- **Frozen path only** — `{a.transform}(col)`. Bare `{a}` raises NotImplementedError. Single-input, single-output referenced transforms only.
- **Acceptance = differential parity** — every new capability proven with a `tests/differential.py` `check(...)` case: DataFusion (oracle) and Rust `InferFn` must return identical values. Per-engine correctness alone is not sufficient.
- **Ponytail (full)** — reuse existing helpers (`state_key`, `state_table_name`, `rewrite_sql`, `parse_and_validate`), stdlib first (`hashlib`, `inspect`, `string.templatelib`), shortest working diff. No abstraction without a second caller.
- **Referenced transforms are read-only** — never mutate or re-fit `a`; the composite owns all fitted state.
- Rebuild the Rust extension only if `src/` changes (it does not in this plan): `uv run maturin develop`. Run tests with `uv run pytest`.

---

### Task 1: Frozen inline, scalar position — the core primitive

Inline a fitted referenced transform's frozen expression into an outer SELECT, remapped onto the call column and state name-scoped. Covers the scalar-position cases (no outer window aggregate over the inlined column): single reference, column remap, zero-state inner, repeated + multiple references.

**Files:**
- Create: `sql_transform/_compose.py`
- Create: `sql_transform/_compose_test.py`
- Modify: `sql_transform/_rewrite.py` (add `extra_marker_tables` param)
- Modify: `sql_transform/__init__.py` (`__init__` accepts `Template`; `fit()` runs the front-end)
- Create: `tests/test_diff_composition.py`

**Interfaces:**
- Consumes: `parse_and_validate` (`_sql.py`), `find_window_aggregates` (`_sql.py`), `build_state_tables` (`_state.py`), `rewrite_sql` (`_rewrite.py`), `synthesize_this_model` (`_schema.py`), `InferFn`, `STATE_MARKER` (`_state.py`), `SQLTransform._rewritten_sql` / `._state_tables` / `._infer_fn` (frozen inner reads).
- Produces:
  - `_compose.desugar_template(template: Template) -> tuple[str, dict[str, Ref]]`
  - `_compose.inline_references(select: exp.Select, refs: dict[str, Ref]) -> InlineResult` where `InlineResult.scoped_state: dict[str, pa.Table]`
  - `_compose.Ref(transform: SQLTransform, frozen: bool, expr_text: str)`
  - `rewrite_sql(select, windows, extra_marker_tables=()) -> str`
  - `SQLTransform.__init__(self, sql: str | Template)`; composites use the same `fit`/`transform`/`infer`.

- [ ] **Step 1: Write the failing differential test (single reference + remap)**

Create `tests/test_diff_composition.py`:

```python
"""Differential parity for SQLTransform composition ({transform}(col))."""

import pyarrow as pa

from sql_transform import SQLTransform
from differential import _rows_equal


def _fit_scaler():
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    scaler = SQLTransform(
        "SELECT (age - AVG(age) OVER ()) / STDDEV(age) OVER () AS s FROM __THIS__"
    ).fit(train)
    return scaler, train


def _parity(composite, batch):
    rows = batch.to_pylist()
    batch_out = composite.transform(batch).to_pylist()
    infer_out = [r.model_dump() for r in composite.infer_batch(rows)]
    assert _rows_equal(batch_out, infer_out), (batch_out, infer_out)
    return batch_out


def test_single_reference_parity():
    scaler, train = _fit_scaler()
    composite = SQLTransform(
        t"SELECT {scaler.transform}(age) AS age_scaled FROM __THIS__"
    ).fit(train)
    out = _parity(composite, train)
    # (age - mean=25) / stddev(sample)=12.909944...
    assert abs(out[0]["age_scaled"] - ((10.0 - 25.0) / 12.909944487358056)) < 1e-9


def test_column_remap_parity():
    scaler, train = _fit_scaler()
    data = pa.table({"income": [10.0, 20.0, 30.0, 40.0]})
    composite = SQLTransform(
        t"SELECT {scaler.transform}(income) AS scaled FROM __THIS__"
    ).fit(train.append_column("income", train.column("age")))
    _parity(composite, data.append_column("age", data.column("income")))
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest tests/test_diff_composition.py -x -q`
Expected: FAIL — `TypeError`/`AttributeError` (SQLTransform can't take a `Template` yet).

- [ ] **Step 3: Write `_compose.py`**

Create `sql_transform/_compose.py`:

```python
"""Fit-time front-end: inline a fitted SQLTransform referenced via a t-string.

`SQLTransform(t"... {a.transform}(col) ...")` desugars to plain SQL with
`__COMPOSE_i__(col)` placeholder calls plus a ref map; at fit() the placeholders
are replaced by the referenced transform's frozen scalar expression, remapped to
`col` and state name-scoped to `__STATE_R{i}__`. Frozen path only.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from string.templatelib import Template

import pyarrow as pa
import sqlglot
from sqlglot import exp


@dataclass(frozen=True)
class Ref:
    transform: object  # a SQLTransform (imported lazily to avoid a cycle)
    frozen: bool       # True for {a.transform}; False for bare {a}
    expr_text: str     # interpolation source, for error messages


@dataclass(frozen=True)
class InlineResult:
    scoped_state: dict[str, pa.Table]


def desugar_template(template: Template) -> tuple[str, dict[str, Ref]]:
    """Turn a t-string into (plain SQL with __COMPOSE_i__ placeholders, ref map)."""
    from sql_transform import SQLTransform

    parts: list[str] = []
    refs: dict[str, Ref] = {}
    i = 0
    for item in template:
        if isinstance(item, str):
            parts.append(item)
            continue
        v = item.value  # Interpolation
        if (
            inspect.ismethod(v)
            and isinstance(v.__self__, SQLTransform)
            and v.__func__ is SQLTransform.transform
        ):
            ref = Ref(v.__self__, frozen=True, expr_text=item.expression)
        elif isinstance(v, SQLTransform):
            ref = Ref(v, frozen=False, expr_text=item.expression)
        else:
            raise TypeError(
                f"interpolation {{{item.expression}}} must be a SQLTransform or "
                f"its .transform, got {type(v).__name__}"
            )
        name = f"__COMPOSE_{i}__"
        refs[name] = ref
        parts.append(name)
        i += 1
    return "".join(parts), refs


def inline_references(select: exp.Select, refs: dict[str, Ref]) -> InlineResult:
    """Replace each __COMPOSE_i__(col) node with the referenced transform's frozen,
    remapped, name-scoped expression. Mutates `select`. Empty refs -> no-op."""
    scoped_state: dict[str, pa.Table] = {}
    for i, (name, ref) in enumerate(refs.items()):
        node = _find_call(select, name, ref)
        argcol = _single_col_arg(node, ref)
        _require_frozen_fitted(ref)
        expr, inner_col, scope, state = _frozen_expr(ref.transform, i)

        def rewrite(n: exp.Expression) -> exp.Expression:
            if isinstance(n, exp.Column):
                if n.table == "__THIS__":
                    col = argcol if n.name == inner_col else n.name
                    return exp.column(col, table="__THIS__")
                if n.table and n.table.startswith("__STATE"):
                    return exp.column(n.name, table=scope)
            return n

        node.replace(expr.transform(rewrite))
        scoped_state.update(state)
    return InlineResult(scoped_state=scoped_state)


def _find_call(select: exp.Select, name: str, ref: Ref) -> exp.Anonymous:
    for n in select.find_all(exp.Anonymous):
        if str(n.this).upper() == name:
            return n
    raise ValueError(
        f"a referenced transform must be applied to a column, "
        f"e.g. {{{ref.expr_text}}}(age)"
    )


def _single_col_arg(node: exp.Anonymous, ref: Ref) -> str:
    args = node.expressions
    if len(args) != 1 or not isinstance(args[0], exp.Column):
        raise ValueError(
            f"a referenced transform must be applied to a single input column, "
            f"e.g. {{{ref.expr_text}}}(age)"
        )
    return args[0].name


def _require_frozen_fitted(ref: Ref) -> None:
    if not ref.frozen:
        raise NotImplementedError(
            f"fit-cascade composition ({{{ref.expr_text}}}(col)) is not yet "
            f"implemented; fit it and reference {{{ref.expr_text}.transform}}(col)"
        )
    if ref.transform._infer_fn is None:
        raise ValueError(
            f"referenced transform {{{ref.expr_text}}} is not fitted; "
            f"call .fit(...) before referencing it"
        )


def _frozen_expr(inner, i: int):
    """(expr, inner input col, scope name, {scope: state table}) for a fitted inner."""
    inner_select = sqlglot.parse_one(inner._rewritten_sql)
    if len(inner_select.expressions) != 1:
        raise ValueError(
            "referenced transform must be single-output (one SELECT expression); "
            "multi-output fan-out is not yet supported"
        )
    expr = inner_select.expressions[0]
    if isinstance(expr, exp.Alias):
        expr = expr.this
    this_cols = {c.name for c in expr.find_all(exp.Column) if c.table == "__THIS__"}
    states = inner._state_tables or {}
    if len(this_cols) > 1 or len(states) > 1:
        raise ValueError(
            "referenced transform must read exactly one input column; "
            "multi-input (incl. PARTITION BY) references are not yet supported"
        )
    inner_col = next(iter(this_cols), None)
    scope = f"__STATE_R{i}__"
    scoped = {scope: next(iter(states.values()))} if states else {}
    return expr, inner_col, scope, scoped
```

- [ ] **Step 4: Add `extra_marker_tables` to `rewrite_sql`**

In `sql_transform/_rewrite.py`, change the signature and append marker joins for the inlined scoped state tables. Replace the function signature line and add the loop before `return`:

```python
def rewrite_sql(
    select: exp.Select,
    windows: list[WindowAgg],
    extra_marker_tables: tuple[str, ...] = (),
) -> str:
```

Then, immediately before `return select.sql()`:

```python
    for table in extra_marker_tables:
        select.join(
            exp.to_table(table),
            on=f"{table}.{STATE_MARKER} = 0",
            join_type="LEFT",
            copy=False,
        )

    return select.sql()
```

- [ ] **Step 5: Wire `__init__` and `fit` in `__init__.py`**

In `sql_transform/__init__.py`, add imports and update `__init__`/`fit`:

```python
from string.templatelib import Template

from sql_transform._compose import inline_references, desugar_template
```

Replace `__init__`:

```python
    def __init__(self, sql: str | Template) -> None:
        if isinstance(sql, Template):
            self._sql, self._refs = desugar_template(sql)
        else:
            self._sql, self._refs = sql, {}
        self._state_tables: dict[str, pa.Table] | None = None
        self._rewritten_sql: str | None = None
        self._infer_fn: InferFn | None = None
```

Replace the body of `fit` (between `this_model = ...` and `return self`):

```python
        this_model = this_model or synthesize_this_model(table.schema)

        tree = parse_and_validate(self._sql)
        inline = inline_references(tree, self._refs)
        windows = find_window_aggregates(tree)

        ctx = datafusion.SessionContext()
        ctx.from_arrow(table, name="__THIS__")

        own_state = build_state_tables(windows, ctx, "__THIS__")
        self._state_tables = {**inline.scoped_state, **own_state}
        self._rewritten_sql = rewrite_sql(
            tree, windows, extra_marker_tables=tuple(inline.scoped_state)
        )
        self._infer_fn = InferFn(
            self._rewritten_sql,
            row_tables={"__THIS__": this_model},
            static_tables=self._state_tables,
        )
        return self
```

- [ ] **Step 6: Run the differential test, verify it passes**

Run: `uv run pytest tests/test_diff_composition.py -x -q`
Expected: PASS (2 passed).

- [ ] **Step 7: Add unit + parity tests for zero-state and repeated/multiple references**

Append to `tests/test_diff_composition.py`:

```python
def test_zero_state_inner_parity():
    train = pa.table({"age": [1.0, 2.0, 3.0]})
    doubler = SQLTransform("SELECT age * 2 AS d FROM __THIS__").fit(train)
    composite = SQLTransform(
        t"SELECT {doubler.transform}(age) AS d2 FROM __THIS__"
    ).fit(train)
    out = _parity(composite, train)
    assert out[0]["d2"] == 2.0


def test_repeated_and_multiple_references_parity():
    scaler, train = _fit_scaler()
    doubler = SQLTransform("SELECT age * 2 AS d FROM __THIS__").fit(train)
    composite = SQLTransform(
        t"SELECT {scaler.transform}(age) AS a, {scaler.transform}(age) AS b, "
        t"{doubler.transform}(age) AS c FROM __THIS__"
    ).fit(train)
    out = _parity(composite, train)
    assert out[0]["a"] == out[0]["b"]
    assert out[0]["c"] == 2.0
```

Create `sql_transform/_compose_test.py`:

```python
from string.templatelib import Template

from sql_transform._compose import desugar_template


def test_desugar_static_template_has_no_refs():
    # A t-string with no interpolations desugars to itself with an empty ref map.
    sql, refs = desugar_template(Template("SELECT 1 AS x"))
    assert sql == "SELECT 1 AS x"
    assert refs == {}
```

- [ ] **Step 8: Run tests, verify pass**

Run: `uv run pytest tests/test_diff_composition.py sql_transform/_compose_test.py -q`
Expected: PASS (4 + 1).

- [ ] **Step 9: Run the full suite (no regressions)**

Run: `uv run pytest -q`
Expected: PASS — prior baseline (138 passed, 2 xfailed) plus the new tests.

- [ ] **Step 10: Commit**

```bash
git add sql_transform/_compose.py sql_transform/_compose_test.py sql_transform/_rewrite.py sql_transform/__init__.py tests/test_diff_composition.py
git commit -m "feat: compose SQLTransforms via {a.transform}(col) — frozen inline (scalar)"
```

---

### Task 2: Error contract & non-mutation

Lock down every misuse with an explicit error, and prove the referenced transform is never mutated. These are the guards `inline_references` already routes through; this task drives each with a test-first case.

**Files:**
- Modify: `sql_transform/_compose_test.py` (error-matrix unit tests)
- Modify: `tests/test_diff_composition.py` (non-mutation parity)

**Interfaces:**
- Consumes: `desugar_template`, `inline_references`, `Ref` (Task 1); `SQLTransform`.
- Produces: no new code paths if Task 1's guards are complete; any gap found is fixed in `_compose.py`.

- [ ] **Step 1: Write failing error tests**

Append to `sql_transform/_compose_test.py`:

```python
import pyarrow as pa
import pytest

from sql_transform import SQLTransform


def _fitted_scaler():
    train = pa.table({"age": [1.0, 2.0, 3.0]})
    return SQLTransform(
        "SELECT (age - AVG(age) OVER ()) / STDDEV(age) OVER () AS s FROM __THIS__"
    ).fit(train), train


def test_bare_reference_is_not_implemented():
    scaler, train = _fitted_scaler()
    with pytest.raises(NotImplementedError, match="fit-cascade"):
        SQLTransform(t"SELECT {scaler}(age) AS s FROM __THIS__").fit(train)


def test_frozen_reference_on_unfit_errors():
    unfit = SQLTransform("SELECT age * 2 AS d FROM __THIS__")
    train = pa.table({"age": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="not fitted"):
        SQLTransform(t"SELECT {unfit.transform}(age) AS d FROM __THIS__").fit(train)


def test_reference_not_applied_to_column_errors():
    scaler, train = _fitted_scaler()
    with pytest.raises(ValueError, match="single input column"):
        SQLTransform(
            t"SELECT {scaler.transform}(age + 1) AS s FROM __THIS__"
        ).fit(train)


def test_multi_input_reference_errors():
    train = pa.table({"age": [1.0, 2.0], "city": ["x", "y"]})
    grouped = SQLTransform(
        "SELECT age / AVG(age) OVER (PARTITION BY city) AS e FROM __THIS__"
    ).fit(train)
    with pytest.raises(ValueError, match="one input column"):
        SQLTransform(t"SELECT {grouped.transform}(age) AS e FROM __THIS__").fit(train)


def test_non_transform_interpolation_errors():
    train = pa.table({"age": [1.0, 2.0]})
    with pytest.raises(TypeError, match="SQLTransform"):
        SQLTransform(t"SELECT {42}(age) AS s FROM __THIS__").fit(train)
```

- [ ] **Step 2: Run, verify pass or fix the gap**

Run: `uv run pytest sql_transform/_compose_test.py -q`
Expected: PASS. If any test fails (e.g. multi-output not guarded), add the missing guard to `_compose.py` (`_frozen_expr` / `_require_frozen_fitted`) until green. The multi-output guard already lives in `_frozen_expr`; add a test only if a single-output referenced transform can't express fan-out yet (it can't — skip).

- [ ] **Step 3: Write the non-mutation test**

Append to `tests/test_diff_composition.py`:

```python
def test_referenced_transform_not_mutated():
    scaler, train = _fit_scaler()
    before = scaler.transform(train).to_pylist()
    SQLTransform(t"SELECT {scaler.transform}(age) AS s FROM __THIS__").fit(train)
    after = scaler.transform(train).to_pylist()
    assert before == after  # scaler still fitted + unchanged
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_diff_composition.py sql_transform/_compose_test.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sql_transform/_compose_test.py tests/test_diff_composition.py
git commit -m "test: composition error contract + referenced-transform non-mutation"
```

---

### Task 3: Window aggregate over an expression (base-engine generalization)

Relax the "aggregate argument must be a single plain column" restriction so a window aggregate can wrap any scalar expression. This is a standalone SQL capability (no composition), and the prerequisite for Task 4. Deterministic hashed state key for expression args; plain-column args keep the existing `avg_age` naming.

**Files:**
- Modify: `sql_transform/_sql.py` (`find_window_aggregates`, `WindowAgg`)
- Modify: `sql_transform/_state.py` (`state_key` unchanged; group/dedup by precomputed key; value expr from arg node)
- Modify: `sql_transform/_rewrite.py` (use `w.key` instead of recomputing `state_key`)
- Modify: `sql_transform/_sql_test.py` / existing tests if they read `WindowAgg.col` (adjust to new fields)
- Create/extend: a differential test for `AGG(<expr>) OVER ()`

**Interfaces:**
- Consumes: `state_key`, `state_table_name`, `STATE_MARKER`.
- Produces: `WindowAgg` gains `arg: exp.Expression` and `key: str`; `col: str | None`. `rewrite_sql` and `build_state_tables` read `w.key` / `w.arg`.

- [ ] **Step 1: Write the failing differential test**

Append to `tests/test_diff_expressions.py` (or create `tests/test_diff_window_expr.py` if that file's parametrization doesn't fit):

```python
from sql_transform import SQLTransform
from differential import _rows_equal
import pyarrow as pa


def test_window_aggregate_over_expression_parity():
    train = pa.table({"x": [1.0, 2.0, 3.0, 4.0]})
    t = SQLTransform(
        "SELECT x / AVG(x + 1) OVER () AS r FROM __THIS__"
    ).fit(train)
    batch_out = t.transform(train).to_pylist()
    infer_out = [r.model_dump() for r in t.infer_batch(train.to_pylist())]
    assert _rows_equal(batch_out, infer_out)
    # AVG(x+1) over [2,3,4,5] = 3.5
    assert abs(batch_out[0]["r"] - (1.0 / 3.5)) < 1e-9
```

- [ ] **Step 2: Run, verify it fails**

Run: `uv run pytest tests/test_diff_expressions.py -k window_aggregate_over_expression -x -q`
Expected: FAIL — `ValueError: Window aggregate argument must be a single plain column`.

- [ ] **Step 3: Generalize `find_window_aggregates` + `WindowAgg`**

In `sql_transform/_sql.py`, add imports at top:

```python
import hashlib
```

Replace the `WindowAgg` dataclass fields:

```python
@dataclass(frozen=True)
class WindowAgg:
    node: exp.Window
    fn: str
    arg: exp.Expression        # the aggregate's single argument (column or expression)
    col: str | None            # arg's column name, or None if arg is an expression
    key: str                   # state value-column name (fn_col, or fn_e<hash>)
    partition_cols: tuple[str, ...]
    has_partition: bool
    has_order: bool
```

In `find_window_aggregates`, replace the arg-validation + append block (the `if len(args) != 1 ...` through the `windows.append(...)`). Compute the key **inline** — do **not** `import state_key` into `_sql.py`: `_state` already imports `WindowAgg` from `_sql`, so that import would be circular. The column branch duplicates one line of `state_key`'s formula; that is the correct trade over a cycle.

```python
        if len(args) != 1:
            raise ValueError(
                "Window aggregate must take exactly one argument: "
                f"{node.sql()!r}"
            )
        arg = args[0]
        if isinstance(arg, exp.Column):
            col = arg.name
            key = f"{fn.lower()}_{col.lower()}"          # matches state_key(fn, col)
        else:
            col = None
            digest = hashlib.blake2s(arg.sql().encode(), digest_size=4).hexdigest()
            key = f"{fn.lower()}_e{digest}"

        partition_by = node.args.get("partition_by") or []
        partition_cols: list[str] = []
        for p in partition_by:
            if not isinstance(p, exp.Column):
                raise ValueError(
                    f"PARTITION BY must be a list of plain columns: {node.sql()!r}"
                )
            partition_cols.append(p.name)

        windows.append(
            WindowAgg(
                node=node,
                fn=fn,
                arg=arg,
                col=col,
                key=key,
                partition_cols=tuple(partition_cols),
                has_partition=bool(node.args.get("partition_by")),
                has_order=bool(node.args.get("order")),
            )
        )
```

- [ ] **Step 4: Update `_rewrite.py` to use `w.key`**

In `sql_transform/_rewrite.py`, replace the `window_ref` construction to use the precomputed key:

```python
    window_ref = {
        id(w.node): (state_table_name(w.partition_cols), w.key, w.partition_cols)
        for w in windows
    }
```

Remove the now-unused `state_key` from the import line (keep `STATE_MARKER, state_table_name`).

- [ ] **Step 5: Update `build_state_tables` to group/dedup by key and emit from the arg node**

In `sql_transform/_state.py`, replace the grouping + per-group value-expr construction. The group value becomes a list of `WindowAgg`:

```python
    groups: dict[tuple[str, ...], list[WindowAgg]] = {}
    for w in windows:
        if w.has_order:
            raise NotImplementedError(
                "ORDER BY window aggregates are not yet supported by "
                "the Rust-backed SQLTransform pipeline"
            )
        groups.setdefault(w.partition_cols, []).append(w)

    tables: dict[str, pa.Table] = {}
    for partition_cols, members in groups.items():
        selected: dict[str, WindowAgg] = {}
        for w in members:
            existing = selected.get(w.key)
            if existing is not None and existing.arg.sql() != w.arg.sql():
                raise ValueError(
                    f"Ambiguous window aggregate: {w.fn}({w.arg.sql()}) normalizes "
                    f"to the same state key {w.key!r} as another aggregate in this "
                    "query"
                )
            selected[w.key] = w

        value_exprs = [
            f"{w.fn}({w.arg.sql()}) AS {key}" for key, w in selected.items()
        ]
```

The rest of the function (partitioned vs global SQL, nullable widening, marker column) is unchanged. Add `from sql_transform._sql import WindowAgg` is already present.

- [ ] **Step 6: Fix any test/reference that used old `WindowAgg` fields or `state_key` in `_rewrite`**

Run: `uv run pytest -q` and fix collection/attribute errors (e.g. `_sql_test.py` constructing `WindowAgg(...)` without `arg`/`key`, or asserting `.col`). Update those constructions to pass `arg=exp.column("age")`, `col="age"`, `key="avg_age"`.

- [ ] **Step 7: Run the new differential test + full suite**

Run: `uv run pytest tests/test_diff_expressions.py -k window_aggregate_over_expression -x -q` → PASS.
Run: `uv run pytest -q` → PASS (baseline + new).

- [ ] **Step 8: Commit**

```bash
git add sql_transform/_sql.py sql_transform/_state.py sql_transform/_rewrite.py sql_transform/_sql_test.py tests/test_diff_expressions.py
git commit -m "feat: window aggregates over an expression argument (hashed state key)"
```

---

### Task 4: Outer aggregate over the inlined column — the capstone

The done-criterion: `SELECT {scaler.transform}(age) / AVG({scaler.transform}(age)) OVER () AS z FROM __THIS__`. After inlining, the outer's `AVG(...) OVER ()` wraps an expression that references the inlined scoped state (`__STATE_R1__`), so state extraction must cross-join the scoped state tables. Combines Task 1 (inline) + Task 3 (agg-over-expression).

**Files:**
- Modify: `sql_transform/_state.py` (`build_state_tables` gains `join_tables`)
- Modify: `sql_transform/__init__.py` (`fit` passes `join_tables=inline.scoped_state`)
- Modify: `tests/test_diff_composition.py` (capstone parity)

**Interfaces:**
- Consumes: `inline_references().scoped_state` (Task 1), agg-over-expression (Task 3).
- Produces: `build_state_tables(windows, ctx, table_name, join_tables=None)` — registers `join_tables` in `ctx` and cross-joins them in the extraction query.

- [ ] **Step 1: Write the failing capstone test**

Append to `tests/test_diff_composition.py`:

```python
def test_outer_aggregate_over_inlined_column_parity():
    scaler, train = _fit_scaler()
    composite = SQLTransform(
        t"SELECT {scaler.transform}(age) "
        t"/ AVG({scaler.transform}(age)) OVER () AS z FROM __THIS__"
    ).fit(train)
    _parity(composite, train)
```

- [ ] **Step 2: Run, verify it fails**

Run: `uv run pytest tests/test_diff_composition.py -k outer_aggregate_over_inlined -x -q`
Expected: FAIL — state extraction runs `AVG(<expr referencing __STATE_R1__>)` but `__STATE_R1__` is not registered, so DataFusion errors ("table not found") at `fit`.

- [ ] **Step 3: Add `join_tables` to `build_state_tables`**

In `sql_transform/_state.py`, change the signature and build the FROM with cross joins:

```python
def build_state_tables(
    windows: list[WindowAgg],
    ctx: datafusion.SessionContext,
    table_name: str,
    join_tables: dict[str, pa.Table] | None = None,
) -> dict[str, pa.Table]:
```

Right after the `groups` are built and before the `tables` loop, register + assemble the FROM prefix:

```python
    join_tables = join_tables or {}
    for name, tbl in join_tables.items():
        ctx.from_arrow(tbl, name=name)
    from_sql = table_name
    for name in join_tables:
        from_sql += f" CROSS JOIN {name}"
```

Then use `from_sql` in both branch queries (replace `FROM {table_name}`):

```python
        if partition_cols:
            key_list = ", ".join(f'"{c}"' for c in partition_cols)
            sql = (
                f"SELECT {key_list}, {', '.join(value_exprs)} "
                f"FROM {from_sql} GROUP BY {key_list}"
            )
            ...
        else:
            sql = f"SELECT {', '.join(value_exprs)} FROM {from_sql}"
            ...
```

Scoped tables are one-row/global, so the cross join preserves cardinality and is a no-op when no aggregate references them.

- [ ] **Step 4: Pass `join_tables` from `fit`**

In `sql_transform/__init__.py`, change the `build_state_tables` call in `fit`:

```python
        own_state = build_state_tables(
            windows, ctx, "__THIS__", join_tables=inline.scoped_state
        )
```

- [ ] **Step 5: Run the capstone test, verify pass**

Run: `uv run pytest tests/test_diff_composition.py -k outer_aggregate_over_inlined -x -q`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — baseline (138 passed, 2 xfailed) + all composition tests.

- [ ] **Step 7: Commit**

```bash
git add sql_transform/_state.py sql_transform/__init__.py tests/test_diff_composition.py
git commit -m "feat: outer window aggregate over an inlined composed column"
```

---

## Notes for the implementer

- **Do not touch `src/` (Rust).** Composition is pure fit-time rewrite + state merge; `InferFn` needs no change. If you think it does, re-read the spec's "Rust `InferFn`: unchanged".
- **`WindowAgg` matches by node identity** — never re-parse the SQL between `find_window_aggregates` and `rewrite_sql`. `inline_references` mutates the tree first; run `find_window_aggregates` after.
- **Don't dedup repeated references** (Task 1) — correctness first; two copies of the same frozen state is fine.
- **Ponytail:** each non-trivial guard already leaves a runnable check (the tests). Don't add fixtures/frameworks; the differential `check`/`_rows_equal` helpers already exist.
