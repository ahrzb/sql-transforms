# Transformer Refs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an authored `SQLTransform` reference a fitted sklearn transformer as an opaque `{ref}` in a t-string, invoked struct-in/struct-out, with nested threading `f(g(x))`, `transform`==`infer` parity.

**Architecture:** Python-only. The Rust engine (`Expr::Transform`, `resolve_transformers`) and the DataFusion `_transformer_udf` already handle single and nested opaque calls — reused unchanged. New work: extend the t-string front-end to accept a fitted transformer as a ref, and at fit() rewrite each ref's call into the one-`named_struct`-arg form the engines expect, deriving each transformer's in/out schema by probing `.transform` on the training batch (innermost-first for nested chains).

**Tech Stack:** Python 3.14 (PEP 750 t-strings), sqlglot (AST), pyarrow, datafusion, sklearn, pydantic; Rust via maturin (not modified).

## Global Constraints

- Differential parity is the oracle: `transform` (DataFusion) MUST equal `infer` (Rust) for every case. Never weaken a parity assertion.
- v0, pre-1.0: breaking API changes made directly, no compat shims.
- Out of scope (error clearly, do not implement): fusing/routing optimization; flat top-level output columns; mixed leaf+nested args in one call; **aggregates over a transformer's output** (needs inline DataFusion field access — deferred).
- Reserved placeholder names are `__COMPOSE_i__` (already emitted by `desugar_template`).
- A transformer ref is duck-typed: an object with both `feature_names_in_` and `transform`.
- Windows build/test note: `uv run maturin develop` after any Rust change (none here); run Python tests with `uv run pytest`.

---

## File Structure

- **Create** `sql_transform/_transformer_ref.py` — resolve transformer-ref placeholder calls: wrap leaf args into `named_struct`, derive in/out schema by probing, handle nesting innermost-first. One responsibility, isolated from the SQLTransform-inlining path in `_compose.py`.
- **Modify** `sql_transform/_compose.py` — `desugar_template` accepts a fitted transformer as an interpolation value; `Ref` gains `is_transformer`.
- **Modify** `sql_transform/__init__.py` — `fit()` splits refs, resolves transformer refs, threads the registry into `InferFn` and `run_batch`.
- **Modify** `sql_transform/_batch.py` — `run_batch` registers transformer UDFs (per-context) for the batch path.
- **Create** `tests/test_transformer_ref.py` — differential-parity tests.

---

### Task 1: Single transformer ref, end-to-end parity

**Files:**
- Modify: `sql_transform/_compose.py` (`Ref`, `desugar_template`)
- Modify: `sql_transform/_batch.py` (`run_batch`)
- Create: `sql_transform/_transformer_ref.py`
- Modify: `sql_transform/__init__.py` (`fit`)
- Test: `tests/test_transformer_ref.py`

**Interfaces:**
- Produces:
  - `sql_transform._transformer_ref.is_transformer(obj) -> bool`
  - `sql_transform._transformer_ref.resolve_transformer_refs(select: exp.Select, tfm_refs: dict[str, object], table: pa.Table) -> dict[str, tuple[object, pa.Schema, pa.Schema]]` — mutates `select` (wraps each leaf call's args into a `named_struct`); returns `{placeholder_name: (obj, in_schema, out_schema)}`.
  - `run_batch(rewritten_sql, table, state_tables, transformers=None)` where `transformers: dict[str, tuple[object, pa.Schema, pa.Schema]] | None`.
- Consumes: Part-1 `sql_transform._transformer_udf._transformer_udf(obj, in_schema, out_schema, name)`; `InferFn(sql, row_tables, static_tables, transformers=…)` where a transformers entry is `{name: (obj, out_schema)}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transformer_ref.py
"""Differential parity for fitted transformers referenced as {ref} in a t-string."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sklearn.preprocessing import StandardScaler

from sql_transform import SQLTransform

# The nameless-input warning is a known false positive (see test_transformer_udf).
pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names:UserWarning"
)


def _both_engines(t, test_df):
    """transform (DataFusion) and infer (Rust) as plain dicts; assert equal."""
    batch = t.transform(pa.Table.from_pandas(test_df)).to_pylist()
    infer = [r.model_dump() for r in t.infer_batch(test_df.to_dict("records"))]
    assert infer == batch, (infer, batch)
    return batch


def test_single_scaler_ref_parity():
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train)
    t = SQLTransform(
        t"SELECT {sc}(age, income) AS out FROM __THIS__"
    ).fit(pa.Table.from_pandas(train))

    test = pd.DataFrame({"age": [25.0, 35.0], "income": [2.5, 3.5]})
    batch = _both_engines(t, test)

    expected = sc.transform(test)
    got = np.array([[b["out"]["age"], b["out"]["income"]] for b in batch])
    assert np.allclose(got, expected)
```

- [ ] **Step 2: Run it — verify it fails**

Run: `uv run pytest tests/test_transformer_ref.py -q`
Expected: FAIL — `desugar_template` raises `TypeError` (interpolation isn't a SQLTransform) or `fit` doesn't wire the transformer.

- [ ] **Step 3: Extend `Ref` and `desugar_template` to accept a transformer**

In `sql_transform/_compose.py`, add a flag to `Ref` and a branch in `desugar_template`:

```python
@dataclass(frozen=True)
class Ref:
    transform: object          # a SQLTransform, or a fitted transformer if is_transformer
    frozen: bool               # True for {a.transform}; False for bare {a}
    expr_text: str
    is_transformer: bool = False
```

In `desugar_template`, before the final `else: raise TypeError`, add:

```python
        elif hasattr(v, "feature_names_in_") and hasattr(v, "transform"):
            ref = Ref(v, frozen=False, expr_text=item.expression, is_transformer=True)
```

(Keep the existing SQLTransform / `.transform` branches unchanged; the transformer branch precedes the `else`.)

- [ ] **Step 4: Add `_transformer_ref.py` (leaf resolution + schema probe)**

```python
# sql_transform/_transformer_ref.py
"""Resolve transformer-ref placeholder calls in a desugared SELECT.

A transformer ref desugars to a __COMPOSE_i__(arg...) call. Here we wrap a leaf
call's column args into a single named_struct (the struct arg the engines'
opaque callout expects), and derive the transformer's in/out schema by probing
.transform on the training batch. numpy lives here, mirroring _transformer_udf.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from sqlglot import exp


def is_transformer(obj: object) -> bool:
    return hasattr(obj, "feature_names_in_") and hasattr(obj, "transform")


def _find_call(select: exp.Select, name: str) -> exp.Anonymous:
    for n in select.find_all(exp.Anonymous):
        if str(n.this).upper() == name:
            return n
    raise ValueError(f"transformer ref {name} must be applied to columns, e.g. {{t}}(a, b)")


def _named_struct(cols: list[str]) -> exp.Anonymous:
    """named_struct('c0', c0, 'c1', c1, ...) keyed by column name."""
    args: list[exp.Expression] = []
    for c in cols:
        args.append(exp.Literal.string(c))
        args.append(exp.column(c))
    return exp.Anonymous(this="named_struct", expressions=args)


def _derive_schemas(
    obj: object, cols: list[str], table: pa.Table
) -> tuple[pa.Schema, pa.Schema]:
    """in_schema from the training columns; out_schema by probing .transform."""
    in_schema = pa.schema([(c, table.schema.field(c).type) for c in cols])
    x = np.column_stack([table.column(c).to_numpy(zero_copy_only=False) for c in cols])
    y = np.asarray(obj.transform(x))
    names = [str(n) for n in obj.get_feature_names_out()]
    if y.ndim != 2 or y.shape[1] != len(names):
        raise ValueError(
            f"cannot derive out_schema for {type(obj).__name__}: expected 2-D width "
            f"{len(names)}, got shape {y.shape}"
        )
    out_schema = pa.schema([(n, pa.from_numpy_dtype(y.dtype)) for n in names])
    return in_schema, out_schema


def resolve_transformer_refs(
    select: exp.Select, tfm_refs: dict[str, object], table: pa.Table
) -> dict[str, tuple[object, pa.Schema, pa.Schema]]:
    """Wrap each leaf transformer call's args into a named_struct and derive its
    schema. Returns {placeholder_name: (obj, in_schema, out_schema)}.
    (Nested calls: Task 2.)"""
    registry: dict[str, tuple[object, pa.Schema, pa.Schema]] = {}
    for name, obj in tfm_refs.items():
        call = _find_call(select, name)
        cols = [a.name for a in call.expressions]
        if len(cols) != len(call.expressions):
            raise ValueError(f"{name} args must be plain columns (nested calls: Task 2)")
        feat = [str(n) for n in obj.feature_names_in_]
        if set(cols) != set(feat):
            raise ValueError(
                f"{name} columns {cols} must match feature_names_in_ {feat}"
            )
        in_schema, out_schema = _derive_schemas(obj, cols, table)
        call.set("expressions", [_named_struct(cols)])
        registry[name] = (obj, in_schema, out_schema)
    return registry
```

- [ ] **Step 5: Register transformer UDFs in `run_batch`**

In `sql_transform/_batch.py`, add the import and parameter, and register per-context:

```python
from sql_transform._transformer_udf import _transformer_udf

def run_batch(rewritten_sql, table, state_tables, transformers=None):
    ...
    ctx = datafusion.SessionContext()
    ctx.from_arrow(numbered_table, name="__THIS__")
    for name, state_table in state_tables.items():
        ctx.from_arrow(state_table, name=name)
    for name, (obj, in_schema, out_schema) in (transformers or {}).items():
        ctx.register_udf(_transformer_udf(obj, in_schema, out_schema, name))
    ...
```

- [ ] **Step 6: Wire `fit()` / `transform()` in `__init__.py`**

Store the batch registry on the instance and split refs in `fit`:

```python
# __init__.py, in __init__:
        self._udf_specs: dict[str, tuple] = {}

# fit(), after this_model, before inline_references:
        from sql_transform._transformer_ref import resolve_transformer_refs
        tree = parse_and_validate(self._sql)
        ctx = datafusion.SessionContext()
        ctx.from_arrow(table, name="__THIS__")

        sqlt_refs = {n: r for n, r in self._refs.items() if not r.is_transformer}
        tfm_refs = {n: r.transform for n, r in self._refs.items() if r.is_transformer}
        self._udf_specs = resolve_transformer_refs(tree, tfm_refs, table)

        inline = inline_references(tree, sqlt_refs, ctx, table)
        windows = find_window_aggregates(tree)
        own_state = build_state_tables(windows, ctx, "__THIS__", join_tables=inline.scoped_state)
        self._state_tables = {**inline.scoped_state, **own_state}
        self._rewritten_sql = rewrite_sql(tree, windows, extra_marker_tables=tuple(inline.scoped_state))
        self._infer_fn = InferFn(
            self._rewritten_sql,
            row_tables={"__THIS__": this_model},
            static_tables=self._state_tables,
            transformers={n: (obj, out_s) for n, (obj, in_s, out_s) in self._udf_specs.items()},
        )
        return self
```

And in `transform()`:

```python
        return run_batch(self._rewritten_sql, table, self._state_tables, self._udf_specs)
```

- [ ] **Step 7: Build and run the test — verify it passes**

Run: `uv run maturin develop && uv run pytest tests/test_transformer_ref.py -q`
Expected: PASS (1 passed).

- [ ] **Step 8: Run the full suite (no regressions)**

Run: `uv run pytest -q`
Expected: PASS (prior 197 + 1 new).

- [ ] **Step 9: Commit**

```bash
git add sql_transform/_compose.py sql_transform/_batch.py sql_transform/_transformer_ref.py sql_transform/__init__.py tests/test_transformer_ref.py
git commit -m "feat: fitted transformer as {ref} in authored SQL (single, parity)"
```

---

### Task 2: Nested threading `f(g(x))` parity

**Files:**
- Modify: `sql_transform/_transformer_ref.py` (`resolve_transformer_refs`: innermost-first, nested arg handling)
- Test: `tests/test_transformer_ref.py`

**Interfaces:**
- Same `resolve_transformer_refs` signature; now also handles a call whose single argument is another transformer-ref call (leaves that arg intact instead of wrapping) and probes an outer transformer on the inner's materialized output.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_transformer_ref.py
from sklearn.decomposition import PCA

def test_nested_threading_parity():
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train)
    scaled = sc.transform(train)
    pca = PCA(n_components=1).fit(scaled)  # feature_names_in_ = sc.get_feature_names_out()
    t = SQLTransform(
        t"SELECT {pca}({sc}(age, income)) AS out FROM __THIS__"
    ).fit(pa.Table.from_pandas(train))

    test = pd.DataFrame({"age": [25.0, 35.0], "income": [2.5, 3.5]})
    batch = _both_engines(t, test)

    expected = pca.transform(sc.transform(test))
    out_names = [str(n) for n in pca.get_feature_names_out()]
    got = np.array([[b["out"][n] for n in out_names] for b in batch])
    assert np.allclose(got, expected)
```

- [ ] **Step 2: Run it — verify it fails**

Run: `uv run pytest tests/test_transformer_ref.py::test_nested_threading_parity -q`
Expected: FAIL — Task 1's `resolve_transformer_refs` treats the inner `__COMPOSE_j__(...)` call as a "plain column" arg and errors, or derives the outer schema wrong.

- [ ] **Step 3: Handle nesting innermost-first**

Replace the loop body of `resolve_transformer_refs` so each ref's inputs are resolved before it. For a call whose single arg is another transformer-ref call, feed the inner's materialized output (a `pa.Table` of the inner's `out_schema` columns) to the outer's probe, and do NOT wrap the arg. Compute inner outputs once and reuse:

```python
def resolve_transformer_refs(select, tfm_refs, table):
    registry = {}
    materialized: dict[str, pa.Table] = {}  # name -> inner output as a table (for outer probe)

    def call_arg_ref(call):
        """If the call's single arg is another transformer-ref call, return its name."""
        if len(call.expressions) == 1 and isinstance(call.expressions[0], exp.Anonymous):
            inner = str(call.expressions[0].this).upper()
            if inner in tfm_refs:
                return inner
        return None

    def resolve(name):
        if name in registry:
            return
        call = _find_call(select, name)
        obj = tfm_refs[name]
        inner = call_arg_ref(call)
        if inner is not None:
            resolve(inner)                       # innermost first
            in_tbl = materialized[inner]         # inner's output columns
            cols = [str(n) for n in obj.feature_names_in_]
            in_schema, out_schema = _derive_schemas(obj, cols, in_tbl)
            # arg stays the inner call; do not wrap
        else:
            cols = [a.name for a in call.expressions]
            feat = [str(n) for n in obj.feature_names_in_]
            if set(cols) != set(feat):
                raise ValueError(f"{name} columns {cols} must match feature_names_in_ {feat}")
            in_schema, out_schema = _derive_schemas(obj, cols, table)
            call.set("expressions", [_named_struct(cols)])
        registry[name] = (obj, in_schema, out_schema)
        materialized[name] = _materialize(obj, cols, table if inner is None else materialized[inner], out_schema)

    for name in tfm_refs:
        resolve(name)
    return registry
```

Add the materialize helper (produces the transformer's output as a pyarrow Table whose columns are `out_schema`, so an outer transformer can probe on real data):

```python
def _materialize(obj, cols, src: pa.Table, out_schema: pa.Schema) -> pa.Table:
    x = np.column_stack([src.column(c).to_numpy(zero_copy_only=False) for c in cols])
    y = np.asarray(obj.transform(x))
    arrays = [pa.array(y[:, i], type=out_schema.field(i).type) for i in range(len(out_schema))]
    return pa.table(arrays, schema=out_schema)
```

- [ ] **Step 4: Run the nested test — verify it passes**

Run: `uv run pytest tests/test_transformer_ref.py::test_nested_threading_parity -q`
Expected: PASS.

- [ ] **Step 5: Run the file + full suite**

Run: `uv run pytest tests/test_transformer_ref.py -q && uv run pytest -q`
Expected: PASS (Task 1 test still green; full suite +2).

- [ ] **Step 6: Commit**

```bash
git add sql_transform/_transformer_ref.py tests/test_transformer_ref.py
git commit -m "feat: nested transformer threading f(g(x)) with parity"
```

---

### Task 3: Non-float dtype + window-agg coexistence parity

**Files:**
- Test: `tests/test_transformer_ref.py`
- Modify (only if a test surfaces a bug): `sql_transform/_transformer_ref.py`

**Interfaces:** none new — exercises existing code on new inputs.

- [ ] **Step 1: Write the failing/regression tests**

```python
# append to tests/test_transformer_ref.py
from sklearn.preprocessing import OrdinalEncoder

def test_ordinal_encoder_ref_string_in_int_out():
    train = pd.DataFrame({"color": ["red", "green", "blue", "red"], "size": ["S", "M", "L", "M"]})
    enc = OrdinalEncoder(dtype=np.int64).fit(train)
    t = SQLTransform(
        t"SELECT {enc}(color, size) AS out FROM __THIS__"
    ).fit(pa.Table.from_pandas(train))

    test = pd.DataFrame({"color": ["blue", "red"], "size": ["L", "S"]})
    batch = _both_engines(t, test)

    expected = enc.transform(test)
    got = np.array([[b["out"]["color"], b["out"]["size"]] for b in batch])
    assert (got == expected).all()


def test_transformer_and_native_window_agg_coexist():
    # A native window agg over __THIS__ alongside a transformer call: proves the
    # two compose without either engine choking. No agg reads the transformer output.
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train[["age", "income"]])
    t = SQLTransform(
        t"SELECT {sc}(age, income) AS out, age / (MEAN(age) OVER ()) AS an FROM __THIS__"
    ).fit(pa.Table.from_pandas(train))

    test = pd.DataFrame({"age": [25.0, 35.0], "income": [2.5, 3.5]})
    batch = _both_engines(t, test)
    assert np.allclose([b["an"] for b in batch], test["age"].to_numpy() / train["age"].mean())
```

- [ ] **Step 2: Run them — verify they fail (or reveal a real bug)**

Run: `uv run pytest tests/test_transformer_ref.py -k "ordinal or coexist" -q`
Expected: PASS if the design holds; a FAIL here is a real signal — STOP and diagnose (do not weaken the parity assertion). If the failure is a Rust-engine divergence, add an xfail-on-rust (strict) test and request a BACKLOG ticket instead of fixing inline.

- [ ] **Step 3: Fix only if a test surfaced a real Python-side bug**

If (and only if) a Python-side defect is found (e.g. `pa.from_numpy_dtype` on the encoder's int dtype, or the window-agg rewrite walking into the `named_struct`), fix it minimally in `_transformer_ref.py` or the fit wiring. Otherwise no code change.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (+2 from this task).

- [ ] **Step 5: Commit**

```bash
git add tests/test_transformer_ref.py sql_transform/_transformer_ref.py
git commit -m "test: transformer ref non-float dtype + native window-agg coexistence"
```

---

## Self-Review

**Spec coverage:**
- Authoring (`{ref}` in t-string, leaf multi-col, nested) → Tasks 1–2. ✓
- Fit (no agg over transformer output; probe schemas inner-before-outer) → Tasks 1–2 (`resolve_transformer_refs`). ✓
- Serve single-pass (`transform` UDF via `run_batch`; `infer` via InferFn registry) → Task 1 Steps 5–6. ✓
- Schema derivation by probe, out_schema=natural-dtype invariant → `_derive_schemas` (dtype straight from probe, no coercion). ✓
- Errors: unfitted transformer (no `feature_names_in_` → not treated as ref / clear Rust build error); missing/mismatched input fields (`set(cols) != set(feat)`). ✓
- Non-goals errored/untouched: mixed leaf+nested (Task 1 Step 4 rejects non-column leaf args); agg-over-output not built. ✓
- Testing: single, nested, dtype, window-agg coexistence → Tasks 1–3. ✓

**Placeholder scan:** none — all steps carry runnable code/commands.

**Type consistency:** `resolve_transformer_refs` returns `{name: (obj, in_schema, out_schema)}` in both tasks; `run_batch`'s `transformers` param and `_udf_specs` use the same 3-tuple; `InferFn` gets the 2-tuple `(obj, out_schema)` projection. Consistent.
