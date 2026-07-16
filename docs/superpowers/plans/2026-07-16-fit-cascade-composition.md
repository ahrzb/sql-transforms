# Recursive (fit-cascade) composition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an outer `SQLTransform` reference an **unfit** `SQLTransform` via `{a}(col)`, fitting `a` into the composite during `fit` (staged, sklearn-style), with arbitrary nesting `{a}({b}(x))`. Extends the shipped frozen path `{a.transform}(col)`.

**Architecture:** Generalize `inline_references` to walk nested placeholder calls bottom-up; for an unfit ref, **fit its definition into a name-scoped state** (`fit_into_scope`) over its input expression via `build_state_tables(join_tables=…)`, then inline exactly like the frozen path. Fit is topologically ordered by the AST nesting; inference is the same fused inline. Thread provenance tags through every inline from the start.

**Tech Stack:** Python 3.14, sqlglot, DataFusion (`datafusion`), the Rust `InferFn`, pyarrow, Pydantic v2, pytest, the differential harness `tests/differential.py`.

**Spec:** [docs/superpowers/specs/2026-07-16-fit-cascade-composition-design.md](../specs/2026-07-16-fit-cascade-composition-design.md)

## Global Constraints

- **v0, no back-compat.** Change `_compose.py` / `__init__.py.fit` signatures in place.
- **Frozen path unchanged.** `{a.transform}(col)` keeps working; unfit `{a}` is the new path; both mix in one query.
- **Clone contract:** `a` is **never** mutated or `.fit()`'d — its state is computed *into the composite's* scope, `a` stays unfit afterward (assert it).
- **Single-input / single-output** referenced transforms; output is a **scalar** (single-output auto-unwrap). Fan-out (multi-output struct+`unnest`) + multi-input + unfit-composite refs are **deferred** (explicit errors).
- **Acceptance = differential parity** — `transform` (DataFusion) == `infer` (Rust) for every case.
- **Provenance-ready pipeline (not implemented here):** keep all inlining centralized in `inline_references` (the single choke point), so the error-attribution BACKLOG item can later thread origin tags (ref + scope + authored SQL span) through *one* place. Full attribution is deferred — it needs Rust work: the rewritten SQL reaches `InferFn` as a *string*, so a runtime error tag must be carried through the interpreter and back out, which this plan does not build. Do NOT scatter inlining across call sites.
- Reuse shipped machinery: `build_state_tables(join_tables=…)` (Task 4 of composition), agg-over-expression, `rewrite_sql(extra_marker_tables=…)`, the frozen `inline_references`.

## File Structure

- `sql_transform/_compose.py` — generalize `inline_references` (recursive + unfit path); new `fit_into_scope`; provenance tags.
- `sql_transform/__init__.py` — `fit()` passes `ctx` + training table into `inline_references` (already builds them).
- `sql_transform/_compose_test.py` — unit tests (errors, clone contract, provenance).
- `tests/test_diff_composition.py` — differential parity (single/nested/mixed/outer-agg).

---

### Task 1: `fit_into_scope` + single flat unfit `{a}(col)`

An unfit `{a}(col)` (a single-in/out plain `SQLTransform`, `col` a plain `__THIS__` column) fits `a`'s definition into a fresh scope during the composite's `fit`, then inlines like the frozen path.

**Files:** Modify `sql_transform/_compose.py`, `sql_transform/__init__.py`. Test: `tests/test_diff_composition.py`.

**Interfaces:**
- Consumes: `parse_and_validate`, `find_window_aggregates` (`_sql.py`), `build_state_tables(windows, ctx, table_name, join_tables=None)` (`_state.py`), `rewrite_sql` (`_rewrite.py`), `Ref`/`_require_frozen_fitted`/`_single_col_arg`/`_frozen_expr` (`_compose.py`).
- Produces: `fit_into_scope(ref, input_expr, scope, deeper_states, ctx, training) -> (frozen_expr: exp.Expression, scoped_state: dict[str, pa.Table])`; `inline_references(select, refs, ctx, training)` (now takes ctx + training).

- [ ] **Step 1: Failing differential test** — append to `tests/test_diff_composition.py`:
```python
def test_unfit_single_reference_parity():
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    scaler = SQLTransform(  # NOT fitted
        "SELECT (age - AVG(age) OVER ()) / STDDEV(age) OVER () AS s FROM __THIS__"
    )
    composite = SQLTransform(t"SELECT {scaler}(age) AS s FROM __THIS__").fit(train)
    out = _parity(composite, train)
    assert abs(out[0]["s"] - ((10.0 - 25.0) / 12.909944487358056)) < 1e-9
    assert scaler._infer_fn is None  # clone contract: scaler still unfit
```
- [ ] **Step 2: Run, expect fail** — `uv run pytest tests/test_diff_composition.py::test_unfit_single_reference_parity -x` → FAIL: bare `{scaler}` currently raises `NotImplementedError("fit-cascade … not yet implemented")` (from the frozen slice).
- [ ] **Step 3: Implement `fit_into_scope` + wire the unfit branch.** In `sql_transform/_compose.py`:
```python
def fit_into_scope(ref, input_expr, scope, deeper_states, ctx, training):
    """Fit ref's DEFINITION into `scope`, its input remapped to input_expr,
    cross-joining deeper scopes' states. Returns (frozen_expr, {scope: state})."""
    tree = parse_and_validate(ref.transform._sql)
    if len(tree.expressions) != 1:
        raise ValueError("referenced transform must be single-output")
    proj = tree.expressions[0]
    proj = proj.this if isinstance(proj, exp.Alias) else proj
    inner_cols = {c.name for c in tree.find_all(exp.Column)}
    if len(inner_cols) != 1:
        raise ValueError(
            "referenced transform must read exactly one input column "
            "(multi-input not yet supported)"
        )
    inner_col = next(iter(inner_cols))

    # Remap inner's single __THIS__ column -> input_expr throughout the tree,
    # so its window aggregates are over input_expr (agg-over-expression).
    def remap(n):
        if isinstance(n, exp.Column) and n.name == inner_col and not (
            n.table and n.table.startswith("__STATE")
        ):
            return input_expr.copy()
        return n
    tree = tree.transform(remap)

    windows = find_window_aggregates(tree)
    own = build_state_tables(windows, ctx, "__THIS__", join_tables=deeper_states)
    # Scope: rename ref's produced state tables into `scope` and rewrite refs.
    scoped_state, rename = {}, {}
    for name, tbl in own.items():
        rename[name] = scope
        scoped_state[scope] = tbl
    frozen = rewrite_sql(tree, windows, extra_marker_tables=())  # str
    frozen_expr = sqlglot.parse_one(frozen).expressions[0]
    frozen_expr = frozen_expr.this if isinstance(frozen_expr, exp.Alias) else frozen_expr
    frozen_expr = frozen_expr.transform(
        lambda n: exp.column(n.name, table=scope)
        if isinstance(n, exp.Column) and n.table in rename else n
    )
    return frozen_expr, scoped_state
```
Generalize `inline_references(select, refs, ctx, training)`: for each placeholder, resolve its arg to `input_expr` (a plain `__THIS__.col` for Task 1); if `ref.frozen` use the existing frozen inline, else call `fit_into_scope(ref, input_expr, f"__STATE_R{i}__", scoped_state, ctx, training)` and inline the result. In `sql_transform/__init__.py.fit`, register `__THIS__` in `ctx` before calling `inline_references(tree, self._refs, ctx, table)` and pass the training `table`.
- [ ] **Step 4: Run** — `uv run pytest tests/test_diff_composition.py::test_unfit_single_reference_parity -x` → PASS; full suite `uv run pytest -q` (baseline **177 passed, 0 xfailed** + new) green.
- [ ] **Step 5: Commit** — `git commit -am "feat: fit-cascade single unfit reference {a}(col)"`

---

### Task 2: Nesting / chaining `{a}({b}(x))` + mixing frozen refs

Process references **bottom-up**: a placeholder's arg may contain deeper placeholders; inline those first (into `input_expr`), accumulating their scopes' states, then fit/inline the outer ref cross-joining them.

**Files:** Modify `sql_transform/_compose.py` (recursive `inline_references`). Test: `tests/test_diff_composition.py`.

**Interfaces:** Consumes Task 1's `fit_into_scope`. Produces the recursive resolution (arg → inlined `input_expr` + accumulated states).

- [ ] **Step 1: Failing tests** — append:
```python
def test_nested_unfit_refs_parity():
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    a = SQLTransform("SELECT (v - AVG(v) OVER ()) / STDDEV(v) OVER () AS s FROM __THIS__")
    b = SQLTransform("SELECT v * 2 AS d FROM __THIS__")
    composite = SQLTransform(t"SELECT {a}({b}(age)) AS z FROM __THIS__").fit(train)
    _parity(composite, train)  # b feeds a; a scales b(age)

def test_mixed_frozen_and_unfit_parity():
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    frozen = SQLTransform("SELECT v * 2 AS d FROM __THIS__").fit(train)
    unfit = SQLTransform("SELECT (v - AVG(v) OVER ()) / STDDEV(v) OVER () AS s FROM __THIS__")
    composite = SQLTransform(t"SELECT {unfit}({frozen.transform}(age)) AS z FROM __THIS__").fit(train)
    _parity(composite, train)
```
(Referenced transforms `a`/`b`/`unfit` read input column `v`; the composite applies them to `age`.)
- [ ] **Step 2: Run, expect fail** — nested placeholders not resolved (arg is a `__COMPOSE_j__(...)` call, not a plain column).
- [ ] **Step 3: Make `inline_references` recursive.** Resolve each placeholder's arg by first recursively inlining any placeholders inside it → `input_expr` (accumulating deeper `scoped_state`); then frozen-inline or `fit_into_scope` the ref over that `input_expr`, cross-joining the accumulated states (`join_tables=scoped_state`). Assign scope index by placeholder so nested refs get distinct `__STATE_R{i}__`. `fit_into_scope` already accepts `deeper_states` — thread the accumulated dict.
- [ ] **Step 4: Run** — both tests PASS; full suite green.
- [ ] **Step 5: Commit** — `git commit -am "feat: fit-cascade nesting/chaining + mixed frozen refs"`

---

### Task 3: Outer aggregate over the cascade

The outer `SQLTransform` taking its own window aggregate over an unfit-composed column.

**Files:** Test-only (the machinery from Tasks 1–2 + shipped `build_state_tables(join_tables=…)` already handles it). `tests/test_diff_composition.py`.

- [ ] **Step 1: Failing test** — append:
```python
def test_outer_aggregate_over_unfit_cascade_parity():
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    scaler = SQLTransform("SELECT (v - AVG(v) OVER ()) / STDDEV(v) OVER () AS s FROM __THIS__")
    composite = SQLTransform(
        t"SELECT {scaler}(age) / MAX({scaler}(age)) OVER () AS z FROM __THIS__"
    ).fit(train)
    _parity(composite, train)
```
- [ ] **Step 2: Run** — if it already passes (machinery reused), that confirms coverage; if it fails, the outer's `build_state_tables` isn't cross-joining the unfit scopes — fix `fit()` to pass the merged scoped states as `join_tables` to the outer's `build_state_tables` (same as the frozen capstone).
- [ ] **Step 3: Confirm/fix + full suite green.**
- [ ] **Step 4: Commit** — `git commit -am "test: outer aggregate over an unfit cascade"`

---

### Task 4: Error contract + clone contract

Lock the semantics with explicit tests.

**Files:** Modify `sql_transform/_compose_test.py`, `tests/test_diff_composition.py`.

- [ ] **Step 1: Failing/locking tests** — add: `{a}` on a **fitted** `a` raises `ValueError` ("already fitted; use {a.transform}"); `{a.transform}` on an **unfit** `a` raises `ValueError` (already exists — assert it still holds with the new path); a **multi-input** unfit ref (reads 2 columns) raises `ValueError`; a **multi-output** unfit ref (2 SELECT exprs) raises `ValueError`; **clone contract** — after `composite.fit`, the referenced unfit `a` is still unfit (`a._infer_fn is None`) and `a.fit(...)` afterward still works standalone.
- [ ] **Step 2: Run** — some guards already exist (frozen slice); add any missing to `_compose.py` (`fit_into_scope`'s single-in/out checks; the fitted-`{a}` branch). Confirm each RED→GREEN.
- [ ] **Step 3: Full suite green. Commit** — `git commit -am "test: fit-cascade error + clone contract"`

---

## Notes for the implementer

- **`fit_into_scope` is the crux** — it's `SQLTransform.fit`'s pipeline (find-windows → build-state → rewrite) applied to the ref's *definition* with input remapped + state scoped. Reuse the shipped functions; don't reimplement aggregation.
- **The frozen path already handles** a fitted ref's inline + outer-aggregate-over-inlined (composition Tasks 1/4). Unfit just adds the "fit `a`'s state into the composite first" step in front.
- **Do NOT mutate `a`** — parse `a._sql` (the definition), never call `a.fit()`. Assert `a._infer_fn is None` after.
- **Deferred (explicit errors, not this plan):** multi-output fan-out (`unnest({a}(col))`), multi-input refs, unfit-composite refs.
- **Provenance is out of scope for this plan** (it's the separate error-attribution BACKLOG item, which needs the Rust-side string-boundary design). The only obligation here is *readiness*: do all inlining in `inline_references` so tags can later thread through one place. sqlglot nodes carry `.meta` (survives `.copy()`/`.transform()`) — that's the eventual hook, don't invent a parallel structure.
