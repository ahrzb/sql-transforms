# Transformer-ref Follow-ups (TASK-3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six TASK-3 acceptance criteria — remove a duplicate `.transform()` call at fit, replace two misleading errors with actionable ones, support ndarray-fit transformers, and document the surface.

**Architecture:** Four independent changes to the transformer-ref front-end, all in `sql_transform/_transformer_ref.py` and `sql_transform/_compose.py`, plus tests and a README section. No changes to `src/*.rs` and no maturin rebuild — the design deliberately synthesises `feature_names_in_` onto a copy so both engines keep aligning by name exactly as they do today.

**Tech Stack:** Python 3.14, sqlglot (AST), pyarrow, numpy, scikit-learn, pytest, DataFusion (batch oracle), Rust `InferFn` via pyo3 (native, untouched here).

## Global Constraints

- Design spec of record: `docs/superpowers/specs/2026-07-23-transformer-ref-followups-design.md`. Read it before starting.
- DataFusion is the parity oracle (decision-1). Native and codegen must match it bug-for-bug.
- Every test must be mutation-checked: break the mechanism it covers, confirm the test fails, restore. A test that has never failed proves nothing.
- **Never `git checkout` a file to undo a mutation while other work is uncommitted** — commit first, then mutate, then restore. This destroyed an uncommitted guard during TASK-2.
- Do not edit `sql_transform/_codegen/plan.py` or `tests/test_codegen_coverage.py` — another developer owns them.
- Do not modify `src/*.rs`. If a change appears to require it, stop and report.
- Found a native-engine parity bug? Write an `xfail(strict=True)` test and request a ticket. Never fix it inline.
- v0, no backward compatibility. Breaking API changes are fine and need no shims.
- Run tests with `uv run pytest`. Never run `cargo test` (fails in this environment with a pyo3 DLL error, unrelated to code).
- Baseline before starting: **527 passed, 9 skipped, 4 xfailed**. The 4 xfailed are pre-existing native container gaps; leave them alone.
- Land as a PR: `git push origin <branch>` then `gh pr create --body-file -`. Never `git push . <branch>:master`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `sql_transform/_transformer_ref.py` | Resolve `{tfm}(col)` refs, derive schemas, wrap args into `named_struct` | Modify — detection predicate, column binding, probe/materialise refactor, aggregate guard |
| `sql_transform/_compose.py` | Desugar t-strings into placeholders + ref map | Modify — unfitted-transformer error at interpolation time |
| `tests/test_transformer_ref.py` | Contract + parity tests for the ref surface | Modify — 9 new tests |
| `sql_transform/_compose_test.py` | Compose-path unit tests | Modify — 2 new tests |
| `README.md` | User-facing docs | Modify — new transformer-ref section |

Task order matters: Task 1 changes the predicate that Tasks 2 and 4 depend on. Tasks 3 and 5 are independent of each other.

---

### Task 1: Transformer detection and column binding (AC#5, AC#2b)

**Files:**
- Modify: `sql_transform/_transformer_ref.py:16-17` (`is_transformer`), `:104-119` (leaf binding in `resolve`)
- Modify: `sql_transform/_compose.py:61-67` (interpolation type dispatch)
- Test: `tests/test_transformer_ref.py`, `sql_transform/_compose_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `is_transformer(obj) -> bool` with the new `n_features_in_` predicate. Tasks 2 and 4 rely on it recognising ndarray-fit transformers.

**Background:** `n_features_in_` is set by `fit()` for both DataFrame and ndarray input and is absent until fitted. `feature_names_in_` is set *only* for DataFrame input. So `n_features_in_` means "has been fitted" and `feature_names_in_` means "was fitted with names".

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_transformer_ref.py`:

```python
def test_ndarray_fit_transformer_binds_positionally():
    # sklearn records feature_names_in_ only for DataFrame fit. An ndarray-fit
    # transformer has no names, so arguments bind positionally in call order.
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train.to_numpy())
    assert not hasattr(sc, "feature_names_in_")

    t = SQLTransform(t"SELECT {sc}(age, income) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    batch = _both_engines(t, train)
    expected = sc.transform(train.to_numpy())
    got = np.array([[b["out"]["age"], b["out"]["income"]] for b in batch])
    assert np.allclose(got, expected)

    # clone contract: the user's object must be left untouched
    assert not hasattr(sc, "feature_names_in_")


def test_ndarray_fit_arity_mismatch_raises():
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train.to_numpy())  # n_features_in_ == 2
    t = SQLTransform(t"SELECT {sc}(age) AS out FROM __THIS__")
    with pytest.raises(ValueError, match="bind positionally"):
        t.fit(pa.Table.from_pandas(train))


def test_named_fit_column_mismatch_still_raises():
    # The named path is unchanged: names are validated as a set.
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train)
    t = SQLTransform(t"SELECT {sc}(age) AS out FROM __THIS__")
    with pytest.raises(ValueError, match="must match feature_names_in_"):
        t.fit(pa.Table.from_pandas(train))
```

Add to `sql_transform/_compose_test.py`:

```python
def test_unfitted_transformer_raises_not_fitted_error():
    # Before: TypeError blaming the interpolation TYPE ("must be a SQLTransform"),
    # which hides the real cause. An unfitted transformer has .transform but no
    # n_features_in_, so we can name the actual problem.
    from sklearn.preprocessing import StandardScaler

    train = pa.table({"age": [10.0, 20.0], "income": [1.0, 2.0]})
    with pytest.raises(ValueError, match="not fitted"):
        SQLTransform(t"SELECT {StandardScaler()}(age, income) AS o FROM __THIS__").fit(train)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_transformer_ref.py::test_ndarray_fit_transformer_binds_positionally tests/test_transformer_ref.py::test_ndarray_fit_arity_mismatch_raises sql_transform/_compose_test.py::test_unfitted_transformer_raises_not_fitted_error -v`

Expected: all three FAIL. The two ndarray tests fail with `TypeError: interpolation {sc} must be a SQLTransform or its .transform, got StandardScaler` (because `is_transformer` currently requires `feature_names_in_`). The unfitted test fails because a `TypeError` is raised where `ValueError` is expected.

- [ ] **Step 3: Change the detection predicate**

In `sql_transform/_transformer_ref.py`, replace `is_transformer`:

```python
def is_transformer(obj: object) -> bool:
    """A FITTED sklearn-style transformer.

    Keys off `n_features_in_`, not `feature_names_in_`: sklearn sets
    `n_features_in_` on any successful fit, but `feature_names_in_` only when
    fitted on named data (a DataFrame). Gating on the latter would reject
    perfectly good ndarray-fit transformers. Absence of `n_features_in_` means
    "not fitted", which is what lets us give that its own error.
    """
    return hasattr(obj, "transform") and hasattr(obj, "n_features_in_")
```

- [ ] **Step 4: Add the unfitted error**

In `sql_transform/_compose.py`, in `desugar_template`, insert a branch immediately **before** the final `else: raise TypeError(...)`:

```python
        elif hasattr(v, "transform") and not hasattr(v, "n_features_in_"):
            # Transformer-shaped but unfitted. Without this the generic TypeError
            # below blames the interpolation's TYPE, which sends users looking in
            # entirely the wrong place.
            raise ValueError(
                f"interpolation {{{item.expression}}}: {type(v).__name__} is not "
                f"fitted -- call .fit(...) before referencing it"
            )
```

- [ ] **Step 5: Add positional column binding**

In `sql_transform/_transformer_ref.py`, inside `resolve`, replace the leaf-path block that currently reads:

```python
            cols = [a.name for a in call.expressions]
            feat = [str(n) for n in obj.feature_names_in_]
            if set(cols) != set(feat):
                raise ValueError(
                    f"{name} columns {cols} must match feature_names_in_ {feat}"
                )
```

with:

```python
            cols = [a.name for a in call.expressions]
            feat = getattr(obj, "feature_names_in_", None)
            if feat is None:
                # Fitted without names (ndarray). Names are METADATA -- they ride
                # the named_struct as Arrow field names and both engines align on
                # them. sklearn never recorded any, so synthesise them from the
                # call site. Order is the user's contract, exactly as it is when
                # calling sklearn directly; only arity is checkable.
                if len(cols) != obj.n_features_in_:
                    raise ValueError(
                        f"{name} takes {obj.n_features_in_} columns (fitted without "
                        f"names, so arguments bind positionally in call order), "
                        f"got {len(cols)}: {cols}"
                    )
                # copy.copy, never mutate: doc-8's clone contract. Shallow, so the
                # fitted state is shared rather than duplicated.
                obj = copy.copy(obj)
                obj.feature_names_in_ = np.array(cols)
            else:
                feat = [str(n) for n in feat]
                if set(cols) != set(feat):
                    raise ValueError(
                        f"{name} columns {cols} must match feature_names_in_ {feat}"
                    )
```

Add `import copy` to the top of the file, after `from __future__ import annotations`.

**Critical:** `obj` is rebound to the copy, so the copy is what gets stored in `registry` and passed to both engines. Do not move the `registry[...] = (obj, ...)` assignment above this block.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_transformer_ref.py sql_transform/_compose_test.py -v`
Expected: PASS, including all pre-existing tests in both files.

- [ ] **Step 7: Mutation-check the new tests**

Temporarily revert `is_transformer` to `hasattr(obj, "feature_names_in_") and hasattr(obj, "transform")`.
Run: `uv run pytest tests/test_transformer_ref.py::test_ndarray_fit_transformer_binds_positionally -v`
Expected: FAIL. Then restore the predicate by re-editing (do NOT `git checkout`).

Temporarily change `np.array(cols)` to `np.array(list(reversed(cols)))`.
Run: `uv run pytest tests/test_transformer_ref.py::test_ndarray_fit_transformer_binds_positionally -v`
Expected: FAIL (values swapped). Restore by re-editing.

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest -q`
Expected: `531 passed, 9 skipped, 4 xfailed` (527 + 4 new tests).

- [ ] **Step 9: Commit**

```bash
git add sql_transform/_transformer_ref.py sql_transform/_compose.py tests/test_transformer_ref.py sql_transform/_compose_test.py
git commit -m "feat(transformers): support ndarray-fit transformers, name the unfitted case — TASK-3

is_transformer now keys off n_features_in_ (set by any successful fit) rather
than feature_names_in_ (set only for DataFrame fit), so ndarray-fit
transformers are recognised. Their column names are synthesised from the call
site onto a copy.copy(), preserving doc-8's clone contract and leaving both
engines' name-alignment untouched -- no src/*.rs change needed.

An unfitted transformer now raises a ValueError naming fittedness, instead of
a TypeError blaming the interpolation type."
```

---

### Task 2: Probe once, materialise only when consumed (AC#1)

**Files:**
- Modify: `sql_transform/_transformer_ref.py:45-72` (replace `_derive_schemas` and `_materialize`), `:75-134` (`resolve_transformer_refs`)
- Test: `tests/test_transformer_ref.py`

**Interfaces:**
- Consumes: `is_transformer` from Task 1.
- Produces: `_probe(obj, cols, table) -> tuple[pa.Schema, pa.Schema, np.ndarray]` and `_table_from_probe(y, out_schema) -> pa.Table`. `_materialize` is deleted; nothing may reference it afterwards.

**Background:** Every leaf ref calls `.transform()` twice — once in `_derive_schemas` for the output schema, once in `_materialize` for a table only an *outer* ref's probe reads. A leaf with no outer discards that table. This is the fit-time half of "fuse at inference, stage at fit": fit is a staged cascade, and a leaf with no consumer has no next stage.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transformer_ref.py`:

```python
class _SpyTransformer:
    """Wraps a fitted transformer and counts .transform() calls."""

    def __init__(self, obj):
        self._obj = obj
        self.calls = 0
        self.feature_names_in_ = obj.feature_names_in_
        self.n_features_in_ = obj.n_features_in_

    def transform(self, x):
        self.calls += 1
        return self._obj.transform(x)

    def get_feature_names_out(self):
        return self._obj.get_feature_names_out()


def test_leaf_ref_probes_transform_once():
    # A leaf ref's materialised output is only ever read by an OUTER ref's probe.
    # With no outer, materialising it is a discarded .transform() call.
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    spy = _SpyTransformer(StandardScaler().fit(train))

    SQLTransform(t"SELECT {spy}(age, income) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    assert spy.calls == 1, f"expected a single probe, got {spy.calls}"


def test_nested_refs_probe_once_each():
    # The inner IS consumed, so it must still be materialised -- but from the
    # probe's own output, not a second .transform() call.
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train)
    scaled = pd.DataFrame(sc.transform(train), columns=sc.get_feature_names_out())
    inner = _SpyTransformer(sc)
    outer = _SpyTransformer(PCA(n_components=1).fit(scaled))

    SQLTransform(t"SELECT {outer}({inner}(age, income)) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    assert (inner.calls, outer.calls) == (1, 1), (inner.calls, outer.calls)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_transformer_ref.py::test_leaf_ref_probes_transform_once tests/test_transformer_ref.py::test_nested_refs_probe_once_each -v`
Expected: FAIL — `assert 2 == 1` for the leaf, `(2, 2) != (1, 1)` for the nested case.

- [ ] **Step 3: Replace `_derive_schemas` and `_materialize`**

In `sql_transform/_transformer_ref.py`, replace both functions with:

```python
def _probe(
    obj: object, cols: list[str], table: pa.Table
) -> tuple[pa.Schema, pa.Schema, np.ndarray]:
    """in_schema from `cols`; out_schema by probing .transform; plus the probe's
    own output `y`, so a caller that needs the materialised table can build it
    without running .transform a second time."""
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
    return in_schema, out_schema, y


def _table_from_probe(y: np.ndarray, out_schema: pa.Schema) -> pa.Table:
    """The probe's output as a pa.Table shaped like out_schema, so an outer
    transformer can probe on real data. Reuses `y` -- no second .transform."""
    arrays = [
        pa.array(y[:, i], type=out_schema.field(i).type) for i in range(len(out_schema))
    ]
    return pa.table(arrays, schema=out_schema)
```

- [ ] **Step 4: Rewire `resolve_transformer_refs`**

Two changes inside `resolve_transformer_refs`.

First, `materialized` currently doubles as the already-resolved guard. Leaves no longer populate it, so add a separate set. Replace:

```python
    materialized: dict[
        str, pa.Table
    ] = {}  # name -> this ref's output, for outer probes
```

with:

```python
    materialized: dict[
        str, pa.Table
    ] = {}  # name -> this ref's output; ONLY for refs an outer consumes
    resolved: set[str] = set()  # every processed ref -- `materialized` is now partial

    # Which refs are consumed as another ref's argument? Must be computed BEFORE
    # any resolution: resolve() rewrites call args into a named_struct, which
    # destroys the nested-call signal call_arg_ref() reads.
    consumed = {
        inner
        for n in tfm_refs
        if (inner := call_arg_ref(_find_call(select, n))) is not None
    }
```

**Note:** `consumed` must appear after `call_arg_ref` is defined. Place this block immediately after the `call_arg_ref` function definition, not before it.

Second, in `resolve`, change the guard and the tail. Replace:

```python
    def resolve(name: str) -> None:
        if name in materialized:
            return
```

with:

```python
    def resolve(name: str) -> None:
        if name in resolved:
            return
```

and replace the trailing two statements:

```python
        registry[name.lower()] = (obj, in_schema, out_schema)
        materialized[name] = _materialize(
            obj, cols, table if inner is None else materialized[inner], out_schema
        )
```

with:

```python
        registry[name.lower()] = (obj, in_schema, out_schema)
        resolved.add(name)
        if name in consumed:
            # Only an outer ref's probe reads this. A leaf has no next stage, so
            # building it would be a discarded table.
            materialized[name] = _table_from_probe(y, out_schema)
```

Then update the two `_derive_schemas` call sites to `_probe`, binding `y`:

```python
            in_schema, out_schema, y = _probe(obj, cols, in_tbl)
```

and

```python
            in_schema, out_schema, y = _probe(obj, cols, table)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_transformer_ref.py -v`
Expected: PASS, including the pre-existing `test_nested_threading_parity`.

- [ ] **Step 6: Verify `_materialize` is gone**

Run: `grep -rn "_materialize" sql_transform/ tests/`
Expected: no output.

- [ ] **Step 7: Mutation-check**

Temporarily change `if name in consumed:` to `if True:`.
Run: `uv run pytest tests/test_transformer_ref.py::test_leaf_ref_probes_transform_once -v`
Expected: FAIL (`assert 2 == 1`). Restore by re-editing.

Temporarily change `_table_from_probe(y, out_schema)` back to a fresh `.transform` call via `_probe(obj, cols, src)[2]`.
Run: `uv run pytest tests/test_transformer_ref.py::test_nested_refs_probe_once_each -v`
Expected: FAIL (inner called twice). Restore by re-editing.

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest -q`
Expected: `533 passed, 9 skipped, 4 xfailed`.

- [ ] **Step 9: Commit**

```bash
git add sql_transform/_transformer_ref.py tests/test_transformer_ref.py
git commit -m "perf(transformers): probe .transform once per ref at fit — TASK-3

Every leaf ref called .transform() twice: once to derive out_schema, once to
materialise a table only an OUTER ref's probe reads. A leaf has no outer, so
that table was discarded. Probe once, reuse the probe's own output to build
the table, and only build it for refs another ref consumes.

{sc}(a, b): 2 -> 1 calls. {pca}({sc}(a, b)): 4 -> 2. _materialize is deleted.
The already-resolved guard moves to its own set, since materialized is now
deliberately partial."
```

---

### Task 3: Aggregate-over-output pre-check (AC#2a)

**Files:**
- Modify: `sql_transform/_transformer_ref.py:20-26` (`_find_call`)
- Test: `tests/test_transformer_ref.py`

**Interfaces:**
- Consumes: nothing. Independent of Tasks 1, 2, 4.
- Produces: `_in_window_agg(node) -> bool`.

**Background:** `AVG({sc}(age, income)) OVER ()` currently fails with `Error during planning: Invalid function '__compose_0__'. Did you mean 'power'?` — naming an internal placeholder the user never wrote and suggesting an unrelated function. Cause: at fit, `find_window_aggregates` freezes the aggregate by evaluating it in DataFusion before the transformer UDF is registered.

This is a real limit of the opaque mechanism, not a bug. Expressing it needs a subquery (materialise the output, then aggregate), and `parse_and_validate` rejects a subquery in `FROM`.

**Scope is critical:** the guard must live in `resolve_transformer_refs`, which only sees `tfm_refs`. A `SQLTransform` ref inlines to a plain scalar, so aggregating over *it* is ordinary flat SQL and legitimately works today.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_transformer_ref.py`:

```python
def test_aggregate_over_transformer_output_raises():
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train)
    t = SQLTransform(t"SELECT AVG({sc}(age, income)) OVER () AS m FROM __THIS__")
    with pytest.raises(ValueError, match="two-stage"):
        t.fit(pa.Table.from_pandas(train))


def test_aggregate_over_sqltransform_ref_still_works():
    # The guard must NOT overreach. A SQLTransform ref inlines to a scalar, so an
    # aggregate over it is ordinary flat SQL -- a shipped, documented capability.
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    inner = SQLTransform("SELECT age / MEAN(age) OVER () AS a FROM __THIS__").fit(
        pa.table({"age": [1.0, 2.0, 3.0]})
    )
    t = SQLTransform(
        t"SELECT AVG({inner.transform}(age)) OVER () AS m FROM __THIS__"
    ).fit(train)
    assert t.transform(train).column("m").to_pylist() == [12.5, 12.5, 12.5, 12.5]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_transformer_ref.py::test_aggregate_over_transformer_output_raises tests/test_transformer_ref.py::test_aggregate_over_sqltransform_ref_still_works -v`
Expected: the first FAILS (raises `ValueError` whose message is the DataFusion planning error, not matching "two-stage"). The second PASSES already — it pins existing behaviour.

- [ ] **Step 3: Add the guard**

In `sql_transform/_transformer_ref.py`, add above `_find_call`:

```python
def _in_window_agg(node: exp.Expression) -> bool:
    """Is `node` inside a window aggregate's argument?"""
    p = node.parent
    while p is not None:
        if isinstance(p, exp.Window):
            return True
        p = p.parent
    return False
```

Then in `_find_call`, inside the matching branch and alongside the existing `require_in_projection` call:

```python
def _find_call(select: exp.Select, name: str) -> exp.Anonymous:
    for n in select.find_all(exp.Anonymous):
        if str(n.this).upper() == name:
            require_in_projection(select, n, f"transformer ref {name}")
            if _in_window_agg(n):
                raise ValueError(
                    f"{name} output cannot feed a window aggregate: aggregating over "
                    f"transformer output is inherently two-stage (materialise the "
                    f"output, then aggregate it), which needs a subquery -- "
                    f"SQLTransform's single-SELECT surface has none. Aggregate over an "
                    f"input column instead, or use a SQLTransform reference, which "
                    f"inlines to a scalar."
                )
            return n
    raise ValueError(
        f"transformer ref {name} must be applied to columns, e.g. {{t}}(a, b)"
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_transformer_ref.py -v`
Expected: PASS.

- [ ] **Step 5: Mutation-check**

Temporarily change `if _in_window_agg(n):` to `if False:`.
Run: `uv run pytest tests/test_transformer_ref.py::test_aggregate_over_transformer_output_raises -v`
Expected: FAIL. Restore by re-editing.

Temporarily change `_in_window_agg` to `return True` unconditionally.
Run: `uv run pytest tests/test_transformer_ref.py -v`
Expected: many FAIL, proving the guard is reachable and scoped. Restore by re-editing.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: `535 passed, 9 skipped, 4 xfailed`.

- [ ] **Step 7: Commit**

```bash
git add sql_transform/_transformer_ref.py tests/test_transformer_ref.py
git commit -m "feat(transformers): actionable error for aggregate-over-transformer-output — TASK-3

AVG({sc}(x)) OVER () failed with \"Invalid function '__compose_0__'. Did you
mean 'power'?\" -- an internal placeholder the user never wrote, plus an
unrelated suggestion. The real cause is structural: fit freezes aggregate
state by evaluating it in DataFusion before the transformer UDF exists.

Guard is scoped to the opaque path. A SQLTransform ref inlines to a scalar, so
aggregating over one is ordinary flat SQL and keeps working -- pinned by test."
```

---

### Task 4: Contract and regression tests (AC#3, AC#4)

**Files:**
- Test: `tests/test_transformer_ref.py`

**Interfaces:**
- Consumes: `is_transformer` (Task 1), the probe refactor (Task 2), the aggregate guard (Task 3).
- Produces: nothing.

**Background:** Four of these pass on arrival — they pin behaviour that is currently correct but undocumented and easy to "fix" wrongly later. They still get mutation-checked.

- [ ] **Step 1: Write the tests**

Add to `tests/test_transformer_ref.py`:

```python
def test_mixed_leaf_and_nested_args_raises():
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train)
    scaled = pd.DataFrame(sc.transform(train), columns=sc.get_feature_names_out())
    pca = PCA(n_components=2).fit(scaled)
    t = SQLTransform(t"SELECT {pca}({sc}(age, income), age) AS o FROM __THIS__")
    with pytest.raises(ValueError, match="plain columns or another transformer ref"):
        t.fit(pa.Table.from_pandas(train))


def test_transformer_alongside_partitioned_window_agg():
    # Lock-in: a transformer callout and a PARTITION BY window agg coexist, and
    # both engines agree. Currently works; nothing must silently break it.
    train = pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0, 40.0],
            "income": [1.0, 2.0, 3.0, 4.0],
            "city": ["a", "a", "b", "b"],
        }
    )
    sc = StandardScaler().fit(train[["age", "income"]])
    t = SQLTransform(
        t"SELECT {sc}(age, income) AS o, AVG(age) OVER (PARTITION BY city) AS m "
        t"FROM __THIS__"
    ).fit(pa.Table.from_pandas(train))

    batch = t.transform(pa.Table.from_pandas(train)).to_pylist()
    infer = [r.model_dump() for r in t.infer_batch(train.to_dict("records"))]
    assert batch == infer
    assert [r["m"] for r in batch] == [15.0, 15.0, 35.0, 35.0]


def test_unfit_ref_is_fitted_once_globally_not_per_partition():
    # An unfit ref under an outer PARTITION BY is fitted ONCE over all rows; the
    # partitioning applies to the outer aggregate over its output. Matches
    # sklearn, where a Pipeline step is fitted once on all training data.
    # Per-group fitting is a separate feature (DRAFT-14), not this.
    train = pa.table({"age": [10.0, 20.0, 30.0, 50.0], "city": ["a", "a", "b", "b"]})
    norm = SQLTransform("SELECT age / MEAN(age) OVER () AS a FROM __THIS__")
    t = SQLTransform(
        t"SELECT AVG({norm}(age)) OVER (PARTITION BY city) AS m FROM __THIS__"
    ).fit(train)

    # global mean 27.5, NOT per-city 15.0 / 40.0
    assert t._state_tables["__STATE_R0__"].column("avg_age").to_pylist() == [27.5]
    got = t.transform(train).column("m").to_pylist()
    assert np.allclose(got, [0.5454545, 0.5454545, 1.4545455, 1.4545455])


def test_three_level_nesting_parity():
    # AC#4. Load-bearing after Task 2: the `consumed` set decides materialisation
    # by nesting position, and 3 levels is the only shape where a ref is
    # simultaneously consumed AND a consumer.
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    a = StandardScaler().fit(train)
    a_out = pd.DataFrame(a.transform(train), columns=a.get_feature_names_out())
    b = StandardScaler().fit(a_out)
    b_out = pd.DataFrame(b.transform(a_out), columns=b.get_feature_names_out())
    c = PCA(n_components=1).fit(b_out)

    t = SQLTransform(t"SELECT {c}({b}({a}(age, income))) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    batch = _both_engines(t, train)
    expected = c.transform(b.transform(a.transform(train)))
    out_names = [str(n) for n in c.get_feature_names_out()]
    got = np.array([[r["out"][n] for n in out_names] for r in batch])
    assert np.allclose(got, expected)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_transformer_ref.py -v`
Expected: PASS. All four pass immediately — three pin existing behaviour, and the 3-level test works because nesting already recurses.

- [ ] **Step 3: Mutation-check each**

For `test_three_level_nesting_parity`, break the Task 2 logic it exists to cover — temporarily change `consumed` to `set()`:
Run: `uv run pytest tests/test_transformer_ref.py::test_three_level_nesting_parity -v`
Expected: FAIL (`KeyError` on the missing materialised inner). Restore by re-editing.

For `test_unfit_ref_is_fitted_once_globally_not_per_partition`, temporarily change the assertion's expected value to `[15.0]`:
Run the test. Expected: FAIL, confirming the assertion reads real state. Restore.

For `test_transformer_alongside_partitioned_window_agg`, temporarily change expected `m` to `[10.0, 20.0, 30.0, 40.0]`:
Run the test. Expected: FAIL. Restore.

For `test_mixed_leaf_and_nested_args_raises`, temporarily relax the `isinstance(a, exp.Column)` check in `resolve` to `True`:
Run the test. Expected: FAIL (no `ValueError`). Restore by re-editing.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: `539 passed, 9 skipped, 4 xfailed`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_transformer_ref.py
git commit -m "test(transformers): contract and regression coverage — TASK-3

Four tests. Three pin behaviour that is correct today but undocumented and
plausibly 'fixable' in the wrong direction -- notably that an unfit ref under
an outer PARTITION BY is fitted ONCE globally (state holds the global mean
27.5, not per-city 15/40), matching sklearn's fit-once-per-Pipeline-step.

The 3-level nesting test is load-bearing rather than confirmatory: it is the
only shape where a ref is both consumed and a consumer, which is exactly what
the new materialisation logic keys on."
```

---

### Task 5: README documentation (AC#6, AC#5 doc half)

**Files:**
- Modify: `README.md`
- Modify: `sql_transform/_transformer_ref.py` (docstring only)

**Interfaces:**
- Consumes: the binding rules from Task 1.
- Produces: nothing.

**Background:** The README is 139 lines and contains no transformer-ref documentation. Two facts bite users immediately: output is a single Arrow struct column (not one column per feature), and binding is by name or position depending on how the transformer was fitted.

- [ ] **Step 1: Verify the documented facts before writing them**

Run:

```bash
uv run python -c "
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, pyarrow as pa
from sklearn.preprocessing import StandardScaler
from sql_transform import SQLTransform
train = pd.DataFrame({'age':[10.,20.,30.,40.], 'income':[1.,2.,3.,4.]})
sc = StandardScaler().fit(train)
t = SQLTransform(t'SELECT {sc}(age, income) AS scaled FROM __THIS__').fit(pa.Table.from_pandas(train))
out = t.transform(pa.Table.from_pandas(train))
print(out.schema.field('scaled').type)
print(out.flatten().schema.names)
"
```

Expected output:
```
struct<age: double, income: double>
['scaled.age', 'scaled.income']
```

If this does not match, stop — the docs would be wrong.

- [ ] **Step 2: Add the README section**

Append to `README.md`, after the existing usage section:

````markdown
### Referencing a fitted sklearn transformer

Interpolate a fitted transformer into a t-string and apply it to columns:

```python
sc = StandardScaler().fit(train_df)          # fit on a DataFrame -> records feature_names_in_
t = SQLTransform(t"SELECT {sc}(age, income) AS scaled FROM __THIS__").fit(table)
```

**Output is a single Arrow struct column**, not one column per feature:

```python
t.transform(table).schema            # scaled: struct<age: double, income: double>
t.transform(table).flatten().schema  # ['scaled.age', 'scaled.income']
```

Call `.flatten()` to get flat columns for an sklearn handoff.

**Column binding** depends on how the transformer was fitted:

| fitted with | `feature_names_in_` | binding |
|---|---|---|
| `fit(DataFrame)` | recorded | by **name** — call order is free, and is validated against the names |
| `fit(ndarray)` | absent | by **position**, in call order — only the count is checked |

With positional binding, `{sc}(income, age)` against a transformer fitted as
`[age, income]` silently swaps the features. Fit on a DataFrame when you can.

Aggregating over a transformer's output (`AVG({sc}(age)) OVER ()`) is not
supported — it is inherently two-stage and needs a subquery. Aggregate over an
input column, or use a `SQLTransform` reference, which inlines to a scalar and
composes with aggregates freely.
````

- [ ] **Step 3: Add the `is_transformer` docstring**

Already added in Task 1, Step 3. Verify it is present:

Run: `grep -A 3 "def is_transformer" sql_transform/_transformer_ref.py`
Expected: the docstring explaining the `n_features_in_` contract.

- [ ] **Step 4: Verify the README code samples run**

Run the Step 1 command again plus:

```bash
uv run python -c "
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, pyarrow as pa
from sklearn.preprocessing import StandardScaler
from sql_transform import SQLTransform
train = pd.DataFrame({'age':[10.,20.,30.,40.], 'income':[1.,2.,3.,4.]})
sc = StandardScaler().fit(train)
try:
    SQLTransform(t'SELECT AVG({sc}(age, income)) OVER () AS m FROM __THIS__').fit(pa.Table.from_pandas(train))
    print('ERROR: should have raised')
except ValueError as e:
    assert 'two-stage' in str(e); print('aggregate guard message confirmed')
"
```
Expected: `aggregate guard message confirmed`.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: `539 passed, 9 skipped, 4 xfailed` (unchanged — docs only).

- [ ] **Step 6: Commit**

```bash
git add README.md sql_transform/_transformer_ref.py
git commit -m "docs: document the transformer-ref surface — TASK-3

The README had no transformer-ref content at all. Adds the two facts users hit
first: output is a single Arrow struct column needing .flatten() for an sklearn
handoff, and binding is by name or by position depending on whether the
transformer was fitted on a DataFrame or an ndarray, with the order-safety
warning that follows from the positional case.

Deliberately does NOT document the ticket's obj.feature_names_in_ = names
workaround: it is no longer needed, and as written it is broken (the native
engine calls .tolist(), so a plain list raises)."
```

---

### Task 6: Land as a PR

**Files:** none.

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: a PR URL.

- [ ] **Step 1: Confirm the full suite is green**

Run: `uv run pytest -q`
Expected: `539 passed, 9 skipped, 4 xfailed`.

- [ ] **Step 2: Confirm no Rust file was touched**

Run: `git diff --name-only master...HEAD -- src/`
Expected: no output. If any `src/*.rs` file appears, stop and report — the design requires none.

- [ ] **Step 3: Rebase onto master**

```bash
git fetch origin master
git rebase origin/master
uv run pytest -q
```
Expected: still green after rebase.

- [ ] **Step 4: Push and open the PR**

Use `--body-file -` with a heredoc. Do **not** use `--body @-`, which sets the literal string `@-`.

````bash
git push origin task-3-transformer-ref-followups
gh pr create --base master --head task-3-transformer-ref-followups \
  --title "TASK-3: transformer-ref follow-ups" --body-file - <<'EOF'
Follow-ups on the shipped `{transform}(col)` authoring surface. Six ACs, no
`src/*.rs` change.

## AC#1 — one probe per ref at fit

Every leaf ref called `.transform()` twice: once to derive `out_schema`, once to
materialise a table only an *outer* ref's probe reads. A leaf has no outer, so
that table was discarded.

```
{sc}(a, b)            2 -> 1 transform() calls
{pca}({sc}(a, b))     4 -> 2
```

Measured by a spy that counts calls, so the regression is observable.
`_materialize` is deleted. The already-resolved guard moved to its own set,
since `materialized` is now deliberately partial.

## AC#2 — two misleading errors replaced

```
before:  ValueError: Error during planning: Invalid function '__compose_0__'. Did you mean 'power'?
after:   ValueError: __COMPOSE_0__ output cannot feed a window aggregate: aggregating over
         transformer output is inherently two-stage ... which needs a subquery

before:  TypeError: interpolation {sc} must be a SQLTransform or its .transform, got StandardScaler
after:   ValueError: interpolation {sc}: StandardScaler is not fitted -- call .fit(...) first
```

The first named an internal placeholder the user never wrote and suggested an
unrelated function; the second blamed the interpolation's type when the real
cause was fittedness.

The aggregate guard is scoped to the opaque path only. A `SQLTransform` ref
inlines to a scalar, so aggregating over one is ordinary flat SQL and keeps
working — pinned by a test.

## AC#5 — ndarray-fit transformers now work

`is_transformer` keyed off `feature_names_in_`, which sklearn sets only for
DataFrame fit, so ndarray-fit transformers were not recognised at all. It now
keys off `n_features_in_`, which any successful fit sets.

Column names are metadata — they ride the `named_struct` as Arrow field names.
When sklearn recorded none, they are synthesised from the call site onto a
`copy.copy()`, so the user's object is never mutated (doc-8 clone contract) and
both engines keep aligning by name exactly as before. That is why no Rust
change was needed: the native engine reads `feature_names_in_` at `InferFn`
build (`src/lib.rs:103`) and would otherwise have needed its own fallback.

Order remains the user's responsibility on the positional path, as it is when
calling sklearn directly; only arity is checkable.

## AC#3, AC#4 — contract and regression tests

Includes a lock-in for a previously undocumented semantic: an unfit ref under an
outer `PARTITION BY` is fitted **once globally** (state holds the global mean
27.5, not per-city 15/40), matching sklearn's fit-once-per-Pipeline-step.
Per-group fitting is a separate feature (DRAFT-14).

The 3-level nesting test is load-bearing rather than confirmatory as the ticket
assumed: after AC#1 it is the only shape where a ref is both consumed and a
consumer, which is what the new materialisation logic keys on.

## AC#6 — README

The README had no transformer-ref content. Adds the struct-column output and
`.flatten()` handoff, plus the name-vs-position binding table.

The ticket's `obj.feature_names_in_ = names` workaround is deliberately not
documented — AC#5 removes the need for it, and as written it is broken: the
native engine calls `.tolist()`, so a plain list raises.

## Verification

- Suite: 527 -> 539 passed, 9 skipped, 4 xfailed (pre-existing, unrelated).
- Every new test mutation-checked: break the mechanism, confirm the test fails.
- No `src/*.rs` change, no maturin rebuild.

Design spec: `docs/superpowers/specs/2026-07-23-transformer-ref-followups-design.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
````

- [ ] **Step 5: Verify the PR body landed**

Run: `gh pr view --json body --jq .body | head -20`
Expected: the real body text, not `@-`.

- [ ] **Step 6: Report to the PM**

Message Iris with the PR URL, the six ACs' status, the suite delta, and the two findings that changed the ticket (the broken `feature_names_in_ = [...]` workaround, and that AC#4 became load-bearing because of AC#1 rather than being confirmatory as the ticket assumed).
