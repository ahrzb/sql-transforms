# Compose SQLTransforms via `{transform}(col)` references — Design

**Goal:** Let one `SQLTransform` reference **another fitted `SQLTransform` object**
inside its SQL, applied to a column, and combine the two into a single fused
transform that fits/transforms/infers correctly and is bit-identical between the
DataFusion batch path and the Rust inference path. This is the first slice of the
transformer-execution-model backlog item — the primitive our `Pipeline` and
sklearn composition are later built on.

Target surface (PEP 750 t-string; Python floor is now 3.14):

```python
scaler = SQLTransform(
    "SELECT (age - AVG(age) OVER ()) / STDDEV(age) OVER () AS s FROM __THIS__"
).fit(train)

composite = SQLTransform(
    t"SELECT {scaler.transform}(age) AS age_scaled FROM __THIS__"
).fit(train)
composite.transform(batch)     # DataFusion
composite.infer({"age": 40})   # Rust InferFn — identical value
```

## Scope

**In scope (this slice) — the frozen path only:**

- `{a.transform}(col)` — inline a **fitted** referenced transform's frozen
  transform, remapped onto `col`. No fitting of `a` happens.
- `{a}(col)` — the fit-cascade form. **Designed into the syntax, not implemented
  here**: a bare `{a}` raises an explicit "fit-cascade not yet implemented" error.
- **Single-input, single-output referenced transforms only.** `{a.transform}(col)`
  maps one input column to one output column (scaler / imputer shape). A referenced
  transform that reads >1 input column or emits >1 output column is a clear error
  pointing at the deferred slice.
- The **outer** may use the inlined column in any scalar position **and may take its
  own window aggregate over it** (`… / AVG({a.transform}(age)) OVER ()`). This
  requires generalizing window aggregates to accept an expression argument, not just
  a plain column (see "Supporting generalization").
- A referenced transform may be interpolated **more than once** in the same query,
  and **more than one** distinct transform may be referenced.

**Out of scope (deferred; designed-around, not built):**

- **Fit-cascade** (`{a}(col)` on an unfit transform) — the staged fit + nested-window
  problem. Deferred; bare `{a}` errors.
- **Multi-output fan-out** (OneHot → N columns) — output naming/placement +
  column-count-from-state.
- **Multi-input** referenced transforms — `{t}(a, b)` positional/named binding.
- **Partitioned referenced transforms.** A frozen inner with
  `… OVER (PARTITION BY city)` reads a second `__THIS__` column (`city`) and is
  therefore multi-input → excluded by the single-input rule above. Consequence: in
  this slice every referenced transform's frozen state is **global** (one-row,
  marker-keyed), which simplifies the state merge and join.

## Motivation

Post-fit, every `SQLTransform`'s rewrite is a **scalar expression over `__THIS__` +
frozen `__STATE__`** (window aggregates already resolved to state-table columns). So
nesting a fitted transform inside another is expression **inlining over frozen
state**: substitute the inner's scalar expression for the reference, remapped to the
outer's column. The result is one fused per-row expression — a single `InferFn` pass
at inference, no intermediate materialized. That is the serving thesis end-to-end,
and it is the primitive `Pipeline`/`ColumnTransformer` composition reduces to.

The frozen path is chosen first because it is dramatically cheaper: a frozen inner's
window aggregates are already constants, so the inline produces plain flat SQL with
no nested window aggregate, and the outer then fits + rewrites in one normal pass. No
staging, no cascade.

## Decisions locked (from prior brainstorming)

- **Reference forms encode fit intent.** `{a}` = fittable (cascade); `{a.transform}`
  = frozen reuse. The `.transform` at the call site makes "no fitting happens"
  unmissable. `{a.transform}` on an *unfit* object errors; bare `{a}` on a fitted
  object is ambiguous and errors (in this slice it errors regardless of fitted state,
  as fit-cascade is unimplemented).
- **References embed by Python object**, via t-string interpolation — an embedded
  `SQLTransform` arrives as the *real object*, not a stringified repr. (Registry-by-
  name is not the surface.)
- **Referenced transformers are definitions, never mutated.** The composite owns all
  fitted state; a reference is read-only on `a`. `{a.transform}` reads `a`'s existing
  frozen state; `a` is never re-fit or mutated. (sklearn's clone contract.)
- **Single-in / single-out** referenced transforms only (see Scope).

## Architecture & data flow

A composite is an ordinary `SQLTransform` whose SQL was authored as a t-string. All
composition work happens at `fit()` time, in a front-end pass that runs **before**
the existing find-windows → build-state → rewrite pipeline and reduces the composite
to a plain (non-composite) SELECT plus a set of pre-computed state tables. From that
point the existing pipeline runs unchanged.

```
SQLTransform(t"…"):
  desugar the Template -> plain SQL with synthetic placeholder calls + a ref map
    "SELECT {scaler.transform}(age) AS s FROM __THIS__"
      -> sql   = "SELECT __COMPOSE_0__(age) AS s FROM __THIS__"
      -> refs  = { "__COMPOSE_0__": Ref(transform=scaler, frozen=True) }
  store (sql, refs); str input -> refs = {} (behaves exactly as today)

composite.fit(train):
  tree = parse_and_validate(sql)                 # placeholder calls parse fine
  # ---- composition front-end (new; no-op when refs is empty) ----
  inline = inline_references(tree, refs)
    for each __COMPOSE_i__(argcol) node:
      validate: frozen & fitted; single-in; single-out; applied-to-a-column
      inner_expr = the referenced transform's single frozen projection expression
      remap inner's one __THIS__.<innercol> -> __THIS__.<argcol>
      rescope inner's __STATE*… -> __STATE_R{i}__…   (per-reference name-scope)
      node.replace(inner_expr)
    returns: mutated tree
           + scoped_state:  { "__STATE_R{i}__": <inner one-row state table> }
  # ---- existing pipeline, now over a plain tree ----
  windows = find_window_aggregates(tree)         # outer's own aggs; may be over exprs
  ctx: register __THIS__ = train, and every scoped_state table
  own_state = build_state_tables(windows, ctx, "__THIS__",
                                 join_tables=scoped_state)   # cross-joins scoped state
  state = scoped_state | own_state               # merged, name-scoped, no collisions
  rewritten = rewrite_sql(tree, windows,
                          extra_marker_tables=scoped_state.keys())  # + inner LEFT JOINs
  self._infer_fn = InferFn(rewritten, row_tables={"__THIS__": this_model},
                           static_tables=state)
```

**Key invariant (unchanged):** both engines run the *same* rewritten SQL against the
*same* merged frozen state — `transform` (DataFusion) and `infer` (Rust) agree on the
normal numeric path. Composition adds state tables and joins; it does not add a second
execution model. This is enforced by the differential harness.

### Worked example

Inner `scaler` (fitted), its stored `_rewritten_sql`:

```sql
SELECT (__THIS__.age - __STATE__.avg_age) / __STATE__.stddev_age AS s
FROM __THIS__ LEFT JOIN __STATE__ ON __STATE__.__state_marker__ = 0
```

Composite `t"SELECT {scaler.transform}(age) / AVG({scaler.transform}(age)) OVER () AS z FROM __THIS__"`:

1. Desugar → `SELECT __COMPOSE_0__(age) / AVG(__COMPOSE_1__(age)) OVER () AS z FROM __THIS__`
   (two interpolations of the same object → two placeholders, both `frozen=True`).
2. Inline both. Inner input col is `age`; call arg is `age` (identity remap here).
   Rescope `__STATE__` → `__STATE_R0__` and `__STATE_R1__` (one per reference):

   ```sql
   SELECT (__THIS__.age - __STATE_R0__.avg_age) / __STATE_R0__.stddev_age
        / AVG((__THIS__.age - __STATE_R1__.avg_age) / __STATE_R1__.stddev_age) OVER ()
     AS z FROM __THIS__
   ```
3. `find_window_aggregates` finds the `AVG(<expr>) OVER ()` (argument is an
   expression — see generalization). State extraction runs
   `AVG((age - avg) / std)` over `__THIS__ CROSS JOIN __STATE_R1__`, freezing it into
   the outer's own `__STATE__.avg_<hash>`.
4. `rewrite_sql` replaces the `AVG(...) OVER ()` node with `__STATE__.avg_<hash>` and
   appends LEFT JOINs for the outer's own `__STATE__` **and** marker joins for
   `__STATE_R0__`, `__STATE_R1__`.

Both engines then evaluate one flat scalar expression per row.

## Public API

```python
def __init__(self, sql: str | Template) -> None
    # str: behaves exactly as today (refs empty).
    # Template (PEP 750 t-string): desugared to placeholder SQL + a ref map,
    # both stored; composition is resolved at fit().
```

No new public methods. `fit`/`transform`/`infer`/`infer_batch` signatures are
unchanged; a composite is used exactly like any other `SQLTransform`. `from_file`
stays str-only (a file can't carry live object references).

**Reference discrimination (at desugar time), from the interpolation's `.value`:**

| Interpolated value                                         | Meaning                    | This slice |
|------------------------------------------------------------|----------------------------|------------|
| bound method `x.transform` where `x` is a `SQLTransform`   | frozen reuse `{x.transform}`| inline (require `x` fitted) |
| a `SQLTransform` instance                                  | fit-cascade `{x}`          | raise NotImplementedError |
| anything else                                              | misuse                     | raise TypeError |

Detection: `frozen = inspect.ismethod(v) and isinstance(v.__self__, SQLTransform)
and v.__func__ is SQLTransform.transform`.

## Module structure

```
sql_transform/_compose.py       (new) — the composition front-end
sql_transform/_compose_test.py  (new) — unit tests for desugar + inline + errors
sql_transform/_sql.py           (mod) — window aggregates over an expression arg
sql_transform/_state.py         (mod) — state_key over an expression; build_state_tables
                                        gains join_tables (cross-join scoped state)
sql_transform/_rewrite.py       (mod) — rewrite_sql gains extra_marker_tables
sql_transform/__init__.py       (mod) — __init__ accepts Template; fit() runs _compose
tests/test_diff_composition.py  (new) — differential parity (transform vs infer)
```

`_compose.py` owns everything composition-specific so the existing modules stay
focused; it depends on `_sql`/`_state` only for shared names, and imports
`SQLTransform` lazily (inside functions) to avoid a circular import with
`__init__.py`.

### `_compose.py` interface

```python
@dataclass(frozen=True)
class Ref:
    transform: "SQLTransform"   # the referenced object (read-only)
    frozen: bool                # True for {a.transform}, False for bare {a}
    expr_text: str              # interpolation source, for error messages

def desugar_template(template: Template) -> tuple[str, dict[str, Ref]]:
    """Turn a t-string into (plain SQL with __COMPOSE_i__(...) placeholders, ref map).
    Placeholder i is a synthetic identifier substituted for interpolation i; the
    surrounding literal text (including the `(col)` call) is kept verbatim.
    Raises TypeError if an interpolated value is neither a SQLTransform nor its
    .transform bound method."""

@dataclass(frozen=True)
class InlineResult:
    scoped_state: dict[str, pa.Table]   # "__STATE_R{i}__" -> inner one-row state table

def inline_references(select: exp.Select, refs: dict[str, Ref]) -> InlineResult:
    """Replace every __COMPOSE_i__(argcol) node in `select` with the referenced
    transform's frozen, remapped, name-scoped projection expression. Mutates `select`
    in place. Returns the merged scoped state tables (empty when refs is empty).
    Raises ValueError for: bare {a} (fit-cascade unimplemented); {a.transform} on an
    unfit object; a reference not applied to a single plain column; a referenced
    transform that is not single-input or not single-output."""
```

## State naming & scoping

- Existing scheme is unchanged: `state_table_name(())` → `__STATE__`;
  `state_table_name(("city",))` → `__STATE_BY_city__`; `state_key("AVG","age")` →
  `avg_age`.
- **Per-reference scope token:** reference `i` (by placeholder index, so repeated
  references get distinct scopes) maps the inner's global state table to
  `__STATE_R{i}__`. The token starts with `__STATE` so `rewrite_sql`'s "already a
  state column" guard skips the inlined refs, and is a valid bare SQL identifier
  (the illustrative `__STATE__@a` from the backlog is **not** used — `@` isn't a
  valid identifier). A referenced transform has ≤1 (global) state table in this
  slice, so one scoped name per reference suffices.
- Merged state = `scoped_state | own_state`. The scopes are disjoint by construction
  (`__STATE_R{i}__` vs the outer's `__STATE__`/`__STATE_BY_*`), so no collisions.

## Supporting generalization — window aggregate over an expression

Required by the "outer aggregates over the inlined column" done-criterion, and the
load-bearing new capability of this slice.

- **`_sql.py`:** `find_window_aggregates` currently rejects any window aggregate whose
  argument isn't a single plain column ([_sql.py:92](../../../sql_transform/_sql.py)).
  Relax to accept an arbitrary scalar expression argument. `WindowAgg` carries the
  argument **node** (`arg: exp.Expression`); `col` becomes the column name when the
  arg is a plain column, else `None`.
- **`_state.py`:** `state_key(fn, arg)` — plain column → `{fn}_{col}` (unchanged, keeps
  existing state/tests); non-column expression → `{fn}_{h}` where `h` is a short
  deterministic hash of the normalized argument SQL (collision-safe, and identical
  exprs dedup to one key). `build_state_tables` emits `AGG(<arg sql>) AS <key>` and,
  when `join_tables` is supplied, registers them and builds the extraction FROM as
  `__THIS__ CROSS JOIN <each scoped table>` so an aggregate over an inlined expression
  that references scoped state resolves. Scoped tables are one-row/global, so the
  cross join preserves cardinality and is a no-op when unreferenced.
- **`_rewrite.py`:** window replacement is already by node identity, so it handles an
  expression-argument window with no change; `rewrite_sql` gains
  `extra_marker_tables=()` and appends one `LEFT JOIN … ON <t>.__state_marker__ = 0`
  per scoped global-state table.
- **Rust `InferFn`:** unchanged. The aggregate is frozen into a state column at fit;
  inference only ever reads `__STATE*.…` columns, never computes an aggregate.

## Edge cases & errors

- **`str` (non-composite) input:** `refs` empty, `_compose` is a no-op; identical to
  today. All existing tests must stay green.
- **Zero-state referenced transform** (e.g. `SELECT age * 2 AS x`): inner has no state
  table; inline substitutes the pure scalar expression, no scope/join added.
- **Repeated reference:** each interpolation is its own placeholder/scope, so
  `{a.transform}(age)` used twice yields `__STATE_R0__` and `__STATE_R1__` (both
  copies of `a`'s frozen state). Correct, if mildly redundant; dedup is a later
  optimization, explicitly not done here.
- **Reference not applied to a column** (`{a.transform}` with no `(col)`, or
  `{a.transform}(x + 1)`): `ValueError` — "a referenced transform must be applied to
  a single input column, e.g. `{a}(age)`".
- **`{a.transform}` on unfit `a`:** `ValueError` — "referenced transform is not
  fitted; call `.fit(...)` before referencing `{a.transform}`".
- **Bare `{a}`:** `NotImplementedError` — "fit-cascade composition (`{a}(col)`) is not
  yet implemented; fit `a` and reference `{a.transform}(col)`".
- **Multi-input inner** (references >1 distinct `__THIS__` column, incl. any
  partitioned inner): `ValueError` naming the multi-input deferral.
- **Multi-output inner** (>1 SELECT expression): `ValueError` naming the fan-out
  deferral.
- **State-key hash collision** between two genuinely different aggregate expressions:
  `build_state_tables` already raises on a state-key collision with a distinct source;
  that guard extends to the expression case.
- **LEFT-join nullability:** global scoped state is marker-keyed and always matches, so
  composition does not newly exercise the tracked LEFT lookup-join nullability bug
  (Maintenance backlog); partitioned *outer* state uses the existing widened-nullable
  path.

## Testing

Primary suite is the differential harness (`tests/differential.py`): every case runs
through DataFusion (oracle) and the Rust `InferFn` and asserts values match. New
`tests/test_diff_composition.py`, plus `_compose_test.py` for front-end units.

**Frozen inline — parity:**
- Fit a `scaler` (single-in/out); `SELECT {scaler.transform}(age) AS s FROM __THIS__`
  fits + transforms + infers; parity holds; values equal hand-computed
  `(age-mean)/std`.
- Column **remap**: `{scaler.transform}(income)` applies `scaler` to `income`.
- Zero-state inner (`age * 2`) inlines.
- Inlined column in a scalar position: `{scaler.transform}(age) / __THIS__.age AS r`.
- **Outer aggregate over the inlined column** (the load-bearing case):
  `… / AVG({scaler.transform}(age)) OVER () AS z` — parity between `transform` and
  `infer`.
- Repeated + multiple distinct references in one query.

**Errors (each asserted on both engines / at build):**
- Bare `{a}` → NotImplementedError. `{a.transform}` unfit → ValueError.
- Reference not applied to a plain column → ValueError.
- Multi-input inner, multi-output inner, partitioned inner → ValueError.

**Non-mutation:** after `composite.fit(train)`, the referenced `scaler` is unchanged
(same fitted state object; still usable standalone) — asserts the clone contract.

## Non-goals

- No fit-cascade, no fan-out, no multi-input, no partitioned referenced transforms
  (all deferred; each has an explicit error).
- No new Rust `InferFn` ops — composition is pure fit-time rewrite + state merge.
- No `Pipeline`/`ColumnTransformer` classes yet — this is the underlying primitive
  they will use.
- No dedup of repeated references (correctness first; optimization later).
- No back-compat shims (v0): `state_key` / `WindowAgg` / `rewrite_sql` signatures
  change in place.

## Open decisions to confirm at review

1. **Scope token spelling** `__STATE_R{i}__` (index-based, repeat-safe). Alternative:
   scope by the interpolation's source text (`__STATE_R_scaler__`) — more readable but
   not collision-safe. Recommend index-based.
2. **Expression state-key** `{fn}_{hash}` (short hash of normalized arg SQL).
   Alternative: `{fn}_e{n}` positional. Recommend the hash (order-independent, dedups
   identical exprs).
3. **Where composition resolves** — at `fit()` (recommended: inner-fitted check and
   state extraction both live at fit) vs eagerly at `__init__`. Recommend `fit()`.
