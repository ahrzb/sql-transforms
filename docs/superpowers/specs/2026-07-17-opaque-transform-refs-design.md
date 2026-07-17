# Opaque (non-native) transform refs — row→row composition — Design

> **⚠ STATUS: SUPERSEDED — SPLIT IN TWO (AmirHossein, 2026-07-17). Do NOT build from
> this doc.** It bundled two layers, and the *surface* half began dragging real
> engine complexity in for cosmetic reasons: the lowering wants a **derived table**
> (to bind the struct once and project its fields with clean names, since DataFusion
> won't let you alias `unnest` output or do inline field access), and supporting a
> derived table in the row engine means **adding a projection node inside the
> `RelNode` plan tree** (today `Plan { projection, input }` projects only at the top
> level). That's engine surgery bought by a naming limitation in the *other* engine.
>
> The work is now split — see BACKLOG "Opaque transform support — Part 1 → Part 2":
> - **Part 1 (first, active):** the Rust row engine can invoke an opaque
>   already-fitted Python transformer (marshal out → `.transform()` → marshal back).
>   Pure engine capability, independent of SQL expression. **A fresh spec is being
>   written for it** — use that, not this file.
> - **Part 2 (later):** the SQL/authoring surface — the `{ref}` row→row model,
>   multi-output native refs, lowering + output-column naming, the DataFusion-side
>   UDF, cross-engine parity. **The derived-table lowering question belongs here and
>   is being reconsidered.**
>
> Retained below as design reference for Part 2. Its own predecessor was the "mixed
> native+fallback pipeline" draft (`963eea6`), also superseded.

**Goal:** Let a `SQLTransform` reference an **opaque (non-native) transform** — a
fitted sklearn transformer, or a whole fitted sklearn `Pipeline`, for which we have
no engine expression — through the **same `{ref}` mechanism as native composition**.
One `SQLTransform` can then mix native and non-native transforms freely. This is
what makes **partial coverage shippable**: use native where we have it, opaque where
we don't, and later replace any opaque ref with a native `SQLTransform` **with no
change to the authoring surface**.

## The model: every transform is row → row

- A **`SQLTransform`** takes `__THIS__` (a row) and produces a row — its SELECT list.
- An **opaque object** (fitted sklearn transformer *or* whole `Pipeline`) is also
  row → row: array in, array out.
- **Composition is nesting**: `{pipeline}({features}(__THIS__))`.
- **`unnest(...)`** expands the final row (a struct) into output columns.

Native and opaque refs are uniform — one shape to reason about. Routing dissolves:
"which columns does this transform see" is answered by *what the inner transform
SELECTs*, not by a `ColumnTransformer`-style routing object.

## Authoring surface

```python
features = SQLTransform("""
  SELECT (age    - AVG(age)    OVER())/STDDEV(age)    OVER() AS age,
         (income - AVG(income) OVER())/STDDEV(income) OVER() AS income,
         balance
  FROM __THIS__
""")

pipeline = sklearn.pipeline.Pipeline([("pt", PowerTransformer()), ...]).fit(X)  # pre-fitted, opaque

final = SQLTransform(t"""
  SELECT unnest({pipeline}({features}(__THIS__)))
  FROM __THIS__
""").fit(train)
```

**Naming happens where naming belongs — the inner transform's SELECT list**, using
ordinary SQL `AS`. No `named_struct` and no `AS`-inside-a-call at the authoring
layer. (Both were considered and rejected: `named_struct(...)` is unbearably verbose
to author, and while sqlglot *does* parse `f(expr AS name, ...)` into `Alias` nodes —
verified empirically — row-passing makes the whole multi-arg-alignment question moot.)

## The rewrite generates the struct; the author never writes one

`unnest(...)` is the **authoring surface only**. It does *not* survive to the
engines — see "Verified empirically" below: DataFusion's `unnest` names its output
columns after the *expression text* (`__ref_0__(named_struct(Utf8("a"),…)).x`),
which is unusable as a feature name, and the alias in `unnest(…) AS z` is silently
ignored. Inline field access (`(f(…)).x`) is also unsupported. **The lowering is a
derived table binding the struct, with its fields projected under clean aliases:**

```sql
-- authored
SELECT unnest({pipeline}({features}(__THIS__))) FROM __THIS__

-- rewritten (what BOTH engines see), roughly
SELECT s.age AS age, s.income AS income, s.balance AS balance
FROM (SELECT __ref_pipeline__(named_struct(
              'age',     (age    - <fit-time mean>)/<fit-time std>,
              'income',  (income - <fit-time mean>)/<fit-time std>,
              'balance', balance)) AS s
      FROM __THIS__)
```

The outer projection's aliases come from `obj.get_feature_names_out()`. The
verbosity still exists — as *machine output*, which is where it belongs.
`__ref_i__` is a reserved-name function bound in the ref registry: valid SQL, and
unambiguous (reserved prefix + registry; refs can only enter via t-strings).

**Both engines share one rewritten SQL string** (`rewrite_sql` feeds `run_batch`
*and* `InferFn`), so this shape must be interpretable by both — which is what drives
the `InferFn` work below.

## Settled: dispatch by ref type

- **Native ref** (a `SQLTransform`) → **inline its expressions** (today's shipped
  composition). Its `__THIS__` binds to the outer's row, so its column references
  resolve against the outer's columns.
- **Opaque ref** (anything exposing `.transform`) → a **UDF call** bound to that
  object.

**The struct is materialized only at an opaque barrier.** Native→native composition
still inlines expressions with nothing materialized — the fuse-at-inference thesis
is intact. We build a struct only where we must hand values to Python.

## Settled: argument form (generalizes the shipped `{a}(col)`)

The argument is an expression evaluating to the ref's input:
- **Scalar arg** → the ref's single input column: `{a}(col)` — today's shipped form,
  unchanged.
- **Row/struct arg** → the ref's input row: `{a}(__THIS__)`, or a nested ref's output
  row.

`__THIS__` as an argument means "the current row."

## Settled: alignment at the opaque barrier

- **Input — positional.** The struct's **field order** (= the inner SELECT's order)
  determines the array's column order, because sklearn's `.transform(X)` on an array
  is inherently positional. Field *names* are for our side. When the object exposes
  `feature_names_in_`, check the field names against it and raise on mismatch — a
  safety net, not the alignment mechanism.
- **Output — names from `obj.get_feature_names_out()`**, which become the struct's
  field names and therefore the `unnest`ed column names. Called as a plain function
  on their object; needs **zero compliance work on our side**.

## Settled: fit constraint

**No unfit `SQLTransform` may be fit *through* an opaque ref.**
- A native ref **upstream** of an opaque ref fits on its own input — `features` fits
  its window aggregates on raw `__THIS__`, never crossing the barrier. **Allowed**
  (and is the motivating example).
- **Opaque refs must arrive pre-fitted.** We never fit them.

This excludes fit-cascade *across* the barrier, which would require materializing
training data forward through the Python call and staging the fit on the far side
(the "Approach C" materialize-forward path). **That is the main deferred item.**

## Now core (previously deferred)

- **Multi-output native refs.** `features` returns three columns → a struct. Today's
  composition is single-output-only. New rule: **single-output auto-unwraps to a
  scalar** (unchanged, shipped behavior); **multi-output returns a struct and must be
  `unnest`ed**; bare multi-output in a scalar position errors (as any struct does).
  struct + `unnest` is already shipped, so this is expressible with no new type work.
- **Multi-input refs.** Superseded by row-passing — a ref takes the whole row, so the
  shipped "multi-input ref → error" restriction lifts.

## Verified empirically (2026-07-17 probes — do not re-litigate)

- sqlglot **does** parse `f(expr AS name, …)` into `Alias` nodes (so the rejected
  `AS`-in-call surface was viable; row-passing supersedes it anyway).
- DataFusion 54 **accepts a struct-in / struct-out Python UDF** via `register_udf`,
  and it is **vectorized** — the UDF receives the whole batch's `StructArray`. So
  `transform` is **one sklearn call per batch**, not per row. (Nice bonus: the batch
  path is genuinely fast, not merely correct.)
- `SELECT s.x AS x, s.y AS y FROM (SELECT __ref_0__(named_struct(…)) AS s FROM …)`
  → clean column names `['x','y']`. **This is the lowering.**
- `SELECT (__ref_0__(…)).x` → **fails**: "Dot access not supported for non-string
  expr". Inline field access is out.
- `SELECT unnest(__ref_0__(…)) AS z` → alias **ignored**; columns keep the
  expression-text names. `unnest` is out as a lowering.

## Engines

- **`transform` (DataFusion):** register each opaque ref as a Python UDF
  (`ctx.register_udf`, struct in / struct out, vectorized per batch). **Verified
  working end to end** — no unknowns here.
- **`infer` (InferFn, Rust) — the work.** Three bounded pieces, two of which build
  directly on the shipped rich-type spine:
  1. **Python-callout UDF node** *(new — the headline)*: marshal the row's
     `Value::Struct` → a 1×N array (positional by field order) → call the registered
     object's `.transform` → marshal back into a `Value::Struct` named by
     `get_feature_names_out()`. pyo3 holds the object as a `Py<PyAny>`; the
     value↔Python marshalling reuses the shipped schema layer. This is the accepted
     "inefficient but works" serving cost — a Python call + array materialization
     **per row** (contrast the vectorized batch path).
  2. **Derived-table support** *(new, bounded)*: `plan.rs::build_table_factor`
     currently handles only `TableFactor::Table` and errors "Unsupported FROM clause"
     on a subquery. Needs `TableFactor::Derived` → plan the inner query → wrap in the
     **already-existing `RelNode::SubqueryAlias`**.
  3. **`named_struct(…)` → `Expr::Struct`** *(new, small)*: map the function form in
     `expr_build.rs` onto the **already-shipped** `Expr::Struct(Vec<(String, Expr)>)`.
  - **Already shipped, no work:** `Expr::FieldAccess` (eval + `infer_type`), and
    `plan.rs` already rewrites a `Column{table,name}` into a `FieldAccess` when the
    table part isn't a relation alias — exactly the `s.x` case.
- **Parity is by construction at the opaque node** — both engines call the *identical*
  Python object — so the differential test proves the *marshalling and stitching*
  agree, not the arithmetic.

## Deferred

- **Fit-cascade across an opaque barrier** (materialize-forward staged fit) — the
  main one, excluded by the fit constraint above.
- **Fitting opaque refs** — they arrive pre-fitted.
- **Our-transformer sklearn compliance** (`check_estimator`; our transformers
  composing into a *stock* sklearn `Pipeline`/`ColumnTransformer`) — already
  backlogged; the compose-in / hook-1 direction.
- **`ColumnTransformer`-style routing** — dissolved by this model (routing = what the
  inner transform SELECTs).

## Components

- **`sql_transform/_compose.py`** — ref dispatch (native → inline, opaque → UDF call),
  row-passing args (`{ref}(__THIS__)`), multi-output struct handling, the fit
  constraint's error.
- **`sql_transform/_rewrite.py`** — lower an opaque barrier to the derived-table
  shape: `SELECT s.<name> AS <name>, … FROM (SELECT __ref_i__(named_struct(…)) AS s
  FROM …)`, aliases from `get_feature_names_out()`.
- **`sql_transform/__init__.py`** — register opaque refs as DataFusion UDFs for
  `transform`; carry the ref registry into `InferFn`.
- **`src/expr_build.rs`** — `named_struct(…)` → the shipped `Expr::Struct`.
- **`src/plan.rs`** — `TableFactor::Derived` → plan inner query → wrap in the shipped
  `RelNode::SubqueryAlias`.
- **`src/` (interpreter)** — the Python-callout UDF node + registry binding
  (`Py<PyAny>`), reusing the shipped schema marshalling.
- **`tests/test_diff_opaque_refs.py`** (new) — differential tests.

## Testing

Differential parity (`transform` == `infer`) for:
- The motivating case: `unnest({pipeline}({features}(__THIS__)))` — a multi-output
  native ref feeding a pre-fitted opaque sklearn `Pipeline`, `unnest`ed to columns.
- A single opaque transformer (not a whole `Pipeline`):
  `unnest({pt}({features}(__THIS__)))`.
- A native ref consuming an opaque ref's output — proves the barrier composes in both
  directions, not just opaque-last.
- Multi-output native ref alone (`unnest({features}(__THIS__))`) — no opaque node.
- Single-output scalar auto-unwrap still works (`{a}(col)`) — shipped behavior
  unregressed.
- Errors: an **unfit** opaque ref → clear `ValueError`; an unfit native ref whose
  input comes *through* an opaque ref → clear `ValueError` (the fit constraint); bare
  multi-output ref in a scalar position → error.
- One hand-computed value for a single row (the marshalling is not self-evident).

## Next

**writing-plans.** Task sequence (Rust prerequisites first — they unblock the shared
rewritten-SQL shape, and each is independently testable against the existing
differential harness):
1. `named_struct(…)` → `Expr::Struct` (`expr_build.rs`) — small, on shipped types.
2. `TableFactor::Derived` → `RelNode::SubqueryAlias` (`plan.rs`) — unblocks the
   lowering; testable with a plain subquery, no opaque ref involved.
3. Multi-output native refs + row-passing args (`{ref}(__THIS__)`, struct output) —
   pure Python/compose, no opaque node yet.
4. Opaque-ref registry + the derived-table rewrite + the DataFusion Python-UDF path
   (`transform` green end to end; the Rust `infer` path still erroring is fine here).
5. The `InferFn` Python-callout node (`infer`) — the headline Rust work; parity
   `transform == infer` closes.
6. Fit-constraint errors + the remaining differential matrix.
