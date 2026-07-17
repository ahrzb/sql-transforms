# Transformer refs — opaque fitted transformers in authored SQL — Design

**Date:** 2026-07-18
**Status:** Approved — ready for implementation plan
**Builds on:** Part 1 (engine transformer callout, merge `4d5c85c`) — the Rust
`Expr::Transform` node and the DataFusion `_transformer_udf`, both reused **unchanged**.

## Goal

Let an authored `SQLTransform` reference an already-fitted sklearn transformer (or
fitted `Pipeline`) as an opaque `{ref}` in a t-string, invoked struct-in / struct-out.
Support **threading** — nested calls `f(g(x))` — across both engines, with
`transform` (DataFusion) and `infer` (Rust) differentially equal.

```python
sc, w2v, svd = ...  # fitted transformers
t = SQLTransform(t"SELECT {svd}({w2v}(inp)) AS out FROM __THIS__").fit(train)
t.transform(test_table)          # batch, DataFusion
t.infer({"inp": "some text"})    # row-at-a-time, Rust
```

## Non-goals (deferred)

- **Fusing / routing optimization** — stitching multiple independent call-sites of
  the *same* transformer into one batched invocation. Each call-site is its own
  opaque call for now.
- **Flat top-level output columns** — a transformer call yields a struct; the final
  projection is a struct column (or author-aliased). No inline `unnest` / derived-table
  lowering (the engine surgery the prior spec died on).
- **Mixed leaf + nested arguments** in one call (e.g. `{t}(a, {g}(b))`) — a call's
  argument is *either* input columns *or* a single nested call, not both.

## Authoring surface

Reuse the existing t-string composition front-end (`_compose.desugar_template`),
which already turns each interpolation into a `__COMPOSE_i__(...)` placeholder plus a
ref map. Extend a ref to also accept a **fitted transformer**, duck-typed by the
presence of `feature_names_in_` and `transform` (today a ref is only a `SQLTransform`
or its `.transform`).

Argument rules for a transformer call `__COMPOSE_i__(...)`:
- **Leaf call** — args are input column expressions, assembled into a `named_struct`
  keyed by column name. Requires the transformer's `feature_names_in_` to equal that
  name set (aligned by name, Part-1 semantics).
- **Nested call** — the single argument is another transformer call whose struct
  output *is* the input. The inner transformer's declared output field names must
  cover the outer's `feature_names_in_`.

Unlike a `SQLTransform` ref (which is *inlined* as a scalar expression at fit), a
transformer ref stays an **opaque callout node** — it is never inlined.

## Execution model

### Fit = staged state build (the only substantial new work)

State (window-aggregate scalars / lookup tables) can only be computed once the
columns it reads exist. When a native aggregate reads a transformer's output, fit
must materialize that output first — so fit runs in **stages**:

1. `desugar_template` → SQL with `__COMPOSE_i__` placeholders + ref map.
2. Build a dependency DAG: transformer call-sites, and window aggregates whose
   arguments (transitively) read a transformer's output. Topologically layer it.
3. Stage loop over training data, against a working DataFusion table seeded from
   `__THIS__`:
   - For each transformer call whose inputs are now materialized: derive `in_schema`
     from the working table, probe `.transform` over the batch to obtain `out_schema`
     (`get_feature_names_out()` names + observed dtype) and materialize its output
     columns (named to match any downstream `feature_names_in_`); append them to the
     working table.
   - Build and freeze any window-agg state now computable via the existing
     `build_state_tables`, and rewrite those aggregates to frozen state references.
4. Emit the rewritten plain-column SQL: `__COMPOSE_i__` → an opaque callout expression
   carrying `(obj, feature_names_in_, out_schema)`; aggregates → frozen state refs;
   collect the state tables.
5. Build `InferFn(rewritten_sql, transformers=<registry>, static_tables=<state>)`.

### Serve = single pass

All state is frozen, so no staging at serve time:
- **`transform`** — one DataFusion query. Register each transformer as a UDF
  (`_transformer_udf`, unchanged); nested calls evaluate inline; frozen state joined
  as today (via `run_batch`).
- **`infer`** — the Rust engine evaluates the nested `Expr::Transform` tree
  row-at-a-time. Confirmed to work unchanged: `Transform.arg: Box<Expr>` is evaluated
  through the general recursive `eval()` and `Transform` returns a `Value::Struct`, so
  an outer `Transform` consuming an inner one composes naturally
  ([src/expr.rs:371](../../../src/expr.rs)).

Both paths call the identical fitted objects, so parity holds by construction — the
native parts are already differentially covered.

## Schema derivation

Derived at fit by probing on the materialized batch — the user passes only the
transformer:
- `in_schema`: input dtypes read from the working table for the call's input columns.
- `out_schema`: names from `get_feature_names_out()`; dtype observed from the probe
  output. Homogeneous 2-D dense output assumed; raise clearly otherwise (an explicit
  `out_schema` override is the escape hatch). Honors the Part-1 out_schema =
  natural-dtype invariant.

Intermediate struct field names are set to the downstream transformer's
`feature_names_in_` so name-alignment holds across a nested chain. (The naming
conflict flagged during design is not reachable by the cases in scope here.)

## Errors

- Referencing an unfitted transformer → error at fit.
- Transformer input struct missing a `feature_names_in_` field → error (Part-1).
- Cyclic dependency in the call DAG → error at fit.

## Testing — differential parity is the oracle

For each case, assert `transform` (DataFusion) == `infer` (Rust) **and** == the real
sklearn object:
- Single transformer callout via `{ref}`.
- Nested `f(g(x))` (threading).
- A transformer's output feeding a native window aggregate (the staged-state case —
  the one path that exercises new fit machinery end to end).
- Non-float dtypes (e.g. OrdinalEncoder string→int).
