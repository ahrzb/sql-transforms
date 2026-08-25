# Transformers as UDAFs, v0: just run the pipeline

**Date:** 2026-07-29
**Status:** loop 5 of SQLProjection; designed in dialogue with AmirHossein.
**Deliberately deferred** (AmirHossein's scoping): desugaring transformers
into SQL windows/applies (that is the *optimization* pass, soon), and the
`sklearn.*` curated-semantics namespace (needs an index of sklearn
functions). This loop runs the actual sklearn objects.

## Surface

A window call whose function name is **not in DuckDB's function catalog**
resolves to a transformer: first from an explicit registry, then from the
caller's Python scope (the `FROM df` replacement-scan idiom):

```python
embed = Pipeline([("sc", StandardScaler()), ("pca", PCA(2))])
p = SQLProjection(
    "SELECT embed(* EXCLUDE (label, country)) OVER (PARTITION BY country) AS e,"
    " age + 1 AS a1 FROM __THIS__",
    this_model=Row,                      # transformers={"embed": ...} also accepted
).fit(train)
out = p.transform(table)                 # NEW: batch apply (pa.Table -> pa.Table)
```

- Lookup order: `transformers=` dict beats caller scope (locals then globals,
  frames walked past our own library code). The oracle-catalog guard means a
  scope variable can never shadow a real SQL function. Unknown name + no
  match → named refusal with a hint. Captured objects are snapshotted at
  construction and exposed as `p.transformers`.
- A transformer object is accepted iff it has `fit` and `transform`
  (duck-typed; sklearn `clone` used when available, else `copy.deepcopy`).
- Arguments: one bundle — a plain column, a `struct_pack(...)`, or `*` /
  `COLUMNS(...)` with the loop-4 modifiers (schema mode required for
  star forms; the bundle rule from the design dialogue: one call, one named
  struct, model order).
- `PARTITION BY` = per-group fitting (a clone per group, NULL keys are a
  group). `OVER ()` = one global fit. ORDER BY / frames on transformer
  windows: named refusals.
- v0 restrictions, each a named refusal: transformer calls only in the
  final level of a chain; only as a top-level select item (no nesting the
  result in an expression); no transformer inside FILTER/keys/aggregates.

## The plan: a second step kind

`FitStep` grows `kind: "sql" | "fit"` (sql default) plus, for fit steps,
`transformer` (registry name), `features` (bundle field names, model order),
and `keys` (partition key column names in the level table). The level table
materializes the bundle as `__cf_f{n}` columns and keys as `__cf_k{m}` —
same mechanism as windows and keys today. The executor grows one branch:

```
__CF_LEVEL_0__   sql  reads (__THIS__,)   SELECT <items>, <f-cols>, <k-cols> FROM __THIS__
__CF_PARAMS_1__  fit  reads (__CF_LEVEL_0__,)  transformer=embed keys=(country,) features=(age, fare)
```

A fit step's "params table" is not SQL-shaped in v0 (no wide extraction —
that is the deferred optimization): `fit()` stores fitted clones on the
projection, keyed by partition-key tuple, in `p.fitted[step.name]`. The
plan's DAG order and `reads` stay exactly as loop 3 defined them.

## Serving/apply: batch `transform(table)`

`serving_sql` cannot express an opaque sklearn apply, so v0 adds a **batch**
apply path (row-at-a-time `infer` through Confit stays NotImplemented):

- In `serving_sql`, each transformer item is replaced by helper columns:
  its bundle features (`__cf_tf{i}_f{n}`) and its partition keys
  (`__cf_tf{i}_k{m}`), all row-wise expressions the SQL layer can compute.
- `transform(table)` runs `serving_sql` through DuckDB (threads=1, params
  registered), then for each transformer item: group rows by the key tuple,
  look up the fitted clone (unseen group → all-NULL output, the exact-join
  policy), call `.transform` on the feature block, splice the result in at
  the item's position under its alias, drop helpers.
- Transformer output: sklearn returns an (n, k) array → a
  `FixedSizeList(float64, k)` column in v0 (named struct fields are part of
  the deferred optimization; k is discovered at fit, which is why the output
  schema is fit-time knowledge — documented).

## The gate

The round-trip invariant splits by column kind, both halves oracle-anchored:

- SQL columns: DuckDB differential, unchanged, bit-exact.
- Transformer columns: `p.fit(train).transform(train)` must equal an
  **independent reference path** — pandas groupby over the same keys,
  `clone(est).fit(X_g).transform(X_g)` per group — allclose with tight
  tolerance (sklearn's own numerics are BLAS-order dependent; bit-exactness
  was never on the table for this stratum, per the DRAFT-20 sizing).
- Refusal tests for every new named refusal; corpus untouched this loop
  (transformers are not SQL statements; a transformer corpus can come with
  the desugar pass).

## Out of scope, explicitly

Desugar-to-SQL (scaler family etc.), wide params extraction, `sklearn.*`
namespace and its semantics contracts, matvec/tree_ensemble instructions,
Confit wiring of applies, transformer calls in non-final levels, struct
access into transformer results, the DRAFT-20 leakage question.
