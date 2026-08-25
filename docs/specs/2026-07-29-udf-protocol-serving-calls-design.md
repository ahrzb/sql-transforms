# UDF protocol + in-place serving calls (DRAFT-22, step 1 of 3)

Full design: `backlog/drafts/draft-22 - UDF-protocol-and-Confit-externs-serving-transformers-without-special-support.md`.
This loop is the Python half — it fixes the serving_sql shape Confit's
extern loop (step 2) will consume, with zero Confit involvement.

## What changes

**Transformer calls rewrite in place, mid-expression included.** The v0
"top-level select item only" refusal lifts. A transformer window node
anywhere in a final-level item becomes a scalar call:

```sql
SELECT sc(age) OVER (PARTITION BY country) + 1 AS z FROM __THIS__
-- serving_sql:
SELECT (__cf_tf0(__cf_p0.__cf_est, __cf_t.age) + 1) AS z
FROM __THIS__ AS __cf_t
LEFT JOIN __CF_PARAMS_0__ AS __cf_p0
  ON (__cf_t.country IS NOT DISTINCT FROM __cf_p0.country)
```

**Per-group weights are params data.** Each transformer application gets
its own params join (never shared with aggregate joins — dedup is a later
optimization). Its `__CF_PARAMS_{n}__` table is `(join keys, __cf_est)`
where `__cf_est` indexes the fitted clones — produced by the `kind="fit"`
plan step (which is now *named* `__CF_PARAMS_{n}__`: all params tables are
params tables; some come from SQL collapse, some from fitting). Unseen
group -> LEFT JOIN miss -> NULL id -> NULL output; one NULL story.

**The UDF protocol** (`sql_transform/_udf.py`): declared name +
`takes`/`returns` in the engine type vocabulary (`"i1"|"i64"|"f64"|"str"`),
scalar `__call__` as THE semantic contract (plain Python values, None for
NULL), `apply_batch` over pyarrow whose default loops the scalar form
(amortizes the boundary without vectorizing the math — batch stays
bit-identical to row), `register(con)` binds to DuckDB. Determinism is a
documented contract. Scalar UDFs only — a custom aggregate is what
transformers already are.

- `PythonUDF(name, fn, takes, returns)` — any plain function, authorable
  in projection SQL (`SELECT myfn(age) ...`), resolved like transformers
  (registry then caller scope), guarded by the DuckDB function catalog.
  Exactly one return.
- `PythonTransform(name, instances, takes, returns)` — the fitted
  artifact: implicit leading nullable-i64 id; NULL id -> None; id missing
  from `instances` -> raise (broken artifact, not NULL); runtime check of
  each result against `returns`.

**Width comes from the fitted estimator** (probed once at fit). Width 1
registers as `DOUBLE` — a width-1 transform IS scalar-valued, so
mid-expression arithmetic just works. Width k registers as `DOUBLE[]` —
bare-item use yields a list column; arithmetic on it fails in DuckDB's
binder with a typed error at transform time.

**`SQLProjection.transform()` becomes register-and-run.** The helper-column
splicing block deletes; `TransformSpec` deletes (replaced by `UDFSpec(name,
step, transformer)`), `Marginalized.transforms` becomes `udfs` plus
`scalar_udfs` (author UDF names to register at fit and serving).

## Refusals (new and kept)

- kept: non-final-level transformer; ORDER BY / frames / FILTER / DISTINCT
  / IGNORE NULLS on a transformer window; bundle rules (one arg, named
  struct_pack, `fn(*)` = zero args); namespaced transformer.
- new: a transformer call inside a window aggregate's arguments; an
  unknown scalar function with no UDF in registry/scope; a UDF whose
  declared name or arity disagrees with the call site; a UDF inside a
  subquery (later loop); fitting a transformer on an empty training set.

## Gate

Unchanged in spirit: SQL columns bit-exact vs DuckDB at threads=1;
transformer columns vs the independent clone-per-group sklearn reference;
author-UDF queries round-trip exactly (fit+transform == original SQL with
the same UDF registered — the invariant extends verbatim since the UDF is
deterministic and identical on both sides).
