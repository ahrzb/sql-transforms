# SQLProjection serving through Confit (DRAFT-22, the integration loop)

The last leg of DRAFT-22: `infer`/`infer_batch` stop raising and bind the
fitted artifact to Confit. Requires steps 1-3 (PRs #62, #63), both merged.

## What

One artifact, two bindings:

```python
p = SQLProjection(sql, transformers={...}).fit(train)
p.transform(table)         # DuckDB batch: register UDFs, run serving_sql
p.infer(row)               # Confit row-at-a-time: the same three pieces —
p.infer_batch(rows)        #   serving_sql + params tables + UDF objects —
                           #   passed to DuckDBInferFn(udfs=..., shape="map")
```

- `_serving_fn()` prepares the `DuckDBInferFn` lazily after fit and caches
  it; refit invalidates. `backend`/`boundary`/`output_model` delegate.
- The serving **row model derives from the training table's arrow schema**
  (real types, guaranteed), not from `this_model` (which may carry
  `object` fields and stays authoritative only for names/order at
  marginalize time). Unmappable columns become opaque fields — Confit
  accepts them unless the SQL references them. Consequence (v0 contract):
  serving rows have the training table's shape.
- `shape="map"`: every serving query is provably one-row-per-row (LEFT
  params joins only), so `infer` returns exactly one typed instance.

## Gate

`_serving_test.py`: for every admitted family (aggregates keyed/keyless,
transformer width-1 partitioned, width-2 global with struct bundle, author
UDFs, CTE chains, unseen-group NULL), `infer_batch(train rows)` equals
`transform(train)` value-for-value — the Confit binding vs the DuckDB
oracle binding of the same artifact. Dict rows and model rows agree.

## Deliberately out

- Pickle/serialization of the fitted projection (artifact = serving_sql +
  Arrow params + UDF objects; the pickle boundary is a later decision).
- Native `confit.udfs.*` entries (DRAFT-23) — swap into the same `udfs=`
  list when they land.
- Serving-row schema narrowed to referenced columns only.
