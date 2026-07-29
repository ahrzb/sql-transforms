# sql-transform

`SQLProjection` — projections over `__THIS__`, fit once, serve row-at-a-time.

The **fit half works today**: window aggregates over `__THIS__` are
*marginalized* — computed once over training data, materialized into params
tables, and the SQL rewritten to join them instead of recomputing:

```python
from sql_transform import SQLProjection

p = SQLProjection(
    "SELECT (age - avg(age) OVER (PARTITION BY country))"
    " / stddev_samp(age) OVER (PARTITION BY country) AS age_z FROM __THIS__"
).fit(train)   # train: pyarrow.Table

p.serving_sql
# SELECT (__cf_t.age - __cf_p0.__cf_a0) / __cf_p0.__cf_a1 AS age_z
# FROM __THIS__ AS __cf_t
# LEFT JOIN __CF_PARAMS_0__ AS __cf_p0
#   ON (__cf_t.country IS NOT DISTINCT FROM __cf_p0.country)
p.params
# {"__CF_PARAMS_0__": <pyarrow.Table: country, __cf_a0, __cf_a1>}
```

The serving half (`infer`/`infer_batch` through [Confit](../confit)) is a later
loop and still raises `NotImplementedError`.

## The contract

Strict projection: `SELECT <exprs> FROM __THIS__`. Everything else — WHERE,
GROUP BY, joins, subqueries, CTEs, running windows, order-sensitive
aggregates — is refused at construction with a named error. Serve-or-refuse,
no third mode.

Parsing is DuckDB's own (`json_serialize_sql` / `json_deserialize_sql`): the
oracle's grammar and the oracle's printer, no second SQL dialect anywhere.

The correctness gate is differential: for every accepted projection, the
original SQL over training data must be **bit-exact** with `serving_sql`
joined against the fitted params — both executed by DuckDB (single-threaded:
DuckDB's parallel float window aggregation is not bit-deterministic even
against itself). NULL partition keys are one partition and join back via
`IS NOT DISTINCT FROM`; join multiplicity is provable, not tested.

See the design spec:
[2026-07-29-sql-projection-marginalization-design.md](../../docs/superpowers/specs/2026-07-29-sql-projection-marginalization-design.md).
