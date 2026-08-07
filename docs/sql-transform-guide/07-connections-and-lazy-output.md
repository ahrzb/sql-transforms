# Connections and unexecuted output

<details>
<summary>Setup for the examples on this page</summary>

```
>>> import pyarrow as pa
>>> import pyarrow.compute as pc
>>> from datetime import date
>>> from sql_transform.model import SQLTransform, Transform, run

>>> SALES = pa.table({
...     "store": ["S1", "S1", "S1", "S2", "S2", "S2"],
...     "price": [10.0, 20.0, 30.0, 100.0, 300.0, 500.0],
... })

>>> z = SQLTransform("""
...     SELECT t.store, round(t.price / s.m, 4) AS z
...     FROM __THIS__ t, (SELECT avg(price) m FROM __FIT__) s
...     ORDER BY t.store, z
... """)

```

</details>

A transform makes no hidden connection: pass yours, and it uses your catalog.

```
>>> import duckdb
>>> con = duckdb.connect()
>>> con.execute("CREATE TABLE uplift AS SELECT 1.5 AS factor")     # doctest: +ELLIPSIS
<...>
>>> boosted = SQLTransform(
...     "SELECT round(t.price * uplift.factor, 2) AS p FROM __THIS__ t, uplift "
...     "ORDER BY p",
...     connection=con,
... )
>>> boosted.captured             # `uplift` is your table, not a captured object
{}
>>> boosted.fit(SALES)(SALES).to_pylist()[:2]
[{'p': 15.0}, {'p': 30.0}]

```

`output="duckdb"` hands back a `DuckDBPyRelation` that has **not run**, so a
SQL→SQL chain never materialises in between:

```
>>> stage1 = SQLTransform(z.source, connection=con).set_output(transform="duckdb")
>>> stage2 = SQLTransform(
...     "SELECT round(sum(z), 4) AS total FROM __THIS__", connection=con
... ).set_output(transform="duckdb")

>>> stage1.fit(SALES)
Fitted(params=__param_0[1], instances=0)
>>> lazy = stage1.transform(SALES)
>>> type(lazy).__name__
'DuckDBPyRelation'
>>> lazy.columns                      # binding is eager; execution is not
['store', 'z']

>>> stage2.fit(lazy)
Fitted(params=none, instances=0)
>>> stage2.transform(lazy).to_arrow_table().to_pylist()
[{'total': 6.0}]

```

A relation belongs to the connection that built it — it cannot be handed to
another one, not even to a cursor. That is why lazy chaining means giving both
stages the same `connection=`, and why crossing them fails by name instead of
quietly.

Sharing has a price the implementation pays for you: two transforms on one
connection both bind `__THIS__` and both call a parameter `__param_0`. Eagerly
that is harmless, since each materialises before the next registers — but a
relation is not executed yet, so stage one would end up reading stage two's
tables. Every execution therefore registers under names of its own, and the
readable ones stay yours.
