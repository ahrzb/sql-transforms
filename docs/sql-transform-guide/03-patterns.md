# Patterns

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

| you want | write |
| --- | --- |
| learn from training, apply to live | `FROM __THIS__ t, (SELECT ... FROM __FIT__) s` |
| the same, per group | `LATERAL m((FROM __FIT__ WHERE k = g.k), (FROM __THIS__ WHERE k = g.k))` |
| reuse a transform | `FROM m(__FIT__, __THIS__)` |
| chain two | `FROM b(a(__FIT__, __FIT__), a(__FIT__, __THIS__))` |
| a static lookup | name it; it resolves from your frame and is captured by value |
| an sklearn estimator | `Transform.from_estimator(est, takes=..., returns=...)`, then `x_fit`/`x_transform` |
| refit on every batch, deliberately | `x_fit(...) FROM __THIS__` — you had to type `__THIS__` |
