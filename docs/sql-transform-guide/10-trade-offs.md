# Trade-offs

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

What you give up, against a hand-written sklearn transformer:

| cost | detail |
| --- | --- |
| **You debug generated SQL** | The residual is machine-printed and members are spliced inline. A binder error points at text you did not write, with no Python stack. |
| **Per-group fit ships the training set** | The `LATERAL` form cannot freeze once, so `params` is O(\|D\|). A dict of estimators is O(groups). Marginalization is what closes this, and it is not built. |
| **No row-at-a-time serving** | `compile()`/`Inference` is deferred, so serving means running DuckDB over a batch. For a single row a pickled sklearn pipeline is faster today. |
| **DuckDB at serve time** | It is the parser, the planner and the oracle. There is no engine-free artifact. |
| **The AST walk rests on an internal format** | `json_serialize_sql` has no stability promise; `_shapes_test.py` replays DuckDB's own corpus to notice it moving, which is detection, not prevention. |
| **Name resolution reads your frame** | Convenient, and implicit. Anywhere the name is not in the caller's frame — `clone`, a factory, a config loader — you must pass `captured=` yourself. |
| **`fit` returns `Fitted`, not `self`** | Deliberate, and a real protocol deviation. `Pipeline` does not care; a stricter meta-estimator or `check_estimator` will. |
| **Foreign transforms are all DOUBLE** | `takes`/`returns` are declared by hand because DuckDB has no `ANY` type, and a learned output width fails at fit rather than construction. |
| **No `y`** | A supervised transform carries its target as a column in the relation. |
| **Ordered frames over `__FIT__`** | Legal, and a live foot-gun outside a backtest. Chosen over refusing, because refusing overrules a join key the author wrote. |

What you get in exchange is the top of this document: one text for both halves,
an artifact that is data, and laws (`run` versus `fit`/`transform`) that a test
can check. Whether that is the right trade depends on how much your training
and serving paths have drifted, and how much SQL you want to own.
