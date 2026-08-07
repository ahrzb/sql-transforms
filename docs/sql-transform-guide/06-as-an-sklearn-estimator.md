# It is an sklearn estimator

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

`fit_transform`, a stateful `transform`, `get_params`/`set_params` so `clone`
works, and `set_output` so a downstream estimator gets arrays:

```
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> import pandas as pd

>>> frame = SALES.to_pandas()
>>> numeric = SQLTransform(
...     "SELECT round(t.price / s.m, 4) AS z "
...     "FROM __THIS__ t, (SELECT avg(price) m FROM __FIT__) s"
... )
>>> pipe = make_pipeline(numeric.set_output(transform="pandas"), StandardScaler())
>>> pipe.fit_transform(frame).shape
(6, 1)

```

`fit` returns the `Fitted` artifact rather than `self`. That is the currying,
and it costs nothing: `Pipeline` never reads what `fit` returned — it keeps the
object it called and asks *it* to transform later. So both spellings agree.

```
>>> t = SQLTransform(z.source)
>>> t.fit(SALES).transform(SALES).equals(t.transform(SALES))   # artifact, then estimator
True

```

On the training relation, `fit_transform` is exactly `run` — that is the
*freezing is faithful* law, not a coincidence:

```
>>> SQLTransform(z.source).fit_transform(SALES).equals(run(z, SALES))
True

```
