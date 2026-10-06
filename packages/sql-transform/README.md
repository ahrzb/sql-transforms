# sql-transform

sql-transform is the authoring package. You write a transform as SQL over
two tables: the fit data `__FIT__` and the request table `__THIS__`. Fit runs
the fit part of the SQL in DuckDB and writes params tables. The fitted
transform then reads only the request table and those params tables.

- `SQLTransform` is the general model. It composes and fits transforms, and it
  runs them in batch. Its SQL can change the number of rows.
- `SQLProjection` adds the row-local rule: each request gets one answer that
  does not depend on other requests. A fitted projection can compile to a
  [Confit](../confit) function.

The [authoring contract](docs/contract.md) is the specification of this
package. The [glossary](../../GLOSSARY.md) defines the terms.

To install the package, follow the [install steps](../../README.md#install)
in the repository README. `compile()` needs the `confit` package, which the
workspace installs.

## Write fit and request SQL

Write the relation between the fit data and the requests in the SQL:

```python
import pyarrow as pa
from sql_transform import SQLProjection

fit_data = pa.table({"v": [10.0, 20.0]})
requests = pa.table({"v": [12.0, 22.0]})

fitted = SQLProjection(
    "SELECT t.v - p.m AS d FROM __THIS__ t, "
    "(SELECT avg(v) AS m FROM __FIT__) p"
).fit(fit_data)

assert fitted.transform(requests).to_pydict() == {"d": [-3.0, 7.0]}
```

Fit freezes each fit query that does not depend on the requests into a params
table. Fit also rewrites supported correlated subqueries over `__FIT__` into
queries over params tables. The contract calls this decorrelation.

## Serve a projection with Confit

`compile()` builds a Confit function from the fitted projection. If Confit
does not serve a construct in the fitted SQL, `compile()` raises a refusal
that names it.

```python
fn = fitted.compile()
assert fn.infer_rows(requests.to_pylist()) == [{"d": -3.0}, {"d": 7.0}]
assert fn.infer_arrow(requests).to_pydict() == {"d": [-3.0, 7.0]}
```

The fitted projection has four public fields. Together they are the complete
artifact that Confit needs:

| Field | Contents |
|---|---|
| `fitted.sql` | The serving SQL. It reads the request table and the params tables, not the fit data. |
| `fitted.schema` | The schema of the request table |
| `fitted.params` | The params tables and the captured static tables, as Arrow tables |
| `fitted.udfs` | The UDFs that the serving SQL calls, by name |

## Compose general transforms

A `SQLTransform` can call other transforms as members. Each member takes a
fit relation and a request relation. This example fits `scale` on the output
of `shift`:

```python
from sql_transform import SQLTransform, run

shift = SQLTransform(
    "SELECT t.v - p.lo AS v FROM __THIS__ t, (SELECT min(v) AS lo FROM __FIT__) p")
scale = SQLTransform(
    "SELECT t.v / p.m AS z FROM __THIS__ t, (SELECT avg(v) AS m FROM __FIT__) p")
composed = SQLTransform(
    "SELECT * FROM scale(shift(__FIT__, __FIT__), shift(__FIT__, __THIS__)) s",
    captured={"shift": shift, "scale": scale},
)

fitted = composed.fit(fit_data)
assert fitted.transform(requests).to_pydict() == {"z": [0.4, 2.4]}

# run() binds both __FIT__ and __THIS__ to the same data, without fit.
assert fitted.transform(fit_data).equals(run(composed, fit_data))
```

`SQLTransform` also follows the scikit-learn transformer interface: `fit`,
`transform`, `fit_transform`, `get_params`, `set_params`, and `set_output`.

## Fit scikit-learn estimators in SQL

A projection can capture a scikit-learn estimator under a name, for example
`sc`. Then `sc_fit(...)` fits one instance of the estimator, and
`sc_transform(id, ...)` applies that instance. This example fits one scaler for
each group:

```python
from sklearn.preprocessing import StandardScaler

fit_data = pa.table({"g": ["a", "a", None], "v": [10.0, 20.0, 7.0]})
requests = pa.table({"g": ["a", "NEW", None], "v": [20.0, 4.0, 14.0]})

projection = SQLProjection("""
    WITH p AS (SELECT g, sc_fit(v) AS iid FROM __FIT__ GROUP BY g)
    SELECT t.g, sc_transform(p.iid, t.v).v AS z
    FROM __THIS__ t LEFT JOIN p ON t.g IS NOT DISTINCT FROM p.g
""", captured={"sc": StandardScaler()})
fitted = projection.fit(fit_data)

expected = [{"g": "a", "z": 1.0}, {"g": "NEW", "z": None}, {"g": None, "z": 7.0}]
assert fitted.transform(requests).to_pylist() == expected
assert fitted.compile().infer_rows(requests.to_pylist()) == expected
```

An unseen group gives NULL. `IS NOT DISTINCT FROM` makes the NULL group
match its fitted instance. Only `SQLProjection` accepts captured estimators.
`SQLTransform` accepts relation callbacks (`Transform`) instead.

## Native catalog

The native catalog (`sql_transform.native`) translates fitted scikit-learn
transformers into transforms that Confit executes natively, without a call to
Python. Continue the scaler example above:

```python
from confit import DuckDBInferFn
from sql_transform.native import to_native

native = DuckDBInferFn(
    fitted.sql,
    row_tables={"__THIS__": fitted.schema},
    static_tables=fitted.params,
    udfs=[to_native(udf, strict=True) for udf in fitted.udfs.values()],
    shape="map",
)
assert native.infer_rows(requests.to_pylist()) == expected
```

With `strict=True`, `to_native` raises `NotNative` if no catalog entry
serves the fitted transformer. Native selection is always explicit. The twin
of a catalog entry is the `PythonTransform` that calls the fitted estimator.
Each catalog entry must stay within its parity bound against its twin. The
[native catalog README](../../loops/native/README.md) lists the entries and
their bounds.

`TreeBasedTransform` packs fitted scikit-learn tree regressors into tables
that Confit scores natively. See
[serving fitted models](../../docs/serving-fitted-models.md).

## Marginalize window SQL

`SQLProjection.marginalize` accepts one SELECT over `__THIS__` with window
aggregates. It replaces each window aggregate with a join to a params table
that holds its value for each partition:

```python
projection = SQLProjection.marginalize(
    "SELECT g, v - avg(v) OVER (PARTITION BY g) AS d FROM __THIS__")
fitted = projection.fit(pa.table({"g": ["a", "a", None], "v": [10.0, 20.0, 7.0]}))

requests = pa.table({"g": ["NEW", None, "a"], "v": [2.0, 14.0, 12.0]})
assert fitted.transform(requests).to_pydict() == {
    "g": ["NEW", None, "a"], "d": [None, 7.0, -3.0]}
```

`projection.source` shows the explicit fit and request SQL that
`marginalize` derived. The ordinary `SQLProjection` constructor does not
marginalize window SQL. The contract lists the
[window forms that `marginalize` accepts](docs/contract.md#7-bounded-window-marginalization).

## Public interface

| Name | What it is |
|---|---|
| `SQLTransform`, `Fitted` | The general transform and its fitted artifact |
| `SQLProjection`, `FittedProjection` | The row-local transform and its fitted artifact |
| `run` | Runs a transform with both arguments bound to the same data, without fit |
| `Transform` | A transform defined by Python fit and transform callbacks over Arrow tables |
| `UDF`, `PythonUDF`, `PythonTransform`, `TreeBasedTransform` | Functions that SQL can call, with declared Arrow types for the arguments and the result |
| `Named`, `OrderSensitive` | Estimator wrappers: declared output names, and a required `ORDER BY` in the fit call |
| `TransformError` and its subclasses | The errors of authoring and fit |

## Unsupported forms

The contract lists the
[unsupported forms and their explicit alternatives](docs/contract.md#8-unsupported-forms-and-explicit-alternatives).
[Unsupported decorrelation](../../docs/decorrelation-unsupported.md) lists the
correlated subqueries that fit refuses, and the reasons.
