# sql-transform

> **Native transforms.** `sql_transform.native` turns fitted sklearn
> transformers into confit functions (`to_native(step)`), held to their
> Python twins by swap-the-entry parity. It grows by its own loop:
> [loops/native/README.md](../../loops/native/README.md).

`SQLTransform` authors compositional SQL over fit data (`__FIT__`) and request
data (`__THIS__`). Fit binds the fit data and writes params for later execution.
`SQLProjection` adds the row-local rule and offers compilation through
[Confit](../confit).

## Usage

Write the fit relationship explicitly:

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
fn = fitted.compile()
assert fn.infer_rows(requests.to_pylist()) == [{"d": -3.0}, {"d": 7.0}]
assert fn.infer_arrow(requests).to_pydict() == {"d": [-3.0, 7.0]}
```

For the bounded one-level window convenience, call
`SQLProjection.marginalize("SELECT v - avg(v) OVER () AS d FROM __THIS__")`.
The ordinary constructor does not marginalize automatically.

## Current contract

[The authoring contract](docs/contract.md) defines the current API, executable
examples, supported decorrelation, composition, Python callbacks, estimator
IDs, window admission, and Confit refusals.
For admitted projections, `fitted.sql`, `fitted.schema`, `fitted.params`, and
`fitted.udfs` form the complete public serving artifact.
`compile()` returns a fresh Confit function or an explicit refusal.

Window projection chains, declared-schema expansion, hidden aliases, and old
inference forwarding interfaces are not admitted.
Use the contract's [explicit alternatives](docs/contract.md#8-unsupported-forms-and-explicit-alternatives)
for these computations.
The [system specification](../../docs/specs/README.md) links all current topics.
