"""SQLProjection — projections over ``__THIS__``, fit once, serve row-at-a-time.

``fit(table)`` marginalizes every window aggregate into materialized params
tables plus a rewritten ``serving_sql``; transformers fit per group into
``PythonTransform`` UDFs. One artifact, two bindings: ``transform(table)``
runs it through DuckDB (batch, the oracle), ``infer``/``infer_batch`` run it
through Confit (row-at-a-time, bit-exact with the DuckDB path).
"""

from __future__ import annotations

from sql_transform._marginalize import (
    FitStep,
    Marginalized,
    MarginalizeError,
    ParamsSpec,
    UDFSpec,
    marginalize,
)
from sql_transform._projection import SQLProjection
from sql_transform._udf import UDF, PythonTransform, PythonUDF, UDFError

__all__ = [
    "UDF",
    "FitStep",
    "Marginalized",
    "MarginalizeError",
    "ParamsSpec",
    "PythonTransform",
    "PythonUDF",
    "SQLProjection",
    "UDFError",
    "UDFSpec",
    "marginalize",
]
