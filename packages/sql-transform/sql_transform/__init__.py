"""SQLProjection — projections over ``__THIS__``, fit once, serve row-at-a-time.

The fit half works today: ``fit(table)`` marginalizes every window aggregate
into materialized params tables plus a rewritten ``serving_sql``; transformers
and author UDFs become scalar calls that ``transform(table)`` serves by
registering them on the connection. The serving half (``infer``/``infer_batch``
through Confit) is a later loop and raises ``NotImplementedError``.
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
