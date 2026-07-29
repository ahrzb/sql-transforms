"""SQLProjection — projections over ``__THIS__``, fit once, serve row-at-a-time.

The fit half works today: ``fit(table)`` marginalizes every window aggregate
into materialized params tables plus a rewritten ``serving_sql``. The serving
half (``infer``/``infer_batch`` through Confit) is a later loop and raises
``NotImplementedError``.
"""

from __future__ import annotations

from sql_transform._marginalize import (
    FitStep,
    Marginalized,
    MarginalizeError,
    ParamsSpec,
    TransformSpec,
    marginalize,
)
from sql_transform._projection import SQLProjection

__all__ = [
    "FitStep",
    "Marginalized",
    "MarginalizeError",
    "ParamsSpec",
    "SQLProjection",
    "TransformSpec",
    "marginalize",
]
