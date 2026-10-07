"""Fitted sklearn transformers as confit functions: the native catalog.

    from sql_transform.native import to_native, explain_native, check

    fn = to_native(step)       # a confit SqlFunction, or `step` itself
    explain_native(step)       # what it becomes, or why not
    check(step, fn, rows)      # swap-the-entry parity, raises on a breach

`step` is a `PythonTransform`: fitted instances keyed by id, called as
`name(id, features...)`. A translation is a drop-in replacement: same name,
same call, same result (packages/sql-transform/spec/native/catalog-contract.md
says what "same" means). Where no translation exists, `to_native` hands back
the step unchanged, which is always correct to serve, only slower. By default
only a bit-exact translation serves; `to_native(step, allow_bound=True)` also
serves one within a parity bound above 0 of its twin, which can change a
prediction (`_registry.py` says why).

The catalog is one module per family (`scalers.py`, `impute.py`,
`select.py`, `function.py`, `polynomial.py`, `spline.py`, `encode.py`,
`quantile.py`, `isotonic.py`, ...), each entry registered with
`@translates(EstimatorClass)`; `compose.py` composes them (a `Pipeline`,
a `ColumnTransformer`, a `FeatureUnion`).
This package imports confit and sklearn and nothing else of sql_transform
but the step class and `_trees._f32_grid_threshold` (the float32 cutpoint
`discretize.py` and `trees.py` share with the tree predictors), so the
rest of the package can change around it.
"""

from __future__ import annotations

from sql_transform.native import agglomerate as _agglomerate  # noqa: F401  (registers)
from sql_transform.native import compose as _compose  # noqa: F401  (registers)
from sql_transform.native import discretize as _discretize  # noqa: F401  (registers)
from sql_transform.native import encode as _encode  # noqa: F401  (registers)
from sql_transform.native import function as _function  # noqa: F401  (registers)
from sql_transform.native import impute as _impute  # noqa: F401  (registers)
from sql_transform.native import isotonic as _isotonic  # noqa: F401  (registers)
from sql_transform.native import polynomial as _polynomial  # noqa: F401  (registers)
from sql_transform.native import power as _power  # noqa: F401  (registers)
from sql_transform.native import quantile as _quantile  # noqa: F401  (registers)
from sql_transform.native import scalers as _scalers  # noqa: F401  (registers)
from sql_transform.native import select as _select  # noqa: F401  (registers)
from sql_transform.native import spline as _spline  # noqa: F401  (registers)
from sql_transform.native import trees as _trees  # noqa: F401  (registers)
from sql_transform.native._check import ParityError, check
from sql_transform.native._registry import (
    Entry,
    ErrorScale,
    NotNative,
    bound,
    bound_of,
    catalog,
    explain_native,
    to_native,
    translates,
)

__all__ = [
    "Entry",
    "ErrorScale",
    "NotNative",
    "bound",
    "bound_of",
    "ParityError",
    "catalog",
    "check",
    "explain_native",
    "to_native",
    "translates",
]
