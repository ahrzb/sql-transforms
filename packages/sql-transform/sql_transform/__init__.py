"""Compositional SQL transforms: bind fit data, then execute requests.

SQLTransform supports general relation computations. SQLProjection adds the
row-local rule and exposes a fitted artifact for batch execution and Confit.
Window marginalization is an explicit, bounded authoring convenience.
"""

from sql_transform._ast import normalize
from sql_transform._errors import (
    CorrelatedFit,
    KeyNotUnique,
    NestingTooDeep,
    NotFitted,
    NotRowWise,
    TransformError,
    UnknownName,
    WholeTrainingSet,
)
from sql_transform._foreign import Transform
from sql_transform._program import MAX_DEPTH, Fitted
from sql_transform._projection import FittedProjection, SQLProjection
from sql_transform._transform import SQLTransform, run
from sql_transform._trees import TreeBasedTransform, TreePackError
from sql_transform._udf import (
    UDF,
    Named,
    OrderSensitive,
    PythonTransform,
    PythonUDF,
    UDFError,
)

__all__ = [
    "MAX_DEPTH",
    "UDF",
    "CorrelatedFit",
    "Fitted",
    "FittedProjection",
    "KeyNotUnique",
    "Named",
    "NestingTooDeep",
    "NotFitted",
    "NotRowWise",
    "OrderSensitive",
    "PythonTransform",
    "PythonUDF",
    "SQLProjection",
    "SQLTransform",
    "Transform",
    "TransformError",
    "TreeBasedTransform",
    "TreePackError",
    "UDFError",
    "UnknownName",
    "WholeTrainingSet",
    "normalize",
    "run",
]
