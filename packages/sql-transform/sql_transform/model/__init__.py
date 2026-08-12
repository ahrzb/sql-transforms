"""The redesigned data model: a transform is a function ``(F, T) -> R``.

Lands alongside the existing modules; nothing is deleted until the old
implementation and this one agree on the corpus.

The package is split by what each part stands on, so the arrows only ever
point one way:

    _errors     every refusal, and nothing else
    _ast        DuckDB as parser and printer, and the walk over its nodes
    _analysis   what a subtree reads, and what it correlates to
    _foreign    the supplied (fit, transform) pair, and θ
    _plan       freezing: params + residual
    _program    the compiled text — resolution, binding, Program, Fitted
    _transform  the estimator surface — SQLTransform, output currencies, run
"""

from sql_transform.model._ast import normalize
from sql_transform.model._errors import (
    CorrelatedFit,
    NestingTooDeep,
    NotFitted,
    NotRowWise,
    TransformError,
    UnknownName,
    WholeTrainingSet,
)
from sql_transform.model._foreign import Transform
from sql_transform.model._program import MAX_DEPTH, Fitted
from sql_transform.model._projection import FittedProjection, SQLProjection
from sql_transform.model._transform import SQLTransform, run

__all__ = [
    "MAX_DEPTH",
    "CorrelatedFit",
    "Fitted",
    "FittedProjection",
    "NestingTooDeep",
    "NotFitted",
    "NotRowWise",
    "SQLProjection",
    "SQLTransform",
    "Transform",
    "TransformError",
    "UnknownName",
    "WholeTrainingSet",
    "normalize",
    "run",
]
