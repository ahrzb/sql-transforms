"""The redesigned data model: a transform is a function ``(F, T) -> R``.

Lands alongside the existing modules; nothing is deleted until the old
implementation and this one agree on the corpus.
"""

from __future__ import annotations

from sql_transform.model._transform import (
    MAX_DEPTH,
    CorrelatedFit,
    Fitted,
    NestingTooDeep,
    SQLTransform,
    Transform,
    TransformError,
    UnknownName,
    normalize,
    run,
)

__all__ = [
    "MAX_DEPTH",
    "CorrelatedFit",
    "Fitted",
    "NestingTooDeep",
    "SQLTransform",
    "Transform",
    "TransformError",
    "UnknownName",
    "normalize",
    "run",
]
