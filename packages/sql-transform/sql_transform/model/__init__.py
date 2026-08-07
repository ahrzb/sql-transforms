"""The redesigned data model: a transform is a function ``(F, T) -> R``.

Lands alongside the existing modules; nothing is deleted until the old
implementation and this one agree on the corpus.
"""

from __future__ import annotations

from sql_transform.model._transform import (
    CorrelatedFit,
    Fitted,
    SQLTransform,
    TransformError,
    run,
)

__all__ = [
    "CorrelatedFit",
    "Fitted",
    "SQLTransform",
    "TransformError",
    "run",
]
