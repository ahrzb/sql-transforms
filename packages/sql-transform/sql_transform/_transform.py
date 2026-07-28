"""SQLTransform — the authoring surface, awaiting reimplementation.

Every method below raises `NotImplementedError`. The signatures, defaults and
return types are the contract the rebuild has to satisfy; nothing else survives
of the previous implementation.

The intended shape, for whoever picks this up:

    fit(table)  -- run the SQL over training data, freeze each window aggregate
                   into a static table, rewrite the SQL to reference those
                   tables instead of recomputing, and hand the pair to confit,
                   which partially evaluates them into a native function.
    infer(row)  -- one row through that function; no SQL engine at call time.

Confit (`packages/confit`) is already built and does the serving half. What was
removed was the fit half and everything the deleted DataFusion engine touched.
"""

from __future__ import annotations

from string.templatelib import Template
from typing import Any

import pyarrow as pa
from pydantic import BaseModel

_TODO = "SQLTransform is being rebuilt on confit; {} has no implementation yet"


class SQLTransform:
    """SQL feature transforms: fit once, then serve row-at-a-time.

    Not implemented. See the module docstring for the intended shape.
    """

    def __init__(self, sql: str | Template) -> None:
        raise NotImplementedError(_TODO.format("__init__"))

    @classmethod
    def from_file(cls, path: str) -> SQLTransform:
        """Build a transform from a file containing the SQL."""
        raise NotImplementedError(_TODO.format("from_file"))

    def fit(
        self,
        table: pa.Table,
        /,
        this_model: type[BaseModel] | None = None,
    ) -> SQLTransform:
        """Freeze training-time state and specialize; returns self."""
        raise NotImplementedError(_TODO.format("fit"))

    @property
    def backend(self) -> str:
        """Execution backend: "cranelift", "interpreter", or "constant"."""
        raise NotImplementedError(_TODO.format("backend"))

    @property
    def boundary(self) -> str:
        """Boundary path: "marshaller", "generic", or "constant"."""
        raise NotImplementedError(_TODO.format("boundary"))

    def infer(self, row: dict[str, Any] | BaseModel, /) -> BaseModel:
        """Single-row inference; returns the typed output model instance."""
        raise NotImplementedError(_TODO.format("infer"))

    def infer_batch(self, rows: list[dict[str, Any] | BaseModel], /) -> list[BaseModel]:
        """Many-rows inference; returns typed output model instances."""
        raise NotImplementedError(_TODO.format("infer_batch"))
