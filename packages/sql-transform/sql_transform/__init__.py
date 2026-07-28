"""SQLTransform — the authoring surface, awaiting reimplementation.

Every method raises `NotImplementedError`. The signatures are the contract a
rebuild has to satisfy; the previous implementation (a DataFusion batch engine,
a codegen backend, and transform composition) was deleted deliberately.

The serving half already exists and is unaffected: confit takes SQL plus static
tables frozen at fit time and partially evaluates them into a native function.
"""

from __future__ import annotations

from sql_transform._transform import SQLTransform

__all__ = ["SQLTransform"]
