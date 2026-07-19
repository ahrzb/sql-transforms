"""The resolution environment for a Substrait plan.

A Substrait plan is portable: it names tables, references functions by anchor,
and columns by position. It carries no schemas or table data. The `Context`
supplies what the plan omits — the row-table schemas (as Pydantic models, the
same shape `CodegenFn`/`InferFn` take) and the static tables (Arrow tables the
lookup joins index). Registered in the DataFusion session that *produces* the
plan and consulted again when we *consume* it.

Transformers are intentionally out of scope for now (see package docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Context:
    row_tables: dict[str, Any] = field(default_factory=dict)
    """name -> Pydantic model class (the row schema)."""
    static_tables: dict[str, Any] = field(default_factory=dict)
    """name -> pyarrow.Table (frozen fit-state / lookup tables)."""
