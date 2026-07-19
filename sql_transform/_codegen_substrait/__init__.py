"""Substrait front-end for the codegen serving engine.

Same idea as `sql_transform._codegen`, but the plan comes from a **Substrait
plan** instead of SQL parsed by sqlglot. The Substrait plan is translated into
the existing `_codegen` plan IR and handed to the same backend-agnostic pipeline
(`CodegenFn._finalize`): optimize, validate, type-infer, compile, run. The
`rt.*` runtime and the emitter are reused untouched.

References the Substrait plan against a `Context` (registered row-table schemas
and static tables) — the plan carries only names/anchors/indices; the context
resolves them.
"""

from __future__ import annotations

from sql_transform._codegen.plan import UnsupportedInCodegen
from sql_transform._codegen_substrait.context import Context
from sql_transform._codegen_substrait.engine import CodegenSubstraitFn

__all__ = ["CodegenSubstraitFn", "Context", "UnsupportedInCodegen"]
