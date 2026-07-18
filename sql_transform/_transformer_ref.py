"""Resolve transformer-ref placeholder calls in a desugared SELECT.

A transformer ref desugars to a __COMPOSE_i__(arg...) call. Here we wrap a leaf
call's column args into a single named_struct (the struct arg the engines'
opaque callout expects), and derive the transformer's in/out schema by probing
.transform on the training batch. numpy lives here, mirroring _transformer_udf.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from sqlglot import exp


def is_transformer(obj: object) -> bool:
    return hasattr(obj, "feature_names_in_") and hasattr(obj, "transform")


def _find_call(select: exp.Select, name: str) -> exp.Anonymous:
    for n in select.find_all(exp.Anonymous):
        if str(n.this).upper() == name:
            return n
    raise ValueError(f"transformer ref {name} must be applied to columns, e.g. {{t}}(a, b)")


def _named_struct(cols: list[str]) -> exp.Anonymous:
    """named_struct('c0', c0, 'c1', c1, ...) keyed by column name."""
    args: list[exp.Expression] = []
    for c in cols:
        args.append(exp.Literal.string(c))
        args.append(exp.column(c))
    return exp.Anonymous(this="named_struct", expressions=args)


def _derive_schemas(
    obj: object, cols: list[str], table: pa.Table
) -> tuple[pa.Schema, pa.Schema]:
    """in_schema from the training columns; out_schema by probing .transform."""
    in_schema = pa.schema([(c, table.schema.field(c).type) for c in cols])
    x = np.column_stack([table.column(c).to_numpy(zero_copy_only=False) for c in cols])
    y = np.asarray(obj.transform(x))
    names = [str(n) for n in obj.get_feature_names_out()]
    if y.ndim != 2 or y.shape[1] != len(names):
        raise ValueError(
            f"cannot derive out_schema for {type(obj).__name__}: expected 2-D width "
            f"{len(names)}, got shape {y.shape}"
        )
    out_schema = pa.schema([(n, pa.from_numpy_dtype(y.dtype)) for n in names])
    return in_schema, out_schema


def resolve_transformer_refs(
    select: exp.Select, tfm_refs: dict[str, object], table: pa.Table
) -> dict[str, tuple[object, pa.Schema, pa.Schema]]:
    """Wrap each leaf transformer call's args into a named_struct and derive its
    schema. Returns {placeholder_name: (obj, in_schema, out_schema)}.
    (Nested calls: Task 2.)"""
    registry: dict[str, tuple[object, pa.Schema, pa.Schema]] = {}
    for name, obj in tfm_refs.items():
        call = _find_call(select, name)
        cols = [a.name for a in call.expressions]
        if len(cols) != len(call.expressions):
            raise ValueError(f"{name} args must be plain columns (nested calls: Task 2)")
        feat = [str(n) for n in obj.feature_names_in_]
        if set(cols) != set(feat):
            raise ValueError(
                f"{name} columns {cols} must match feature_names_in_ {feat}"
            )
        in_schema, out_schema = _derive_schemas(obj, cols, table)
        call.set("expressions", [_named_struct(cols)])
        # Both engines fold an unquoted function-call name to lowercase before
        # resolving it (DataFusion's ANSI identifier folding; Rust's
        # expr_build::convert_function does `.to_lowercase()` explicitly), and
        # sqlglot's generator upper-normalizes the placeholder's text on
        # print regardless of its stored case. Register under the lowercase
        # form so both engines' lookups actually hit it.
        registry[name.lower()] = (obj, in_schema, out_schema)
    return registry
