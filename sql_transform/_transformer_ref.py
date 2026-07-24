"""Resolve transformer-ref placeholder calls in a desugared SELECT.

A transformer ref desugars to a __COMPOSE_i__(arg...) call. Here we wrap a leaf
call's column args into a single named_struct (the struct arg the engines'
opaque callout expects), and derive the transformer's in/out schema by probing
.transform on the training batch. numpy lives here, mirroring _transformer_udf.
"""

from __future__ import annotations

import copy

import numpy as np
import pyarrow as pa
from sqlglot import exp

from sql_transform._sql import require_in_projection


def is_transformer(obj: object) -> bool:
    """A FITTED sklearn-style transformer.

    Keys off `n_features_in_`, not `feature_names_in_`: sklearn sets
    `n_features_in_` on any successful fit, but `feature_names_in_` only when
    fitted on named data (a DataFrame). Gating on the latter would reject
    perfectly good ndarray-fit transformers. Absence of `n_features_in_` means
    "not fitted", which is what lets us give that its own error.
    """
    return hasattr(obj, "transform") and hasattr(obj, "n_features_in_")


def _find_call(select: exp.Select, name: str) -> exp.Anonymous:
    for n in select.find_all(exp.Anonymous):
        if str(n.this).upper() == name:
            require_in_projection(select, n, f"transformer ref {name}")
            return n
    raise ValueError(
        f"transformer ref {name} must be applied to columns, e.g. {{t}}(a, b)"
    )


def _named_struct(cols: list[exp.Column]) -> exp.Anonymous:
    """named_struct('name', <col>, ...) keyed by the real column name, with each
    value the user's ORIGINAL column ref carried through verbatim. Its quoting is
    preserved (not force-quoted), so an unquoted CamelCase ref folds in DataFusion
    exactly as a hand-written query would -- matching the oracle bug-for-bug
    (TASK-28). Users quote a case-sensitive column themselves: {t}("MSZoning")."""
    args: list[exp.Expression] = []
    for c in cols:
        args.append(exp.Literal.string(c.name))
        args.append(c.copy())
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


def _materialize(
    obj: object, cols: list[str], src: pa.Table, out_schema: pa.Schema
) -> pa.Table:
    """Run obj.transform on src's `cols` and return the result as a pa.Table
    shaped like out_schema, so an outer transformer can probe on real data."""
    x = np.column_stack([src.column(c).to_numpy(zero_copy_only=False) for c in cols])
    y = np.asarray(obj.transform(x))
    arrays = [
        pa.array(y[:, i], type=out_schema.field(i).type) for i in range(len(out_schema))
    ]
    return pa.table(arrays, schema=out_schema)


def resolve_transformer_refs(
    select: exp.Select, tfm_refs: dict[str, object], table: pa.Table
) -> dict[str, tuple[object, pa.Schema, pa.Schema]]:
    """Wrap each leaf transformer call's args into a named_struct and derive its
    schema. A call whose single arg is another transformer-ref call is left
    unwrapped instead -- its outer schema is probed on the inner's materialized
    output, resolving innermost-first. Returns
    {placeholder_name.lower(): (obj, in_schema, out_schema)}."""
    registry: dict[str, tuple[object, pa.Schema, pa.Schema]] = {}
    materialized: dict[
        str, pa.Table
    ] = {}  # name -> this ref's output, for outer probes

    def call_arg_ref(call: exp.Anonymous) -> str | None:
        """If the call's single arg is another transformer-ref call, its name."""
        if len(call.expressions) == 1 and isinstance(
            call.expressions[0], exp.Anonymous
        ):
            inner = str(call.expressions[0].this).upper()
            if inner in tfm_refs:
                return inner
        return None

    def resolve(name: str) -> None:
        if name in materialized:
            return
        call = _find_call(select, name)
        obj = tfm_refs[name]
        inner = call_arg_ref(call)
        if inner is not None:
            resolve(inner)  # innermost first
            in_tbl = materialized[inner]  # inner's output, real data to probe on
            cols = [str(n) for n in obj.feature_names_in_]
            in_schema, out_schema = _derive_schemas(obj, cols, in_tbl)
            # arg is the inner call node; leave it unwrapped.
        else:
            if not all(isinstance(a, exp.Column) for a in call.expressions):
                raise ValueError(
                    f"{name} args must be plain columns or another transformer "
                    f"ref, e.g. {{t}}(a, b) or {{t}}({{u}}(a, b))"
                )
            cols = [a.name for a in call.expressions]
            feat = getattr(obj, "feature_names_in_", None)
            if feat is None:
                # Fitted without names (ndarray). Names are METADATA -- they ride
                # the named_struct as Arrow field names and both engines align on
                # them. sklearn never recorded any, so synthesise them from the
                # call site. Order is the user's contract, exactly as it is when
                # calling sklearn directly; only arity is checkable.
                if len(cols) != obj.n_features_in_:
                    raise ValueError(
                        f"{name} takes {obj.n_features_in_} columns (fitted "
                        f"without names, so arguments bind positionally in "
                        f"call order), got {len(cols)}: {cols}"
                    )
                # copy.copy, never mutate: doc-8's clone contract. Shallow, so
                # the fitted state is shared rather than duplicated.
                obj = copy.copy(obj)
                obj.feature_names_in_ = np.array(cols)
            else:
                feat = [str(n) for n in feat]
                if set(cols) != set(feat):
                    raise ValueError(
                        f"{name} columns {cols} must match feature_names_in_ {feat}"
                    )
            in_schema, out_schema = _derive_schemas(obj, cols, table)
            call.set("expressions", [_named_struct(call.expressions)])
        # Both engines fold an unquoted function-call name to lowercase before
        # resolving it (DataFusion's ANSI identifier folding; Rust's
        # expr_build::convert_function does `.to_lowercase()` explicitly), and
        # sqlglot's generator upper-normalizes the placeholder's text on
        # print regardless of its stored case. Register under the lowercase
        # form so both engines' lookups actually hit it.
        registry[name.lower()] = (obj, in_schema, out_schema)
        materialized[name] = _materialize(
            obj, cols, table if inner is None else materialized[inner], out_schema
        )

    for name in tfm_refs:
        resolve(name)
    return registry
