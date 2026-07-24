"""Wrap a fitted sklearn transformer as a vectorized DataFusion UDF.

The oracle side of the engine transformer-callout capability: `transform`
registers this UDF so a query can call a fitted transformer by name, struct in
/ struct out. The row engine (`InferFn`) performs the identical alignment and
marshalling in Rust; the differential harness proves the two agree.

Input is aligned to the object's `feature_names_in_` order; output is built
from the caller-declared `out_schema` (no introspection). numpy lives here and
here only -- the Rust engine imports no numpy.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from datafusion import ScalarUDF, udf


def check_out_schema_natural(obj: object, y: np.ndarray, out_fields: list) -> None:
    """Enforce out_schema == the transform's NATURAL output dtype (TASK-2 AC#3).

    The two engines reach the declared type by different coercion rules --
    this one via a pyarrow cast (`pa.array(..., type=)`), the native `infer`
    engine via pydantic coercion at model-validate. Those rules only agree
    when no coercion happens, i.e. when the declared type already IS what
    `.transform` produces. A mismatch is therefore a silent cross-engine
    divergence (declaring int64 over a float64 transform truncates on one
    side only), so refuse it rather than let the engines drift apart.
    """
    natural = pa.from_numpy_dtype(y.dtype)
    bad = [f.name for f in out_fields if f.type != natural]
    if bad:
        declared = [f.type for f in out_fields if f.name in bad]
        raise ValueError(
            f"{type(obj).__name__} out_schema declares {declared} for {bad}, but "
            f"the natural .transform dtype is {natural}. The engines coerce to a "
            f"declared type differently (pyarrow cast vs pydantic), so they only "
            f"agree when the declared type IS the natural one."
        )


def _transformer_udf(
    obj: object,
    in_schema: pa.Schema,
    out_schema: pa.Schema,
    name: str,
) -> ScalarUDF:
    """Build a vectorized struct-in/struct-out DataFusion scalar UDF.

    obj: a fitted sklearn transformer/Pipeline exposing `.transform` and
        `feature_names_in_`.
    in_schema: names+types of the struct the SQL feeds in (the authored
        `named_struct`). Its field-name set must equal `feature_names_in_`.
    out_schema: names+types of the returned struct (declared, not introspected).
    name: the reserved SQL identifier the UDF is registered and called under.

    Invariant: `out_schema` must equal the transformer's NATURAL output dtype.
    This engine coerces to it via a pyarrow cast (`pa.array(..., type=...)`);
    the Rust `infer` engine reaches the same declared type by pydantic
    coercion at model-validate time. Those two coercion rules only agree when
    no real coercion happens -- i.e. when the declared type already matches
    what `.transform` produces. Declaring a mismatched dtype would diverge.
    """
    feature_names = [str(n) for n in obj.feature_names_in_]
    in_type = pa.struct(list(in_schema))
    out_type = pa.struct(list(out_schema))
    out_fields = list(out_schema)

    def _apply(struct_array: pa.Array) -> pa.Array:
        # DataFusion hands the whole batch's StructArray in one call. Pull fields
        # BY NAME into feature_names_in_ order. DO NOT DELETE this reorder: the
        # {t}(...) authoring path now emits the struct already in fitted order
        # (TASK-35), so for it this is a no-op -- but hand-authored SQL can call
        # this UDF with a named_struct in any field order, and this is the only
        # thing that keeps such a call correct. src/expr.rs carries the same
        # reorder for the native engine, for the same reason.
        cols = [
            struct_array.field(fname).to_numpy(zero_copy_only=False)
            for fname in feature_names
        ]
        x = np.column_stack(cols)
        y = np.asarray(obj.transform(x))
        check_out_schema_natural(obj, y, out_fields)
        out_cols = [
            pa.array(y[:, i], type=out_fields[i].type) for i in range(len(out_fields))
        ]
        return pa.StructArray.from_arrays(out_cols, fields=out_fields)

    return udf(_apply, [in_type], out_type, "immutable", name=name)
