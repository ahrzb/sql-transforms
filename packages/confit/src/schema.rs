use pyo3::prelude::*;

use crate::error::InterpError;
use crate::specializer::ir::Ty;

/// One row-table column as its `pa.Schema` field declares it: a scalar the
/// engine serves at that exact width, a struct of these (flattened to leaf
/// lanes downstream, TASK-56), or a type outside the vocabulary — kept
/// opaque so an unreferenced foreign column never blocks a build, while any
/// reference refuses by name.
pub enum RowField {
    Scalar {
        ty: Ty,
        nullable: bool,
    },
    Struct {
        nullable: bool,
        fields: Vec<(String, RowField)>,
    },
    Opaque,
}

/// Which acceptance policy a schema is read under.
///
/// One walker, two policies — and they are DIFFERENT today, which is worth
/// stating plainly because the difference used to be accidental (two
/// parsers in two files) and is now deliberate.
///
/// `Row` is exact: a type is served at its declared width or it is opaque.
/// `Static` additionally takes types the catalogue has always taken by
/// widening them — unsigned ints into the i64 lane, float32 and
/// `decimal128(p,s)` into the f64 lane. That widening is lossy (an exact
/// decimal beyond 2^53 is TASK-91's whole subject) and narrowing it to the
/// row policy would break builds that work today, so it stays until that
/// is decided on purpose rather than as a side effect of this refactor.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Row,
    Static,
}

/// Parse a row table's `pa.Schema` into its declaration-ordered columns.
/// This is the boundary where arrow widths become engine widths: int32 IS
/// `Ty::I32`, never a collapse to i64 — the exact information the pydantic
/// surface (width-less `int`) could not carry.
pub fn arrow_row_schema(
    py: Python<'_>,
    table: &str,
    schema_obj: &Py<PyAny>,
) -> Result<Vec<(String, RowField)>, InterpError> {
    arrow_schema_fields(
        py,
        &format!("row_tables['{table}']"),
        schema_obj,
        Policy::Row,
    )
}

/// The same walk for a static table's `pa.Table.schema`.
///
/// TASK-96: the catalogue used to run its own parser, which collapsed every
/// integer width to i64 — so an int32 static emitted int64 where DuckDB
/// emits int32, while an int32 ROW column bound correctly. Same physical
/// vocabulary, two readers, one of them wrong.
pub fn arrow_static_schema(
    py: Python<'_>,
    table: &str,
    schema_obj: &Py<PyAny>,
) -> Result<Vec<(String, RowField)>, InterpError> {
    arrow_schema_fields(
        py,
        &format!("static_tables['{table}']"),
        schema_obj,
        Policy::Static,
    )
}

fn arrow_schema_fields(
    py: Python<'_>,
    ctx: &str,
    schema_obj: &Py<PyAny>,
    policy: Policy,
) -> Result<Vec<(String, RowField)>, InterpError> {
    let bound = schema_obj.bind(py);
    let names: Vec<String> = bound
        .getattr("names")
        .and_then(|n| n.extract())
        .map_err(|e| InterpError::Build(format!("{ctx} is not a pyarrow.Schema: {e}")))?;
    let pa_types = PyModule::import(py, "pyarrow.types")
        .map_err(|e| InterpError::Build(format!("Failed to import pyarrow.types: {e}")))?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let field = bound
            .call_method1("field", (name.as_str(),))
            .map_err(|e| InterpError::Build(format!("Failed to read field '{name}': {e}")))?;
        let rf = arrow_field_to_row_field(&pa_types, &field, policy)
            .map_err(|e| InterpError::Build(format!("Failed to read type of '{name}': {e}")))?;
        out.push((name, rf));
    }
    Ok(out)
}

fn arrow_field_to_row_field(
    pa_types: &Bound<'_, PyModule>,
    field: &Bound<'_, PyAny>,
    policy: Policy,
) -> PyResult<RowField> {
    let nullable: bool = field.getattr("nullable")?.extract()?;
    let ty = field.getattr("type")?;
    if pa_types
        .call_method1("is_struct", (&ty,))?
        .extract::<bool>()?
    {
        let num_fields: usize = ty.getattr("num_fields")?.extract()?;
        let mut fields = Vec::with_capacity(num_fields);
        for i in 0..num_fields {
            let f = ty.call_method1("field", (i,))?;
            let name: String = f.getattr("name")?.extract()?;
            fields.push((name, arrow_field_to_row_field(pa_types, &f, policy)?));
        }
        return Ok(RowField::Struct { nullable, fields });
    }
    let name = ty.str()?.extract::<String>()?;
    // Exact names, not prefixes: "float" (float32) must NOT catch a ride on
    // "float64" the way the old prefix match silently widened it.
    let t = match name.as_str() {
        "bool" => Ty::I1,
        "int8" => Ty::I8,
        "int16" => Ty::I16,
        "int32" => Ty::I32,
        "int64" => Ty::I64,
        "double" => Ty::F64,
        "string" => Ty::Str,
        // The catalogue's historical extras, preserved exactly. Widening,
        // and lossy where it widens — an unsigned payload past i64 refuses
        // downstream by range, and a decimal past 2^53 is TASK-91.
        _ if policy == Policy::Static => match name.as_str() {
            "uint8" | "uint16" | "uint32" | "uint64" => Ty::I64,
            "float" => Ty::F64,
            "large_string" | "utf8" | "large_utf8" => Ty::Str,
            n if n.starts_with("decimal") => Ty::F64,
            _ => return Ok(RowField::Opaque),
        },
        _ => return Ok(RowField::Opaque),
    };
    Ok(RowField::Scalar { ty: t, nullable })
}

