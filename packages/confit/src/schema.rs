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
    /// A type this engine does not serve, carrying its ARROW spelling so a
    /// refusal can name it. Unreferenced it costs nothing; referenced it
    /// refuses.
    Opaque(String),
}

/// Which acceptance policy a schema is read under.
///
/// One walker, two policies — and they are DIFFERENT today, which is worth
/// stating plainly because the difference used to be accidental (two
/// parsers in two files) and is now deliberate.
///
/// `Row` is exact: a type is served at its declared width or it is opaque.
/// `Static` additionally takes `large_string`/`utf8` and `decimal128(p,s)`,
/// and ONLY those — both measured against DuckDB rather than assumed. The
/// difference is the remaining gap between the two, not a design; closing
/// it needs TASK-91 (exact decimal serving) first.
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
    // By POSITION, not by name: a schema may carry the same name twice, and
    // pyarrow's by-name lookup answers an ambiguous name with "does not
    // exist" (TASK-127). Whether a repeat is legal is the caller's rule to
    // state, in its own words.
    for (i, name) in names.into_iter().enumerate() {
        let field = bound
            .call_method1("field", (i,))
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
        // Two catalogue extras survive, because both are MEASURED
        // equivalent rather than merely convenient:
        //
        //   large_string/utf8  DuckDB normalises these to VARCHAR, so the
        //                      value and the output type both match.
        //   decimal128(p,s)    an ordinary fit-path output (sum(BIGINT) is
        //                      decimal128(38,0)). Rides the f64 lane behind
        //                      an exactness guard that refuses any payload
        //                      f64 cannot hold; TASK-91 lands exact serving.
        //
        // float32 and the unsigned widths used to ride here too and were
        // removed 2026-08-15: both DIVERGE. float32 in value AND type
        // (s.v * 3.0 is 0.30000001192092896/FLOAT on DuckDB, f64 arithmetic
        // here), unsigned in type (uint64 stays UINT64 there, int64 here).
        // The row path always refused them; the catalogue widened them
        // silently, which is the third mode the contract forbids.
        _ if policy == Policy::Static => match name.as_str() {
            "large_string" | "utf8" | "large_utf8" => Ty::Str,
            n if n.starts_with("decimal") => Ty::F64,
            _ => return Ok(RowField::Opaque(name)),
        },
        _ => return Ok(RowField::Opaque(name)),
    };
    Ok(RowField::Scalar { ty: t, nullable })
}

