use pyo3::prelude::*;

use crate::error::InterpError;
use crate::specializer::ir::Ty;

/// One row-table column as its `pa.Schema` field declares it: a scalar the
/// engine serves at that exact width, a struct of these (flattened to leaf
/// lanes downstream), or a type outside the vocabulary — kept opaque so an
/// unreferenced foreign column never blocks a build, while any reference
/// refuses by name.
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
/// One walker, two policies — and they are DIFFERENT, which is worth
/// stating plainly, because the difference is a decision rather than an
/// artifact of who wrote which reader.
///
/// `Row` is exact: a type is served at its declared width or it is opaque.
/// `Static` additionally takes `large_string`/`utf8` and the decimal tiers
/// `decimal32`/`decimal64`/`decimal128` as `Ty::Dec(p,s)`, and ONLY those —
/// each measured against DuckDB rather than assumed. `decimal256` stays
/// opaque because DuckDB refuses it outright at arrow register ("Unsupported
/// Internal Arrow Type for Decimal"), at any precision.
///
/// The two remaining differences are `large_string`/`utf8` and the decimal
/// acceptance itself. Decimal ROW columns stay opaque, which is also why the
/// arrow ROW-ingest path needs no decimal arm at all.
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
/// One reader for both, deliberately: row and static columns are the same
/// physical vocabulary, so a second reader of it drifts. A catalogue running
/// a parser of its own collapsed every integer width to i64 — an int32
/// static then emitted int64 where DuckDB emits int32, while an int32 ROW
/// column bound correctly.
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
    // exist". Whether a repeat is legal is the caller's rule to state, in
    // its own words.
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
        //   decimal32/64/128   an ordinary fit-path output (sum(BIGINT) is
        //                      decimal128(38,0)). The payload is the SCALED
        //                      i128, exact from ingest to emit. Every tier
        //                      leaves DuckDB as decimal128(p,s)
        //                      (SetArrowFormat, arrow_converter.cpp), so
        //                      all three normalise to one Ty::Dec(p,s).
        //
        // decimal256 does NOT ride here: DuckDB refuses it at arrow
        // register at any precision, so serving it would be
        // serve-where-DuckDB-refuses. It stays opaque, which refuses by
        // name on reference and costs nothing unreferenced.
        //
        // float32 and the unsigned widths do NOT ride here either: both
        // DIVERGE (measured 2026-08-15). float32 in value AND type
        // (s.v * 3.0 is 0.30000001192092896/FLOAT on DuckDB, f64 arithmetic
        // here), unsigned in type (uint64 stays UINT64 there, int64 here).
        // The row path refuses them; a catalogue that widened them silently
        // instead would be the third mode the contract forbids.
        _ if policy == Policy::Static => match name.as_str() {
            "large_string" | "utf8" | "large_utf8" => Ty::Str,
            n if n.starts_with("decimal") => match decimal_ps(n) {
                Some((p, s)) => Ty::Dec(p, s),
                None => return Ok(RowField::Opaque(name)),
            },
            _ => return Ok(RowField::Opaque(name)),
        },
        _ => return Ok(RowField::Opaque(name)),
    };
    Ok(RowField::Scalar { ty: t, nullable })
}

/// `(p, s)` out of pyarrow's own spelling — `decimal128(38, 0)`,
/// `decimal32(6, 2)`. `None` for `decimal256` (DuckDB refuses it) and for
/// anything outside DECIMAL's 1..=38 / s <= p range.
fn decimal_ps(name: &str) -> Option<(u8, u8)> {
    let (head, args) = name.split_once('(')?;
    if !matches!(head, "decimal32" | "decimal64" | "decimal128" | "decimal") {
        return None;
    }
    let (p, s) = args.strip_suffix(')')?.split_once(',')?;
    let p: u8 = p.trim().parse().ok()?;
    let s: u8 = s.trim().parse().ok()?;
    ((1..=38).contains(&p) && s <= p).then_some((p, s))
}

