use pyo3::prelude::*;

use crate::error::InterpError;
use crate::specializer::ir::Ty;
use crate::types::{Base, FieldType, Schema};

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

/// Parse a row table's `pa.Schema` into its declaration-ordered columns.
/// This is the boundary where arrow widths become engine widths: int32 IS
/// `Ty::I32`, never a collapse to i64 — the exact information the pydantic
/// surface (width-less `int`) could not carry.
pub fn arrow_row_schema(
    py: Python<'_>,
    table: &str,
    schema_obj: &Py<PyAny>,
) -> Result<Vec<(String, RowField)>, InterpError> {
    let bound = schema_obj.bind(py);
    let names: Vec<String> = bound
        .getattr("names")
        .and_then(|n| n.extract())
        .map_err(|e| {
            InterpError::Build(format!(
                "row_tables['{table}'] is not a pyarrow.Schema: {e}"
            ))
        })?;
    let pa_types = PyModule::import(py, "pyarrow.types")
        .map_err(|e| InterpError::Build(format!("Failed to import pyarrow.types: {e}")))?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let field = bound
            .call_method1("field", (name.as_str(),))
            .map_err(|e| InterpError::Build(format!("Failed to read field '{name}': {e}")))?;
        let rf = arrow_field_to_row_field(&pa_types, &field)
            .map_err(|e| InterpError::Build(format!("Failed to read type of '{name}': {e}")))?;
        out.push((name, rf));
    }
    Ok(out)
}

fn arrow_field_to_row_field(
    pa_types: &Bound<'_, PyModule>,
    field: &Bound<'_, PyAny>,
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
            fields.push((name, arrow_field_to_row_field(pa_types, &f)?));
        }
        return Ok(RowField::Struct { nullable, fields });
    }
    // Exact names, not prefixes: "float" (float32) must NOT catch a ride on
    // "float64" the way the old prefix match silently widened it.
    let t = match ty.str()?.extract::<String>()?.as_str() {
        "bool" => Ty::I1,
        "int8" => Ty::I8,
        "int16" => Ty::I16,
        "int32" => Ty::I32,
        "int64" => Ty::I64,
        "double" => Ty::F64,
        "string" => Ty::Str,
        _ => return Ok(RowField::Opaque),
    };
    Ok(RowField::Scalar { ty: t, nullable })
}

/// Extract a Schema from a `pyarrow.Table`'s `.schema`.
pub fn from_arrow_table(py: Python<'_>, table: &Py<PyAny>) -> Result<Schema, InterpError> {
    let bound = table.bind(py);
    let arrow_schema = bound
        .getattr("schema")
        .map_err(|e| InterpError::Build(format!("Not a pyarrow.Table: {e}")))?;
    let names: Vec<String> = arrow_schema
        .getattr("names")
        .and_then(|n| n.extract())
        .map_err(|e| InterpError::Build(format!("Failed to read static table schema: {e}")))?;
    let pa_types = PyModule::import(py, "pyarrow.types")
        .map_err(|e| InterpError::Build(format!("Failed to import pyarrow.types: {e}")))?;

    let mut schema = Schema::new();
    for name in names {
        let field = arrow_schema
            .call_method1("field", (name.as_str(),))
            .map_err(|e| InterpError::Build(format!("Failed to read field '{name}': {e}")))?;
        let field_type = arrow_field_to_field_type(&pa_types, &field)
            .map_err(|e| InterpError::Build(format!("Failed to read type of '{name}': {e}")))?;
        schema.insert(name, field_type);
    }
    Ok(schema)
}

/// Parse a `pyarrow.Schema` object into an order-preserving `Vec` of
/// `(name, FieldType)`. Unlike `from_arrow_table` (which reads a `pa.Table`
/// and returns an unordered `Schema` HashMap), this takes a bare `pa.Schema`
/// and preserves field order -- required because a transformer's declared
/// output becomes a `Base::Struct`/`Value::Struct` whose field order is
/// semantically significant.
pub fn arrow_schema_to_ordered_fields(
    py: Python<'_>,
    schema_obj: &Py<PyAny>,
) -> Result<Vec<(String, FieldType)>, InterpError> {
    let bound = schema_obj.bind(py);
    let names: Vec<String> = bound
        .getattr("names")
        .and_then(|n| n.extract())
        .map_err(|e| {
            InterpError::Build(format!(
                "transformer output schema is not a pyarrow.Schema: {e}"
            ))
        })?;
    let pa_types = PyModule::import(py, "pyarrow.types")
        .map_err(|e| InterpError::Build(format!("Failed to import pyarrow.types: {e}")))?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let field = bound.call_method1("field", (name.as_str(),)).map_err(|e| {
            InterpError::Build(format!("Failed to read output field '{name}': {e}"))
        })?;
        let ft = arrow_field_to_field_type(&pa_types, &field).map_err(|e| {
            InterpError::Build(format!("Failed to read type of output field '{name}': {e}"))
        })?;
        out.push((name, ft));
    }
    Ok(out)
}

/// Recursively resolves a `pyarrow.Field` (name/nullable/type) into a
/// `FieldType`, walking `pa.StructType`/`pa.ListType` children rather than
/// prefix-matching the type's string form — needed since a struct/list type's
/// `str()` doesn't expose its nested field types.
fn arrow_field_to_field_type(
    pa_types: &Bound<'_, PyModule>,
    field: &Bound<'_, PyAny>,
) -> PyResult<FieldType> {
    let nullable: bool = field.getattr("nullable")?.extract()?;
    let ty = field.getattr("type")?;
    let base = arrow_pytype_to_base(pa_types, &ty)?;
    Ok(FieldType { base, nullable })
}

fn arrow_pytype_to_base(pa_types: &Bound<'_, PyModule>, ty: &Bound<'_, PyAny>) -> PyResult<Base> {
    if pa_types
        .call_method1("is_struct", (ty,))?
        .extract::<bool>()?
    {
        let num_fields: usize = ty.getattr("num_fields")?.extract()?;
        let mut fields = Vec::with_capacity(num_fields);
        for i in 0..num_fields {
            let f = ty.call_method1("field", (i,))?;
            let name: String = f.getattr("name")?.extract()?;
            fields.push((name, arrow_field_to_field_type(pa_types, &f)?));
        }
        return Ok(Base::Struct(fields));
    }
    let is_list = pa_types.call_method1("is_list", (ty,))?.extract::<bool>()?
        || pa_types
            .call_method1("is_large_list", (ty,))?
            .extract::<bool>()?;
    if is_list {
        let value_field = ty.getattr("value_field")?;
        let inner = arrow_field_to_field_type(pa_types, &value_field)?;
        return Ok(Base::List(Box::new(inner)));
    }
    let type_str: String = ty.str()?.extract()?;
    Ok(arrow_type_to_base(&type_str))
}

fn arrow_type_to_base(type_str: &str) -> Base {
    if type_str.starts_with("int") || type_str.starts_with("uint") {
        Base::Int
    } else if type_str.starts_with("float")
        || type_str.starts_with("double")
        || type_str.starts_with("decimal")
    {
        Base::Float
    } else if type_str.starts_with("string")
        || type_str.starts_with("utf8")
        || type_str.starts_with("large_string")
    {
        Base::Str
    } else if type_str == "bool" {
        Base::Bool
    } else {
        Base::Other
    }
}
