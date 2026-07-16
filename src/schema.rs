use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::plan::InterpError;
use crate::types::{Base, FieldType, Schema};

/// Extract a Schema from a Pydantic v2 model class's `model_fields`.
pub fn from_pydantic_model(py: Python<'_>, model_class: &Py<PyAny>) -> Result<Schema, InterpError> {
    let bound = model_class.bind(py);
    let fields = bound
        .getattr("model_fields")
        .map_err(|e| InterpError::Build(format!("Not a Pydantic v2 model class: {e}")))?;
    let dict = fields
        .cast::<PyDict>()
        .map_err(|e| InterpError::Build(format!("model_fields must be a dict: {e}")))?;

    let typing = PyModule::import(py, "typing")
        .map_err(|e| InterpError::Build(format!("Failed to import typing: {e}")))?;
    let types_module = PyModule::import(py, "types")
        .map_err(|e| InterpError::Build(format!("Failed to import types: {e}")))?;

    let mut schema = Schema::new();
    for (name, field_info) in dict.iter() {
        let name: String = name
            .extract()
            .map_err(|e| InterpError::Build(format!("Invalid field name: {e}")))?;
        let annotation = field_info
            .getattr("annotation")
            .map_err(|e| InterpError::Build(format!("Field '{name}' has no annotation: {e}")))?;
        let field_type = annotation_to_field_type(py, &typing, &types_module, &annotation)?;
        schema.insert(name, field_type);
    }
    Ok(schema)
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

    let mut schema = Schema::new();
    for name in names {
        let field = arrow_schema
            .call_method1("field", (name.as_str(),))
            .map_err(|e| InterpError::Build(format!("Failed to read field '{name}': {e}")))?;
        let nullable: bool = field
            .getattr("nullable")
            .and_then(|n| n.extract())
            .map_err(|e| {
                InterpError::Build(format!("Failed to read nullability of '{name}': {e}"))
            })?;
        let type_str: String = field
            .getattr("type")
            .and_then(|t| t.str())
            .map(|s| s.to_string())
            .map_err(|e| InterpError::Build(format!("Failed to read type of '{name}': {e}")))?;
        schema.insert(
            name,
            FieldType {
                base: arrow_type_to_base(&type_str),
                nullable,
            },
        );
    }
    Ok(schema)
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

/// Convert a Python type annotation (`int`, `str | None`,
/// `typing.Optional[int]`, ...) to a FieldType. Nullability is detected via
/// `typing.get_origin`/`get_args`, which works uniformly for both `X | None`
/// (`types.UnionType`) and `Optional[X]`/`Union[X, None]` (`typing.Union`) —
/// both produce args containing `NoneType` when nullable. A non-Union
/// generic (`list[int]`, `dict[str, int]`, ...) maps to `Base::Other`.
fn annotation_to_field_type(
    py: Python<'_>,
    typing: &Bound<'_, PyModule>,
    types_module: &Bound<'_, PyModule>,
    annotation: &Bound<'_, PyAny>,
) -> Result<FieldType, InterpError> {
    let origin = typing
        .call_method1("get_origin", (annotation,))
        .map_err(|e| InterpError::Build(format!("Failed to inspect type annotation: {e}")))?;

    if origin.is_none() {
        return Ok(FieldType {
            base: python_type_to_base(annotation),
            nullable: false,
        });
    }

    let is_union = origin.is(&typing
        .getattr("Union")
        .map_err(|e| InterpError::Build(format!("typing.Union missing: {e}")))?)
        || origin.is(&types_module
            .getattr("UnionType")
            .map_err(|e| InterpError::Build(format!("types.UnionType missing: {e}")))?);
    if !is_union {
        return Ok(FieldType {
            base: Base::Other,
            nullable: false,
        });
    }

    let args: Vec<Py<PyAny>> = typing
        .call_method1("get_args", (annotation,))
        .and_then(|a| a.extract())
        .map_err(|e| InterpError::Build(format!("Failed to inspect union args: {e}")))?;

    let none_type = py.None().bind(py).get_type();
    let mut non_none: Vec<Py<PyAny>> = Vec::new();
    let mut nullable = false;
    for arg in args {
        if arg.bind(py).is(&none_type) {
            nullable = true;
        } else {
            non_none.push(arg);
        }
    }

    let base = if non_none.len() == 1 {
        python_type_to_base(non_none[0].bind(py))
    } else {
        Base::Other
    };
    Ok(FieldType { base, nullable })
}

fn python_type_to_base(t: &Bound<'_, PyAny>) -> Base {
    match t
        .getattr("__name__")
        .ok()
        .and_then(|n| n.extract::<String>().ok())
    {
        Some(name) => match name.as_str() {
            "int" => Base::Int,
            "float" => Base::Float,
            "str" => Base::Str,
            "bool" => Base::Bool,
            _ => Base::Other,
        },
        None => Base::Other,
    }
}

/// Convert a FieldType into the Python type object `create_model` needs —
/// the inverse of `annotation_to_field_type`: `Optional[T]` if nullable,
/// else `T` directly. `Base::Other` maps to `typing.Any`.
pub fn field_type_to_python(py: Python<'_>, ft: FieldType) -> PyResult<Py<PyAny>> {
    let builtins = PyModule::import(py, "builtins")?;
    let typing = PyModule::import(py, "typing")?;
    let base_type: Py<PyAny> = match ft.base {
        Base::Int => builtins.getattr("int")?.unbind(),
        Base::Float => builtins.getattr("float")?.unbind(),
        Base::Str => builtins.getattr("str")?.unbind(),
        Base::Bool => builtins.getattr("bool")?.unbind(),
        // Placeholder — Task 3 turns this into a synthesized nested
        // pydantic model.
        Base::Other | Base::Struct(_) => typing.getattr("Any")?.unbind(),
        Base::List(inner) => {
            let inner_type = field_type_to_python(py, *inner)?;
            builtins.getattr("list")?.get_item(inner_type)?.unbind()
        }
    };
    if !ft.nullable {
        return Ok(base_type);
    }
    let none_type = py.None().bind(py).get_type().unbind();
    let union = typing.getattr("Union")?;
    // Subscript via the item-access protocol (Union[T, None]). Python 3.14 made
    // typing.Union a class, so call_method1("__getitem__", ...) hits the unbound
    // descriptor and passes the key as self; get_item binds correctly on 3.13/3.14.
    let optional = union.get_item((base_type, none_type))?;
    Ok(optional.unbind())
}
