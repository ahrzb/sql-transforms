use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::plan::InterpError;
use crate::types::{Base, FieldType, Schema};

/// Extract a Schema from a Pydantic v2 model class's `model_fields`.
pub fn from_pydantic_model(py: Python<'_>, model_class: &Py<PyAny>) -> Result<Schema, InterpError> {
    Ok(pydantic_model_fields_ordered(py, model_class)?
        .into_iter()
        .collect())
}

/// Extract a Pydantic v2 model class's fields as an order-preserving Vec
/// (declaration order, from `model_fields` dict iteration order). `Schema`
/// is a `HashMap` (column lookup by name doesn't care about order), but a
/// nested struct's field order is semantically significant — it feeds
/// `Base::Struct`/`Value::Struct`, whose `PartialEq`/`Hash` are order-
/// sensitive (struct equality, join keys). Two independently-parsed structs
/// with the same fields must agree on field order, so this must NOT round-
/// trip through the `HashMap`-backed `Schema` (whose iteration order is
/// randomized per-instance) before becoming a `Base::Struct` — see
/// `annotation_to_field_type`'s nested-model branch below.
fn pydantic_model_fields_ordered(
    py: Python<'_>,
    model_class: &Py<PyAny>,
) -> Result<Vec<(String, FieldType)>, InterpError> {
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

    let mut out = Vec::new();
    for (name, field_info) in dict.iter() {
        let name: String = name
            .extract()
            .map_err(|e| InterpError::Build(format!("Invalid field name: {e}")))?;
        let annotation = field_info
            .getattr("annotation")
            .map_err(|e| InterpError::Build(format!("Field '{name}' has no annotation: {e}")))?;
        let field_type = annotation_to_field_type(py, &typing, &types_module, &annotation)?;
        out.push((name, field_type));
    }
    Ok(out)
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
            InterpError::Build(format!("transformer output schema is not a pyarrow.Schema: {e}"))
        })?;
    let pa_types = PyModule::import(py, "pyarrow.types")
        .map_err(|e| InterpError::Build(format!("Failed to import pyarrow.types: {e}")))?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let field = bound
            .call_method1("field", (name.as_str(),))
            .map_err(|e| InterpError::Build(format!("Failed to read output field '{name}': {e}")))?;
        let ft = arrow_field_to_field_type(&pa_types, &field)
            .map_err(|e| InterpError::Build(format!("Failed to read type of output field '{name}': {e}")))?;
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
    if pa_types.call_method1("is_struct", (ty,))?.extract::<bool>()? {
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
        || pa_types.call_method1("is_large_list", (ty,))?.extract::<bool>()?;
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
        if is_pydantic_model_class(py, annotation)? {
            let nested = pydantic_model_fields_ordered(py, &annotation.clone().unbind())?;
            return Ok(FieldType {
                base: Base::Struct(nested),
                nullable: false,
            });
        }
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
    if is_union {
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
            let inner =
                annotation_to_field_type(py, typing, types_module, non_none[0].bind(py))?;
            nullable = nullable || inner.nullable;
            inner.base
        } else {
            Base::Other
        };
        return Ok(FieldType { base, nullable });
    }

    let builtins = PyModule::import(py, "builtins")
        .map_err(|e| InterpError::Build(format!("Failed to import builtins: {e}")))?;
    let is_list_generic = origin.is(&builtins
        .getattr("list")
        .map_err(|e| InterpError::Build(format!("builtins.list missing: {e}")))?);
    if is_list_generic {
        let args: Vec<Py<PyAny>> = typing
            .call_method1("get_args", (annotation,))
            .and_then(|a| a.extract())
            .map_err(|e| InterpError::Build(format!("Failed to inspect list[..] args: {e}")))?;
        if let [elem] = args.as_slice() {
            let inner = annotation_to_field_type(py, typing, types_module, elem.bind(py))?;
            return Ok(FieldType {
                base: Base::List(Box::new(inner)),
                nullable: false,
            });
        }
    }

    Ok(FieldType {
        base: Base::Other,
        nullable: false,
    })
}

/// Is `annotation` a `pydantic.BaseModel` subclass (a nested struct field)?
/// `issubclass()` raises `TypeError` for a non-class annotation (e.g.
/// `typing.Any`, `list[int]`) — treated as "not a model", not an error.
fn is_pydantic_model_class(py: Python<'_>, annotation: &Bound<'_, PyAny>) -> Result<bool, InterpError> {
    let pydantic = PyModule::import(py, "pydantic")
        .map_err(|e| InterpError::Build(format!("Failed to import pydantic: {e}")))?;
    let base_model = pydantic
        .getattr("BaseModel")
        .map_err(|e| InterpError::Build(format!("pydantic.BaseModel missing: {e}")))?;
    let builtins = PyModule::import(py, "builtins")
        .map_err(|e| InterpError::Build(format!("Failed to import builtins: {e}")))?;
    match builtins.call_method1("issubclass", (annotation, &base_model)) {
        Ok(result) => result
            .extract()
            .map_err(|e| InterpError::Build(format!("issubclass result error: {e}"))),
        Err(_) => Ok(false),
    }
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
        Base::Other => typing.getattr("Any")?.unbind(),
        Base::Struct(fields) => {
            let pydantic = PyModule::import(py, "pydantic")?;
            let create_model = pydantic.getattr("create_model")?;
            let ellipsis = builtins.getattr("Ellipsis")?;
            let kwargs = PyDict::new(py);
            for (name, field_ft) in fields {
                let field_py_type = field_type_to_python(py, field_ft)?;
                kwargs.set_item(name, (field_py_type, &ellipsis))?;
            }
            create_model.call(("StructModel",), Some(&kwargs))?.unbind()
        }
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
