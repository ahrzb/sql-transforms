use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod expr;
mod expr_build;
mod lookup;
mod plan;
mod schema;
mod types;

use expr::Expr;
use expr::Value;
use lookup::LookupIndex;
use plan::Plan;

#[pyclass]
struct InferFn {
    plan: Plan,
    lookups: HashMap<String, LookupIndex>,
    row_table_columns: HashMap<String, Vec<String>>,
    row_schemas: HashMap<String, types::Schema>,
    #[pyo3(get)]
    output_model: Py<PyAny>,
}

fn synthesize_output_model(
    py: Python<'_>,
    projection: &[(String, Expr)],
    schemas: &HashMap<String, types::Schema>,
) -> PyResult<Py<PyAny>> {
    let pydantic = PyModule::import(py, "pydantic")?;
    let create_model = pydantic.getattr("create_model")?;
    let builtins = PyModule::import(py, "builtins")?;
    let ellipsis = builtins.getattr("Ellipsis")?;

    let kwargs = PyDict::new(py);
    for (alias, expr) in projection {
        let ft = types::infer_type(expr, schemas)?;
        let py_type = schema::field_type_to_python(py, ft)?;
        kwargs.set_item(alias, (py_type, &ellipsis))?;
    }
    let model = create_model.call(("OutputRow",), Some(&kwargs))?;
    Ok(model.unbind())
}

/// Validates a caller-supplied `output_model` against the query's inferred
/// output shape. Only rejects what can be *proven* wrong at build time: a
/// missing/extra field vs. the projection's aliases, or a base-type mismatch
/// `types::compatible` can't excuse. Nullability mismatches are never a
/// build-time error — see `types::compatible`'s docs and Task 2's
/// deliberately conservative `infer_type`.
fn validate_output_model(
    py: Python<'_>,
    model: &Py<PyAny>,
    projection: &[(String, Expr)],
    schemas: &HashMap<String, types::Schema>,
) -> PyResult<()> {
    let declared_schema = schema::from_pydantic_model(py, model)?;

    let mut projected_aliases: HashSet<String> = HashSet::new();
    for (alias, expr) in projection {
        projected_aliases.insert(alias.clone());
        let declared = declared_schema.get(alias).ok_or_else(|| {
            plan::InterpError::Build(format!(
                "output_model is missing field '{alias}' produced by the query"
            ))
        })?;
        let inferred = types::infer_type(expr, schemas)?;
        if !types::compatible(&inferred.base, &declared.base) {
            return Err(plan::InterpError::Build(format!(
                "output_model field '{alias}' is declared as a type incompatible with the \
                 query's inferred output ({:?} vs declared {:?})",
                inferred.base, declared.base
            ))
            .into());
        }
    }

    let declared_fields: HashSet<String> = declared_schema.keys().cloned().collect();
    let extra: Vec<&String> = declared_fields.difference(&projected_aliases).collect();
    if !extra.is_empty() {
        return Err(plan::InterpError::Build(format!(
            "output_model declares fields not produced by the query: {extra:?}"
        ))
        .into());
    }
    Ok(())
}

/// A `transformers` registry entry resolved at build time: the fitted object,
/// its `feature_names_in_` (input alignment order), and the caller-declared
/// output field list (marshalling target, order significant).
struct ResolvedTransformer {
    obj: std::sync::Arc<Py<PyAny>>,
    input_features: Vec<String>,
    output_fields: Vec<(String, types::FieldType)>,
}

/// Reads `obj.feature_names_in_.tolist()`. Absence is a clear build error:
/// the object was fit on bare arrays, so we cannot align inputs by name.
fn read_feature_names_in(py: Python<'_>, obj: &Py<PyAny>) -> Result<Vec<String>, plan::InterpError> {
    let bound = obj.bind(py);
    let attr = bound.getattr("feature_names_in_").map_err(|_| {
        plan::InterpError::Build(
            "transformer has no `feature_names_in_`; fit it on named data \
             (e.g. a pandas DataFrame) so input columns can be aligned by name"
                .to_string(),
        )
    })?;
    attr.call_method0("tolist")
        .and_then(|l| l.extract::<Vec<String>>())
        .map_err(|e| plan::InterpError::Build(format!("could not read feature_names_in_: {e}")))
}

/// Rewrites every `Expr::Function` whose name is a registered transformer into
/// an `Expr::Transform`, recursing through the whole expression tree so a
/// transformer call nested inside arithmetic is still resolved. A transformer
/// call must have exactly one argument.
fn resolve_transformers(
    expr: Expr,
    resolved: &HashMap<String, ResolvedTransformer>,
) -> Result<Expr, plan::InterpError> {
    match expr {
        Expr::Function { name, args } => {
            let mut new_args = Vec::with_capacity(args.len());
            for a in args {
                new_args.push(resolve_transformers(a, resolved)?);
            }
            if let Some(rt) = resolved.get(&name) {
                if new_args.len() != 1 {
                    return Err(plan::InterpError::Build(format!(
                        "transformer '{name}' takes exactly one argument, got {}",
                        new_args.len()
                    )));
                }
                let arg = new_args.into_iter().next().unwrap();
                return Ok(Expr::Transform {
                    obj: rt.obj.clone(),
                    input_features: rt.input_features.clone(),
                    output_fields: rt.output_fields.clone(),
                    arg: Box::new(arg),
                });
            }
            Ok(Expr::Function { name, args: new_args })
        }
        Expr::BinaryOp { op, left, right } => Ok(Expr::BinaryOp {
            op,
            left: Box::new(resolve_transformers(*left, resolved)?),
            right: Box::new(resolve_transformers(*right, resolved)?),
        }),
        Expr::Not(inner) => Ok(Expr::Not(Box::new(resolve_transformers(*inner, resolved)?))),
        Expr::Cast { expr, target } => Ok(Expr::Cast {
            expr: Box::new(resolve_transformers(*expr, resolved)?),
            target,
        }),
        Expr::Struct(fields) => {
            let mut out = Vec::with_capacity(fields.len());
            for (k, v) in fields {
                out.push((k, resolve_transformers(v, resolved)?));
            }
            Ok(Expr::Struct(out))
        }
        Expr::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for e in items {
                out.push(resolve_transformers(e, resolved)?);
            }
            Ok(Expr::List(out))
        }
        Expr::FieldAccess { base, field } => Ok(Expr::FieldAccess {
            base: Box::new(resolve_transformers(*base, resolved)?),
            field,
        }),
        // Column, Literal, and an already-built Transform pass through.
        other => Ok(other),
    }
}

#[pymethods]
impl InferFn {
    #[new]
    #[pyo3(signature = (sql, row_tables, static_tables, output_model=None, transformers=None))]
    fn new(
        py: Python<'_>,
        sql: String,
        row_tables: HashMap<String, Py<PyAny>>,
        static_tables: HashMap<String, Py<PyAny>>,
        output_model: Option<Py<PyAny>>,
        transformers: Option<HashMap<String, (Py<PyAny>, Py<PyAny>)>>,
    ) -> PyResult<Self> {
        let raw_plan = plan::build_plan(&sql)?;
        let row_table_names: HashSet<String> = row_tables.keys().cloned().collect();
        let static_table_names: HashSet<String> = static_tables.keys().cloned().collect();
        let (mut optimized_plan, specs) = plan::optimize(raw_plan, &static_table_names)?;

        let mut row_schemas = HashMap::new();
        for (name, model_class) in &row_tables {
            row_schemas.insert(name.clone(), schema::from_pydantic_model(py, model_class)?);
        }
        let mut static_schemas = HashMap::new();
        for (name, table_obj) in &static_tables {
            static_schemas.insert(name.clone(), schema::from_arrow_table(py, table_obj)?);
        }

        let column_validation = plan::validate_columns(
            &mut optimized_plan,
            &row_table_names,
            &row_schemas,
            &static_schemas,
        )?;

        // Resolve registered transformers AFTER column validation (so
        // validate_columns sees the plain Expr::Function and its named_struct
        // arg -- no Transform arm needed there) and BEFORE output-model
        // synthesis (so infer_type sees Expr::Transform and returns the
        // declared output struct type). Reads feature_names_in_ here (with py).
        let transformers = transformers.unwrap_or_default();
        if !transformers.is_empty() {
            let mut resolved: HashMap<String, ResolvedTransformer> = HashMap::new();
            for (name, (obj, out_schema_obj)) in &transformers {
                let input_features = read_feature_names_in(py, obj)?;
                let output_fields = schema::arrow_schema_to_ordered_fields(py, out_schema_obj)?;
                resolved.insert(
                    name.clone(),
                    ResolvedTransformer {
                        obj: std::sync::Arc::new(obj.clone_ref(py)),
                        input_features,
                        output_fields,
                    },
                );
            }
            let projection = std::mem::take(&mut optimized_plan.projection);
            let mut new_projection = Vec::with_capacity(projection.len());
            for (alias, expr) in projection {
                new_projection.push((alias, resolve_transformers(expr, &resolved)?));
            }
            optimized_plan.projection = new_projection;
        }

        let output_model = match output_model {
            Some(supplied) => {
                validate_output_model(
                    py,
                    &supplied,
                    &optimized_plan.projection,
                    &column_validation.effective_schemas,
                )?;
                supplied
            }
            None => synthesize_output_model(
                py,
                &optimized_plan.projection,
                &column_validation.effective_schemas,
            )?,
        };

        let mut lookups = HashMap::new();
        for spec in specs {
            let table_obj = static_tables.get(&spec.static_table).ok_or_else(|| {
                plan::InterpError::Build(format!(
                    "SQL references static table '{}' that was not provided",
                    spec.static_table
                ))
            })?;
            let index = lookup::build_index(py, table_obj, &spec.key_columns)?;
            lookups.insert(spec.static_table, index);
        }

        Ok(InferFn {
            plan: optimized_plan,
            lookups,
            row_table_columns: column_validation.row_table_columns,
            row_schemas,
            output_model,
        })
    }

    #[pyo3(signature = (tables=None, **kwargs))]
    fn infer(
        &self,
        py: Python<'_>,
        tables: Option<HashMap<String, Vec<Py<PyAny>>>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let mut merged: HashMap<String, Vec<Py<PyAny>>> = tables.unwrap_or_default();
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                let rows: Vec<Py<PyAny>> = v.extract()?;
                merged.insert(key, rows);
            }
        }

        let empty: Vec<String> = Vec::new();
        let mut value_tables: HashMap<String, Vec<HashMap<String, Value>>> = HashMap::new();
        for (table, rows) in &merged {
            let columns = self.row_table_columns.get(table).unwrap_or(&empty);
            let schema = self.row_schemas.get(table);
            let mut out_rows = Vec::with_capacity(rows.len());
            for row_obj in rows {
                let bound = row_obj.bind(py);
                let mut row: HashMap<String, Value> = HashMap::new();
                for col in columns {
                    let attr = bound.getattr(col.as_str()).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Row for table '{table}' is missing attribute '{col}': {e}"
                        ))
                    })?;
                    let value = match schema.and_then(|s| s.get(col)) {
                        Some(ft) => Value::from_pyobject_typed(&attr, &ft.base)?,
                        None => Value::from_pyobject(&attr)?,
                    };
                    row.insert(col.clone(), value);
                }
                out_rows.push(row);
            }
            value_tables.insert(table.clone(), out_rows);
        }

        let result_rows = plan::execute(&self.plan, &value_tables, &self.lookups)?;

        let output_model = self.output_model.bind(py);
        let mut out = Vec::with_capacity(result_rows.len());
        for row in &result_rows {
            let dict = PyDict::new(py);
            for (k, v) in row {
                dict.set_item(k, v.to_pyobject(py)?)?;
            }
            let instance = output_model.call_method1("model_validate", (dict,))?;
            out.push(instance.unbind());
        }
        Ok(out)
    }
}

#[pymodule]
fn _interpreter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InferFn>()?;
    Ok(())
}
