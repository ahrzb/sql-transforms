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

#[pymethods]
impl InferFn {
    #[new]
    fn new(
        py: Python<'_>,
        sql: String,
        row_tables: HashMap<String, Py<PyAny>>,
        static_tables: HashMap<String, Py<PyAny>>,
    ) -> PyResult<Self> {
        let raw_plan = plan::build_plan(&sql)?;
        let row_table_names: HashSet<String> = row_tables.keys().cloned().collect();
        let static_table_names: HashSet<String> = static_tables.keys().cloned().collect();
        let (optimized_plan, specs) = plan::optimize(raw_plan, &static_table_names)?;

        let mut row_schemas = HashMap::new();
        for (name, model_class) in &row_tables {
            row_schemas.insert(name.clone(), schema::from_pydantic_model(py, model_class)?);
        }
        let mut static_schemas = HashMap::new();
        for (name, table_obj) in &static_tables {
            static_schemas.insert(name.clone(), schema::from_arrow_table(py, table_obj)?);
        }

        let column_validation = plan::validate_columns(
            &optimized_plan,
            &row_table_names,
            &row_schemas,
            &static_schemas,
        )?;

        let output_model = synthesize_output_model(
            py,
            &optimized_plan.projection,
            &column_validation.effective_schemas,
        )?;

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
            output_model,
        })
    }

    fn infer(
        &self,
        py: Python<'_>,
        tables: HashMap<String, Vec<Py<PyAny>>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let empty: Vec<String> = Vec::new();
        let mut value_tables: HashMap<String, Vec<HashMap<String, Value>>> = HashMap::new();
        for (table, rows) in &tables {
            let columns = self.row_table_columns.get(table).unwrap_or(&empty);
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
                    row.insert(col.clone(), Value::from_pyobject(&attr)?);
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
