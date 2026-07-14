use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod expr;
mod expr_build;
mod plan;

use expr::Value;
use plan::Plan;

#[pyclass]
struct InferFn {
    plan: Plan,
}

#[pymethods]
impl InferFn {
    #[new]
    fn new(
        sql: String,
        row_tables: Vec<String>,
        static_tables: HashMap<String, Py<PyAny>>,
    ) -> PyResult<Self> {
        let _ = (&row_tables, &static_tables);
        let plan = plan::build_plan(&sql)?;
        Ok(InferFn { plan })
    }

    fn infer(
        &self,
        py: Python<'_>,
        tables: HashMap<String, Vec<Py<PyAny>>>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let mut value_tables: HashMap<String, Vec<HashMap<String, Value>>> = HashMap::new();
        for (table, rows) in &tables {
            let mut out_rows = Vec::with_capacity(rows.len());
            for row_obj in rows {
                let bound = row_obj.bind(py);
                let dict = bound.cast::<PyDict>()?;
                let mut row: HashMap<String, Value> = HashMap::new();
                for (k, v) in dict.iter() {
                    let key: String = k.extract()?;
                    row.insert(key, Value::from_pyobject(&v)?);
                }
                out_rows.push(row);
            }
            value_tables.insert(table.clone(), out_rows);
        }

        let result_rows = plan::execute(&self.plan, &value_tables)?;

        let mut out = Vec::with_capacity(result_rows.len());
        for row in &result_rows {
            let dict = PyDict::new(py);
            for (k, v) in row {
                dict.set_item(k, v.to_pyobject(py)?)?;
            }
            out.push(dict.unbind());
        }
        Ok(out)
    }
}

#[pymodule]
fn _interpreter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InferFn>()?;
    Ok(())
}
