use std::collections::HashMap;

use pyo3::prelude::*;

#[pyclass]
struct InferFn {
    sql: String,
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
        Ok(InferFn { sql })
    }

    fn infer(&self, tables: HashMap<String, Vec<Py<PyAny>>>) -> PyResult<Vec<Py<PyAny>>> {
        let _ = (&self.sql, &tables);
        Ok(Vec::new())
    }
}

#[pymodule]
fn _interpreter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InferFn>()?;
    Ok(())
}
