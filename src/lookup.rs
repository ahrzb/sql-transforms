use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::expr::Value;
use crate::plan::InterpError;

pub struct LookupIndex {
    pub index: HashMap<Vec<Value>, HashMap<String, Value>>,
    pub value_columns: Vec<String>,
}

pub fn build_index(
    py: Python<'_>,
    table: &Py<PyAny>,
    key_columns: &[String],
) -> Result<LookupIndex, InterpError> {
    let bound = table.bind(py);
    let rows_obj = bound
        .call_method0("to_pylist")
        .map_err(|e| InterpError::Build(format!("Failed to read static table: {e}")))?;
    let rows: Vec<Py<PyAny>> = rows_obj.extract().map_err(|e| {
        InterpError::Build(format!("Static table must convert to a list of rows: {e}"))
    })?;

    let all_columns: Vec<String> = bound
        .getattr("column_names")
        .and_then(|c| c.extract())
        .map_err(|e| InterpError::Build(format!("Failed to read static table columns: {e}")))?;
    let value_columns: Vec<String> = all_columns
        .into_iter()
        .filter(|c| !key_columns.contains(c))
        .collect();

    let mut index = HashMap::new();
    for row_obj in rows {
        let row_bound = row_obj.bind(py);
        let dict = row_bound
            .cast::<PyDict>()
            .map_err(|e| InterpError::Build(format!("Static table row must be a dict: {e}")))?;

        let mut rest: HashMap<String, Value> = HashMap::new();
        for (k, v) in dict.iter() {
            let col: String = k
                .extract()
                .map_err(|e| InterpError::Build(format!("Static table column name error: {e}")))?;
            if key_columns.contains(&col) {
                continue;
            }
            let value = Value::from_pyobject(&v)
                .map_err(|e| InterpError::Build(format!("Static table value error: {e}")))?;
            rest.insert(col, value);
        }

        let mut key = Vec::with_capacity(key_columns.len());
        for col in key_columns {
            let v = dict
                .get_item(col)
                .map_err(|e| {
                    InterpError::Build(format!("Static table missing key column '{col}': {e}"))
                })?
                .ok_or_else(|| {
                    InterpError::Build(format!("Static table missing key column '{col}'"))
                })?;
            key.push(
                Value::from_pyobject(&v).map_err(|e| {
                    InterpError::Build(format!("Static table key value error: {e}"))
                })?,
            );
        }

        index.insert(key, rest);
    }

    Ok(LookupIndex {
        index,
        value_columns,
    })
}
