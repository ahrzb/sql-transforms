//! The DuckDB-semantics engine: `DuckDBInferFn`.
//!
//! Same Python API as `InferFn` minus the transformer callout: SQL +
//! pydantic row model + pyarrow static tables in the constructor, then
//! `infer()` maps row objects through the specializer's compiled program.
//! `prepare()` does the SQL work (frontend -> lower -> verify); this module
//! is only the Python boundary — schema extraction on the way in, map
//! materialization for the join probes, output-model rows on the way out.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::InterpError;
use crate::schema;
use crate::specializer::exec::interp::{compile, InterpFn};
use crate::specializer::exec::{Batch, ColData, KeyBits, OutCol, ScalarVal, StaticData};
use crate::specializer::ir::{Col, ColTy, StaticTy, Ty};
use crate::specializer::plan::StaticTable;
use crate::specializer::{prepare, StaticSpec};
use crate::types::{Base, FieldType};

/// The scalar slice of the type lattice the specializer handles. `None`
/// means the column can't cross this boundary (struct/list/Other).
fn base_to_ty(b: &Base) -> Option<Ty> {
    match b {
        Base::Bool => Some(Ty::I1),
        Base::Int => Some(Ty::I64),
        Base::Float => Some(Ty::F64),
        Base::Str => Some(Ty::Str),
        _ => None,
    }
}

fn ty_to_base(t: Ty) -> Base {
    match t {
        Ty::I1 => Base::Bool,
        Ty::I64 => Base::Int,
        Ty::F64 => Base::Float,
        Ty::Str => Base::Str,
    }
}

fn build_err(msg: impl Into<String>) -> PyErr {
    InterpError::Build(msg.into()).into()
}

/// One map static from a pyarrow Table, per its `StaticSpec` recipe: rows
/// with a NULL key are dropped (a NULL never equi-matches), a NULL in a
/// value column is an error, and an int column joined against a float
/// expression converts here (the declared key type is the expression's).
fn materialize_map(
    py: Python<'_>,
    table: &Py<PyAny>,
    spec: &StaticSpec,
    key_tys: &[Ty],
    val_tys: &[Ty],
) -> PyResult<StaticData> {
    let rows = table.bind(py).call_method0("to_pylist")?;
    let mut entries = Vec::new();
    'row: for item in rows.try_iter()? {
        let row = item?;
        let row = row.cast::<PyDict>().map_err(|_| {
            build_err(format!(
                "static table '{}': to_pylist row is not a dict",
                spec.table
            ))
        })?;
        let get = |name: &str| {
            row.get_item(name)?.ok_or_else(|| {
                build_err(format!(
                    "static table '{}' row is missing column '{name}'",
                    spec.table
                ))
            })
        };
        let mut keys = Vec::with_capacity(key_tys.len());
        for (name, ty) in spec.key_cols.iter().zip(key_tys) {
            let v = get(name)?;
            if v.is_none() {
                continue 'row;
            }
            keys.push(match ty {
                Ty::I1 => KeyBits::I1(v.extract()?),
                Ty::I64 => KeyBits::I64(v.extract()?),
                Ty::F64 => KeyBits::F64(v.extract::<f64>()?.to_bits()),
                Ty::Str => KeyBits::Str(v.extract()?),
            });
        }
        let mut vals = Vec::with_capacity(val_tys.len());
        for (name, ty) in spec.val_cols.iter().zip(val_tys) {
            let v = get(name)?;
            if v.is_none() {
                return Err(build_err(format!(
                    "static table '{}' has a NULL in value column '{name}' — joins to NULL \
                     values are not supported",
                    spec.table
                )));
            }
            vals.push(match ty {
                Ty::I1 => ScalarVal::I1(v.extract()?),
                Ty::I64 => ScalarVal::I64(v.extract()?),
                Ty::F64 => ScalarVal::F64(v.extract()?),
                Ty::Str => ScalarVal::Str(v.extract()?),
            });
        }
        entries.push((keys, vals));
    }
    Ok(StaticData::Map(entries))
}

fn synthesize_output_model(py: Python<'_>, out_cols: &[Col]) -> PyResult<Py<PyAny>> {
    let create_model = PyModule::import(py, "pydantic")?.getattr("create_model")?;
    let ellipsis = PyModule::import(py, "builtins")?.getattr("Ellipsis")?;
    let kwargs = PyDict::new(py);
    for c in out_cols {
        let ft = FieldType {
            base: ty_to_base(c.ty.ty),
            nullable: c.ty.nullable,
        };
        kwargs.set_item(&c.name, (schema::field_type_to_python(py, ft)?, &ellipsis))?;
    }
    Ok(create_model.call(("OutputRow",), Some(&kwargs))?.unbind())
}

#[pyclass(unsendable)]
pub struct DuckDBInferFn {
    fun: InterpFn,
    row_table: String,
    in_cols: Vec<Col>,
    out_cols: Vec<Col>,
    #[pyo3(get)]
    output_model: Py<PyAny>,
}

#[pymethods]
impl DuckDBInferFn {
    #[new]
    #[pyo3(signature = (sql, row_tables, static_tables, output_model=None))]
    fn new(
        py: Python<'_>,
        sql: String,
        row_tables: HashMap<String, Py<PyAny>>,
        static_tables: HashMap<String, Py<PyAny>>,
        output_model: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let (row_table, model) = match row_tables.len() {
            1 => row_tables.into_iter().next().unwrap(),
            n => {
                return Err(build_err(format!(
                    "unsupported: the specializer takes exactly one row table, got {n}"
                )))
            }
        };
        let mut in_cols = Vec::new();
        for (name, ft) in schema::pydantic_model_fields_ordered(py, &model)? {
            let ty = base_to_ty(&ft.base).ok_or_else(|| {
                build_err(format!(
                    "unsupported: row column '{name}' has a non-scalar type"
                ))
            })?;
            in_cols.push(Col {
                name,
                ty: ColTy {
                    ty,
                    nullable: ft.nullable,
                },
            });
        }

        // Non-scalar static columns are omitted from the catalog rather than
        // rejected: unreferenced ones cost nothing, referenced ones fail the
        // bind by name.
        let mut catalog = Vec::new();
        for (name, table) in &static_tables {
            let schema_obj = table
                .bind(py)
                .getattr("schema")
                .map_err(|e| {
                    build_err(format!("static table '{name}' is not a pyarrow.Table: {e}"))
                })?
                .unbind();
            let mut cols = Vec::new();
            for (cname, ft) in schema::arrow_schema_to_ordered_fields(py, &schema_obj)? {
                if let Some(ty) = base_to_ty(&ft.base) {
                    cols.push(Col {
                        name: cname,
                        ty: ColTy {
                            ty,
                            nullable: ft.nullable,
                        },
                    });
                }
            }
            catalog.push(StaticTable {
                name: name.clone(),
                cols,
            });
        }

        let prepared =
            prepare(&sql, &row_table, &in_cols, &catalog).map_err(|e| build_err(e.to_string()))?;

        // Program statics and StaticSpecs are both indexed by join id.
        let mut data = Vec::with_capacity(prepared.statics.len());
        for (spec, sty) in prepared.statics.iter().zip(&prepared.program.statics) {
            let StaticTy::Map { keys, values } = sty else {
                return Err(build_err("internal: v0 lowering emits only map statics"));
            };
            let table = static_tables
                .get(&spec.table)
                .expect("spec names come from the catalog");
            data.push(materialize_map(py, table, spec, keys, values)?);
        }

        let fun = compile(&prepared.program, data).map_err(|e| build_err(e.to_string()))?;
        let output_model = match output_model {
            // Supplied models are trusted as-is in v0 (no shape validation).
            Some(m) => m,
            None => synthesize_output_model(py, &prepared.program.out_cols)?,
        };
        Ok(DuckDBInferFn {
            fun,
            row_table,
            in_cols,
            out_cols: prepared.program.out_cols.clone(),
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
                merged.insert(k.extract()?, v.extract()?);
            }
        }
        if let Some(bad) = merged.keys().find(|k| **k != self.row_table) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown table '{bad}' (the row table is '{}')",
                self.row_table
            )));
        }
        let rows = merged.remove(&self.row_table).unwrap_or_default();

        let n = rows.len();
        let mut cols: Vec<ColData> = self
            .in_cols
            .iter()
            .map(|c| match c.ty.ty {
                Ty::I1 => ColData::I1 {
                    valid: Vec::with_capacity(n),
                    data: Vec::with_capacity(n),
                },
                Ty::I64 => ColData::I64 {
                    valid: Vec::with_capacity(n),
                    data: Vec::with_capacity(n),
                },
                Ty::F64 => ColData::F64 {
                    valid: Vec::with_capacity(n),
                    data: Vec::with_capacity(n),
                },
                Ty::Str => ColData::Str {
                    valid: Vec::with_capacity(n),
                    data: Vec::with_capacity(n),
                },
            })
            .collect();
        for row_obj in &rows {
            let bound = row_obj.bind(py);
            for (c, col) in self.in_cols.iter().zip(&mut cols) {
                let attr = bound.getattr(c.name.as_str()).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Row for table '{}' is missing attribute '{}': {e}",
                        self.row_table, c.name
                    ))
                })?;
                let null = attr.is_none();
                if null && !c.ty.nullable {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "column '{}' is not nullable but a row has None",
                        c.name
                    )));
                }
                match col {
                    ColData::I1 { valid, data } => {
                        valid.push(!null);
                        data.push(if null { false } else { attr.extract()? });
                    }
                    ColData::I64 { valid, data } => {
                        valid.push(!null);
                        data.push(if null { 0 } else { attr.extract()? });
                    }
                    ColData::F64 { valid, data } => {
                        valid.push(!null);
                        data.push(if null { 0.0 } else { attr.extract()? });
                    }
                    ColData::Str { valid, data } => {
                        valid.push(!null);
                        data.push(if null { String::new() } else { attr.extract()? });
                    }
                }
            }
        }

        let batch = Batch { rows: n, cols };
        let mut st = self.fun.new_state();
        self.fun
            .run(&batch, &mut st)
            .map_err(|t| PyErr::from(InterpError::Eval(t.0)))?;

        let model = self.output_model.bind(py);
        let mut out = Vec::with_capacity(st.emitted);
        for r in 0..st.emitted {
            let dict = PyDict::new(py);
            for (c, oc) in self.out_cols.iter().zip(&st.out) {
                match oc {
                    OutCol::I1(v) => {
                        let (ok, x) = v[r];
                        dict.set_item(&c.name, ok.then_some(x))?;
                    }
                    OutCol::I64(v) => {
                        let (ok, x) = v[r];
                        dict.set_item(&c.name, ok.then_some(x))?;
                    }
                    OutCol::F64(v) => {
                        let (ok, x) = v[r];
                        dict.set_item(&c.name, ok.then_some(x))?;
                    }
                    OutCol::Str(v) => {
                        let (ok, s) = v[r];
                        dict.set_item(&c.name, ok.then(|| st.arena.get(s)))?;
                    }
                }
            }
            out.push(model.call_method1("model_validate", (dict,))?.unbind());
        }
        Ok(out)
    }
}
