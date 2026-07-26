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
use pyo3::types::{PyDict, PyString};

use crate::error::InterpError;
use crate::schema;
use crate::specializer::exec::cranelift::{self, CraneliftFn};
use crate::specializer::exec::interp::{compile, InterpFn};
use crate::specializer::exec::{Batch, ColData, KeyBits, OutCol, ScalarVal, StaticData};
use crate::specializer::exec::{RunState, Trap};
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

fn model_from_fields(py: Python<'_>, fields: Vec<(String, FieldType)>) -> PyResult<Py<PyAny>> {
    let create_model = PyModule::import(py, "pydantic")?.getattr("create_model")?;
    let ellipsis = PyModule::import(py, "builtins")?.getattr("Ellipsis")?;
    let kwargs = PyDict::new(py);
    for (name, ft) in fields {
        kwargs.set_item(name, (schema::field_type_to_python(py, ft)?, &ellipsis))?;
    }
    Ok(create_model.call(("OutputRow",), Some(&kwargs))?.unbind())
}

fn synthesize_output_model(py: Python<'_>, out_cols: &[Col]) -> PyResult<Py<PyAny>> {
    let fields = out_cols
        .iter()
        .map(|c| {
            (
                c.name.clone(),
                FieldType {
                    base: ty_to_base(c.ty.ty),
                    nullable: c.ty.nullable,
                },
            )
        })
        .collect();
    model_from_fields(py, fields)
}

/// AC #2's constant emitter: a static-tables-only query is evaluated ONCE,
/// here at build time, by DuckDB itself — nothing dynamic remains and no IR
/// is built at all. Statics materialize as native tables (duckdb's
/// registered-arrow scan path has divergent filter semantics — see the
/// builtin-pins spec). Returns the fixed row dicts plus the result schema.
fn eval_static_only(
    py: Python<'_>,
    sql: &str,
    static_tables: &HashMap<String, Py<PyAny>>,
) -> PyResult<(Vec<Py<PyAny>>, Vec<(String, FieldType)>)> {
    let duckdb = PyModule::import(py, "duckdb")?;
    let con = duckdb.call_method0("connect")?;
    for (name, table) in static_tables {
        con.call_method1("register", (format!("__arrow_{name}"), table))?;
        con.call_method1(
            "execute",
            (format!(
                "CREATE TABLE \"{name}\" AS SELECT * FROM \"__arrow_{name}\""
            ),),
        )?;
    }
    let arrow = con
        .call_method1("execute", (sql,))?
        .call_method0("to_arrow_table")?;
    let schema_obj = arrow.getattr("schema")?.unbind();
    let fields = schema::arrow_schema_to_ordered_fields(py, &schema_obj)?;
    let mut rows = Vec::new();
    for r in arrow.call_method0("to_pylist")?.try_iter()? {
        rows.push(r?.unbind());
    }
    Ok((rows, fields))
}

/// The execution backend: cranelift when it compiles, the interpreter as
/// the always-available fallback (AC #2 — an uncovered op must not fail
/// prepare). Both agree byte-for-byte by the 500-seed differential.
enum Backend {
    Cranelift(CraneliftFn),
    Interp(InterpFn),
}

impl Backend {
    fn name(&self) -> &'static str {
        match self {
            Backend::Cranelift(_) => "cranelift",
            Backend::Interp(_) => "interpreter",
        }
    }
    fn new_state(&self) -> RunState {
        match self {
            Backend::Cranelift(f) => f.new_state(),
            Backend::Interp(f) => f.new_state(),
        }
    }
    fn run(&self, input: &Batch, st: &mut RunState) -> Result<(), Trap> {
        match self {
            Backend::Cranelift(f) => f.run(input, st),
            Backend::Interp(f) => f.run(input, st),
        }
    }
}

/// The generated row marshaller (design doc §3 flag 1): everything about the
/// boundary that is knowable at prepare time is done at prepare time —
/// interned attribute-name objects in fixed field order, `model_construct`
/// resolved once, input buffers and run state owned and reused (cleared, not
/// dropped, per call). The generic path stays available behind
/// `SPECIALIZER_GENERIC_BOUNDARY` as the measured baseline.
struct Marshaller {
    in_names: Vec<Py<PyString>>,
    out_names: Vec<Py<PyString>>,
    /// `output_model.model_construct`, resolved at build. Outputs come out
    /// of the typed engine already conformant; re-validating them per row
    /// was the single largest boundary cost.
    construct: Py<PyAny>,
    cols: Vec<ColData>,
    state: RunState,
}

impl Marshaller {
    fn build(
        py: Python<'_>,
        in_cols: &[Col],
        out_cols: &[Col],
        output_model: &Py<PyAny>,
        fun: &Backend,
    ) -> PyResult<Marshaller> {
        Ok(Marshaller {
            in_names: in_cols
                .iter()
                .map(|c| PyString::intern(py, &c.name).unbind())
                .collect(),
            out_names: out_cols
                .iter()
                .map(|c| PyString::intern(py, &c.name).unbind())
                .collect(),
            construct: output_model.bind(py).getattr("model_construct")?.unbind(),
            cols: in_cols.iter().map(|c| ColData::new(c.ty.ty)).collect(),
            state: fun.new_state(),
        })
    }

    /// The hot path: fill reused columns from row objects (dict or model),
    /// run, emit via `model_construct`. Steady-state this allocates only the
    /// output objects — buffers and arena reset, never drop.
    fn call(
        &mut self,
        py: Python<'_>,
        fun: &Backend,
        in_cols: &[Col],
        rows: &[Py<PyAny>],
        row_table: &str,
    ) -> PyResult<Vec<Py<PyAny>>> {
        for col in &mut self.cols {
            col.clear();
        }
        for row_obj in rows {
            let bound = row_obj.bind(py);
            let dict = bound.cast::<PyDict>().ok();
            for ((c, name), col) in in_cols.iter().zip(&self.in_names).zip(&mut self.cols) {
                let attr = match dict {
                    Some(d) => d.get_item(name.bind(py))?.ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Row for table '{row_table}' is missing attribute '{}'",
                            c.name
                        ))
                    })?,
                    None => bound.getattr(name.bind(py)).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Row for table '{row_table}' is missing attribute '{}': {e}",
                            c.name
                        ))
                    })?,
                };
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
                    s @ ColData::Str { .. } => {
                        if null {
                            s.push_str_cell(false, "");
                        } else {
                            s.push_str_cell(true, attr.extract()?);
                        }
                    }
                }
            }
        }

        // The batch borrows the reused columns for the duration of the run;
        // mem::take + restore keeps `Batch` an owning type (an empty Vec
        // does not allocate).
        let batch = Batch {
            rows: rows.len(),
            cols: std::mem::take(&mut self.cols),
        };
        let res = fun.run(&batch, &mut self.state);
        self.cols = batch.cols;
        res.map_err(|t| PyErr::from(InterpError::Eval(t.0)))?;

        let construct = self.construct.bind(py);
        let mut out = Vec::with_capacity(self.state.emitted);
        for r in 0..self.state.emitted {
            let d = PyDict::new(py);
            for (name, oc) in self.out_names.iter().zip(&self.state.out) {
                let k = name.bind(py);
                match oc {
                    OutCol::I1(v) => {
                        let (ok, x) = v[r];
                        d.set_item(k, ok.then_some(x))?;
                    }
                    OutCol::I64(v) => {
                        let (ok, x) = v[r];
                        d.set_item(k, ok.then_some(x))?;
                    }
                    OutCol::F64(v) => {
                        let (ok, x) = v[r];
                        d.set_item(k, ok.then_some(x))?;
                    }
                    OutCol::Str(v) => {
                        let (ok, s) = v[r];
                        d.set_item(k, ok.then(|| self.state.arena.get(s)))?;
                    }
                }
            }
            out.push(construct.call((), Some(&d))?.unbind());
        }
        Ok(out)
    }
}

enum Engine {
    Compiled {
        fun: Backend,
        in_cols: Vec<Col>,
        out_cols: Vec<Col>,
        /// `None` when `SPECIALIZER_GENERIC_BOUNDARY` pinned the generic
        /// boundary at construction (the bench baseline).
        marsh: Option<Marshaller>,
    },
    /// Fixed row dicts from a static-only query, re-validated through the
    /// output model on every `infer` call.
    Constant { rows: Vec<Py<PyAny>> },
}

#[pyclass(unsendable)]
pub struct DuckDBInferFn {
    engine: Engine,
    row_table: String,
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

        use super::specializer::PrepareError;
        let prepared = match prepare(&sql, &row_table, &in_cols, &catalog) {
            Ok(p) => p,
            // Unsupported/unparseable SQL might still be a static-tables-only
            // query (static driving table, aggregation, ORDER BY, DuckDB
            // dialect beyond sqlparser): try the constant-emitter path. It
            // self-validates — a dynamic query references the row table,
            // which DuckDB does not know, so evaluation fails and the
            // original clean error surfaces unchanged. Bind errors stay hard.
            Err(e @ (PrepareError::Unsupported(_) | PrepareError::Parse(_))) => {
                match eval_static_only(py, &sql, &static_tables) {
                    Ok((rows, fields)) => {
                        let output_model = match output_model {
                            Some(m) => m,
                            None => model_from_fields(py, fields)?,
                        };
                        return Ok(DuckDBInferFn {
                            engine: Engine::Constant { rows },
                            row_table,
                            output_model,
                        });
                    }
                    Err(_) => return Err(build_err(e.to_string())),
                }
            }
            Err(e) => return Err(build_err(e.to_string())),
        };

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

        // SPECIALIZER_FORCE_INTERP pins the interpreter — the bench control
        // and a debugging escape hatch.
        let force_interp = std::env::var_os("SPECIALIZER_FORCE_INTERP").is_some();
        let fun = match (force_interp, data) {
            (true, data) => Backend::Interp(
                compile(&prepared.program, data).map_err(|e| build_err(e.to_string()))?,
            ),
            (false, data) => match cranelift::compile(&prepared.program, data) {
                Ok(f) => Backend::Cranelift(f),
                // The failed attempt consumed the static data; rematerialize
                // on this cold path and fall back to the interpreter.
                Err(_) => {
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
                    Backend::Interp(
                        compile(&prepared.program, data).map_err(|e| build_err(e.to_string()))?,
                    )
                }
            },
        };
        let output_model = match output_model {
            // Supplied models are trusted as-is in v0 (no shape validation).
            Some(m) => m,
            None => synthesize_output_model(py, &prepared.program.out_cols)?,
        };
        // SPECIALIZER_GENERIC_BOUNDARY pins the pre-marshaller boundary —
        // the bench baseline, mirroring SPECIALIZER_FORCE_INTERP.
        let marsh = if std::env::var_os("SPECIALIZER_GENERIC_BOUNDARY").is_some() {
            None
        } else {
            Some(Marshaller::build(
                py,
                &in_cols,
                &prepared.program.out_cols,
                &output_model,
                &fun,
            )?)
        };
        Ok(DuckDBInferFn {
            engine: Engine::Compiled {
                fun,
                in_cols,
                out_cols: prepared.program.out_cols.clone(),
                marsh,
            },
            row_table,
            output_model,
        })
    }

    /// Which engine executes: "cranelift", "interpreter", or "constant".
    #[getter]
    fn backend(&self) -> &'static str {
        match &self.engine {
            Engine::Compiled { fun, .. } => fun.name(),
            Engine::Constant { .. } => "constant",
        }
    }

    /// How rows cross the Python boundary: "marshaller" (generated at
    /// prepare), "generic" (env-pinned baseline), or "constant".
    #[getter]
    fn boundary(&self) -> &'static str {
        match &self.engine {
            Engine::Compiled { marsh: Some(_), .. } => "marshaller",
            Engine::Compiled { marsh: None, .. } => "generic",
            Engine::Constant { .. } => "constant",
        }
    }

    #[pyo3(signature = (tables=None, **kwargs))]
    fn infer(
        &mut self,
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
        self.run_rows(py, &rows)
    }

    /// The direct hot entry: the row table's rows, no table-dict plumbing.
    /// `SpecializedTransform.infer_batch` calls this.
    fn infer_rows(&mut self, py: Python<'_>, rows: Vec<Py<PyAny>>) -> PyResult<Vec<Py<PyAny>>> {
        self.run_rows(py, &rows)
    }
}

impl DuckDBInferFn {
    fn run_rows(&mut self, py: Python<'_>, rows: &[Py<PyAny>]) -> PyResult<Vec<Py<PyAny>>> {
        let (fun, in_cols, out_cols, marsh) = match &mut self.engine {
            Engine::Compiled {
                fun,
                in_cols,
                out_cols,
                marsh,
            } => (&*fun, &*in_cols, &*out_cols, marsh),
            Engine::Constant { rows: fixed } => {
                let model = self.output_model.bind(py);
                let mut out = Vec::with_capacity(fixed.len());
                for r in fixed.iter() {
                    out.push(model.call_method1("model_validate", (&*r,))?.unbind());
                }
                return Ok(out);
            }
        };
        if let Some(m) = marsh {
            return m.call(py, fun, in_cols, rows, &self.row_table);
        }

        let n = rows.len();
        let mut cols: Vec<ColData> = in_cols
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
                    buf: String::new(),
                    spans: Vec::with_capacity(n),
                },
            })
            .collect();
        for row_obj in rows {
            let bound = row_obj.bind(py);
            for (c, col) in in_cols.iter().zip(&mut cols) {
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
                    s @ ColData::Str { .. } => {
                        if null {
                            s.push_str_cell(false, "");
                        } else {
                            s.push_str_cell(true, attr.extract()?);
                        }
                    }
                }
            }
        }

        let batch = Batch { rows: n, cols };
        let mut st = fun.new_state();
        fun.run(&batch, &mut st)
            .map_err(|t| PyErr::from(InterpError::Eval(t.0)))?;

        let model = self.output_model.bind(py);
        let mut out = Vec::with_capacity(st.emitted);
        for r in 0..st.emitted {
            let dict = PyDict::new(py);
            for (c, oc) in out_cols.iter().zip(&st.out) {
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
