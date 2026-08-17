//! The DuckDB-semantics engine: `DuckDBInferFn`.
//!
//! SQL + arrow row schema + pyarrow static tables in the constructor, then
//! `infer_rows()` maps dict-or-object rows through the specializer's
//! compiled program. `prepare()` does the SQL work (frontend -> lower ->
//! verify); this module is only the Python boundary — schema extraction on
//! the way in, map materialization for the join probes, dict rows on the
//! way out.

mod arrow;

use std::cell::RefCell;
use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::error::InterpError;
use crate::schema;
use crate::specializer::exec::cranelift::{self, CraneliftFn};
use crate::specializer::exec::interp::{compile_ext, InterpFn};
use crate::specializer::exec::{Batch, ColData, ExternImpl, KeyBits, OutCol, ScalarVal, StaticData};
use crate::specializer::exec::{RunState, Trap};
use crate::specializer::ir::{Col, ColTy, ExternSpec, StaticTy, Ty};
use crate::specializer::plan::StaticTable;
use crate::specializer::{prepare_opaque, StaticSpec, WideOut};

/// The declared type's spelling for boundary refusals — Arrow's, because
/// Arrow is what the caller wrote.
///
/// A refusal about a column the caller declared `pa.int32()` says `int32`,
/// not `INTEGER`, so the message quotes the declaration back instead of
/// making them translate it. Decided 2026-08-15. The DuckDB spellings live
/// on in `dialect/`, where they belong: that module emits SQL text.
pub(super) fn arrow_ty_name(t: Ty) -> &'static str {
    match t {
        Ty::I1 => "bool",
        Ty::I8 => "int8",
        Ty::I16 => "int16",
        Ty::I32 => "int32",
        Ty::I64 => "int64",
        Ty::F64 => "double",
        Ty::Str => "string",
    }
}

/// One row value into its input lane, strictly: the right Python type
/// category for the declared SQL type or a refusal naming the column — no
/// coercion ("1" is not 1, 1 is not 1.0, and bool is not an int value even
/// though Python subclasses it). The int lane takes anything INTEGER-LIKE
/// (`__index__`, which is what `operator.index()` means) so the fixed-width
/// numpy scalars that are Python's natural spelling of a narrow column
/// cross exactly — `np.int32(2)` into an INTEGER column, range still
/// checked. `np.bool_` has no `__index__` and stays out of the int lane,
/// exactly like a Python bool; the DOUBLE lane stays float-only (np.float64
/// IS a float, np.float32 refuses — the engine computes in f64). A narrow
/// width checks its range on the way IN, mirroring `narrow_check` out.
fn push_input_cell(col: &mut ColData, c: &Col, attr: &Bound<'_, PyAny>, null: bool) -> PyResult<()> {
    use pyo3::exceptions::PyOverflowError;
    use pyo3::types::{PyBool, PyFloat, PyInt};
    let type_err = |want: &str| {
        let got = attr
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "?".into());
        pyo3::exceptions::PyValueError::new_err(format!(
            "column '{}' expects {want} for its {} type, got {got}",
            c.name,
            arrow_ty_name(c.ty.ty)
        ))
    };
    // Fast path per arm: `downcast_exact` is one type-object pointer compare
    // (and already excludes bool from int — bool is never exactly PyInt);
    // only subclass values (np.float64, str subclasses) take the isinstance
    // fallback. Keeps the strict boundary at the old unchecked-extract cost.
    match col {
        ColData::I1 { valid, data } => {
            valid.push(!null);
            data.push(if null {
                false
            } else {
                match attr.cast_exact::<PyBool>() {
                    Ok(b) => b.is_true(),
                    // pyo3's bool extraction covers the rest: numpy's
                    // `bool_`, and it refuses ints and strings.
                    Err(_) => attr.extract().map_err(|_| type_err("bool"))?,
                }
            });
        }
        ColData::I64 { valid, data } => {
            valid.push(!null);
            data.push(if null {
                0
            } else {
                let range_err = |v: &dyn std::fmt::Display| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "column '{}' value {v} is outside its {} range",
                        c.name,
                        arrow_ty_name(c.ty.ty)
                    ))
                };
                let v: i64 = match attr.cast_exact::<PyInt>() {
                    Ok(i) => i.extract().map_err(|_| range_err(attr))?,
                    Err(_) => {
                        // A Python bool IS an int and HAS __index__, so it
                        // must be rejected before the integer-like
                        // extraction below.
                        if attr.is_instance_of::<PyBool>() {
                            return Err(type_err("int"));
                        }
                        // Wrong TYPE and out-of-i64 RANGE both fail here;
                        // the exception says which (OverflowError = a real
                        // integer that does not fit; anything else = not
                        // integer-like at all).
                        attr.extract::<i64>().map_err(|e| {
                            if e.is_instance_of::<PyOverflowError>(attr.py()) {
                                range_err(attr)
                            } else {
                                type_err("int")
                            }
                        })?
                    }
                };
                if let Some((lo, hi)) = c.ty.ty.int_range() {
                    if !(lo..=hi).contains(&v) {
                        return Err(range_err(&v));
                    }
                }
                v
            });
        }
        ColData::F64 { valid, data } => {
            valid.push(!null);
            data.push(if null {
                0.0
            } else {
                match attr.cast_exact::<PyFloat>() {
                    Ok(f) => f.value(),
                    Err(_) => {
                        if !attr.is_instance_of::<PyFloat>() {
                            return Err(type_err("float"));
                        }
                        attr.extract()?
                    }
                }
            });
        }
        s @ ColData::Str { .. } => {
            if null {
                s.push_str_cell(false, "");
            } else {
                match attr.cast_exact::<PyString>() {
                    Ok(st) => s.push_str_cell(true, st.to_str()?),
                    Err(_) => {
                        if !attr.is_instance_of::<PyString>() {
                            return Err(type_err("str"));
                        }
                        s.push_str_cell(true, attr.extract()?);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Append a static struct column's scalar leaves as lanes named by their
/// FULL ORDERED PATH (TASK-116), so `w.x.y.z.a` and `w.z.y.x.a` stay
/// distinct names and a lookup either walks the path exactly or misses. A
/// field name holding a '.' would make that encoding ambiguous, so its
/// subtree is skipped — the same rule the row path follows.
fn flatten_static(
    cols: &mut Vec<Col>,
    prefix: &str,
    fields: &[(String, schema::RowField)],
    parent_nullable: bool,
) {
    for (fname, rf) in fields {
        if fname.contains('.') {
            continue;
        }
        let path = format!("{prefix}.{fname}");
        match rf {
            schema::RowField::Scalar { ty, nullable } => cols.push(Col {
                name: path,
                ty: ColTy {
                    ty: *ty,
                    nullable: parent_nullable || *nullable,
                },
            }),
            schema::RowField::Struct { nullable, fields } => {
                flatten_static(cols, &path, fields, parent_nullable || *nullable)
            }
            schema::RowField::Opaque(_) => {}
        }
    }
}

/// A narrow out column's value must fit its declared width on EVERY
/// boundary — infer and infer_arrow answer identically or not at all
/// (fleet 2026-08-13: the row path served what the arrow path refused).
/// The matching runtime trap is m-8 phase 3.
fn narrow_check(ty: Ty, name: &str, v: i64) -> PyResult<()> {
    if let Some((lo, hi)) = ty.int_range() {
        if !(lo..=hi).contains(&v) {
            let ty_name = arrow_ty_name(ty);
            return Err(InterpError::Eval(format!(
                "column '{name}' value {v} is outside its {ty_name} range — \
                 the {ty_name} overflow trap lands with m-8 phase 3"
            ))
            .into());
        }
    }
    Ok(())
}

fn build_err(msg: impl Into<String>) -> PyErr {
    InterpError::Build(msg.into()).into()
}

/// One field of the output boundary: a plain scalar lane, or a wide UDF
/// field assembled from its whole-validity lane plus k component lanes.
/// Empty `names` is the DRAFT-22 unnamed boundary (the field is
/// `list | None`); non-empty assembles a STRUCT keyed by the declared
/// names (slice 5). Either way a NULL whole-validity is the NULL field —
/// distinct from a container of NULLs.
#[derive(Clone)]
pub(crate) enum EmitField {
    Scalar(usize),
    Wide {
        name: String,
        valid: usize,
        first: usize,
        width: usize,
        names: Vec<String>,
    },
}

impl EmitField {
    fn name<'a>(&'a self, out_cols: &'a [Col]) -> &'a str {
        match self {
            EmitField::Scalar(i) => &out_cols[*i].name,
            EmitField::Wide { name, .. } => name,
        }
    }
}

/// Collapse the engine's out lanes into boundary fields, in projection
/// order. `wide` entries are frontend-ordered and non-overlapping.
fn emit_plan(out_cols: &[Col], wide: &[WideOut]) -> Vec<EmitField> {
    let mut plan = Vec::new();
    let mut w = wide.iter().peekable();
    let mut i = 0usize;
    while i < out_cols.len() {
        if let Some(wo) = w.peek() {
            if wo.first as usize == i {
                plan.push(EmitField::Wide {
                    name: wo.name.clone(),
                    valid: i,
                    first: i + 1,
                    width: wo.width as usize,
                    names: wo.names.clone(),
                });
                i += 1 + wo.width as usize;
                w.next();
                continue;
            }
        }
        plan.push(EmitField::Scalar(i));
        i += 1;
    }
    plan
}

/// One engine lane's value at row `r`, as a Python object (None for NULL).
fn lane_py(py: Python<'_>, st: &RunState, lane: usize, r: usize) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    Ok(match &st.out[lane] {
        OutCol::I1(v) => {
            let (ok, x) = v[r];
            if ok {
                x.into_py_any(py)?
            } else {
                py.None()
            }
        }
        OutCol::I64(v) => {
            let (ok, x) = v[r];
            if ok {
                x.into_py_any(py)?
            } else {
                py.None()
            }
        }
        OutCol::F64(v) => {
            let (ok, x) = v[r];
            if ok {
                x.into_py_any(py)?
            } else {
                py.None()
            }
        }
        OutCol::Str(v) => {
            let (ok, s) = v[r];
            if ok {
                st.arena.get(s).into_py_any(py)?
            } else {
                py.None()
            }
        }
    })
}

/// A wide field's value at row `r`: None when the whole-call validity lane
/// says NULL, else the k-element list (unnamed extern) or the dict keyed by
/// the declared field names (named extern — DuckDB's struct value). Values
/// may be None either way. `tys` are the children's declared types: a
/// struct_pack child is an arbitrary expression, so it can carry a narrow
/// width, and the width contract holds for children exactly as for scalar
/// columns (see narrow_check) — on this boundary and the arrow one alike.
pub(crate) fn wide_py(
    py: Python<'_>,
    st: &RunState,
    valid: usize,
    first: usize,
    width: usize,
    names: &[String],
    tys: &[Ty],
    field: &str,
    r: usize,
) -> PyResult<Py<PyAny>> {
    let OutCol::I1(vlane) = &st.out[valid] else {
        return Err(build_err("internal: wide validity lane is not i1"));
    };
    let (ok, whole) = vlane[r];
    if !(ok && whole) {
        return Ok(py.None());
    }
    let items = (first..first + width)
        .enumerate()
        .map(|(j, l)| {
            if let (OutCol::I64(v), Some(_)) = (&st.out[l], tys[j].int_range()) {
                let (ok, x) = v[r];
                if ok {
                    let child = match names.get(j) {
                        Some(n) => format!("{field}.{n}"),
                        None => format!("{field}[{j}]"),
                    };
                    narrow_check(tys[j], &child, x)?;
                }
            }
            lane_py(py, st, l, r)
        })
        .collect::<PyResult<Vec<_>>>()?;
    if names.is_empty() {
        return Ok(pyo3::types::PyList::new(py, items)?.unbind().into_any());
    }
    let d = PyDict::new(py);
    for (n, v) in names.iter().zip(items) {
        d.set_item(n, v)?;
    }
    Ok(d.unbind().into_any())
}

/// Declared UDFs, parsed off the Python objects at construction: the
/// engine-facing signature plus the callable itself.
struct UdfDecl {
    spec: ExternSpec,
    obj: Py<PyAny>,
}

/// One arrow type as an engine type. The engine computes in exactly
/// `i64` / `f64` / string / bool, so the four spellings that map are the
/// whole vocabulary — a narrower arrow type refuses rather than widening
/// silently, which would make the declared schema a lie about what is served.
fn arrow_ty(name: &str, label: &str, t: &Bound<'_, PyAny>) -> PyResult<Ty> {
    let s = t.str()?.extract::<String>()?;
    Ok(match s.as_str() {
        "bool" => Ty::I1,
        "int64" => Ty::I64,
        "double" => Ty::F64,
        "string" => Ty::Str,
        other => {
            return Err(build_err(format!(
                "udf '{name}': {label} type '{other}' is not one of \
                 bool/int64/double/string"
            )))
        }
    })
}

/// `takes`: a `pa.Schema`, one field per argument, in call order.
fn parse_takes(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<(Vec<String>, Vec<Ty>)> {
    let bad = || build_err(format!("udf '{name}': `takes` must be a pyarrow Schema"));
    let names: Vec<String> = obj.getattr("names").map_err(|_| bad())?.extract().map_err(|_| bad())?;
    let types = obj.getattr("types").map_err(|_| bad())?;
    let mut tys = Vec::with_capacity(names.len());
    for t in types.try_iter().map_err(|_| bad())? {
        tys.push(arrow_ty(name, "takes", &t?)?);
    }
    if names.len() != tys.len() {
        return Err(bad());
    }
    Ok((names, tys))
}

/// `returns`: the SQL return TYPE, which is also what says how wide the call
/// is and whether its lanes are addressable.
///
/// * a scalar type — an ordinary scalar expression;
/// * `pa.struct([...])` — width-k with addressable field names (TASK-63),
///   struct-valued at EVERY width including 1;
/// * `pa.list_(t, k)` — width-k unnamed, the DRAFT-22 list boundary. Fixed
///   size because the width is part of the declaration; a variable-length
///   list would leave it unsaid.
fn parse_returns(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<(Vec<String>, Vec<Ty>)> {
    let s = obj.str()?.extract::<String>()?;
    if s.starts_with("struct<") {
        let n: usize = obj.getattr("num_fields")?.extract()?;
        if n == 0 {
            return Err(build_err(format!(
                "udf '{name}': `returns` struct declares no fields"
            )));
        }
        let mut names = Vec::with_capacity(n);
        let mut tys = Vec::with_capacity(n);
        for i in 0..n {
            let f = obj.call_method1("field", (i,))?;
            names.push(f.getattr("name")?.extract::<String>()?);
            tys.push(arrow_ty(name, "returns", &f.getattr("type")?)?);
        }
        return Ok((names, tys));
    }
    if s.starts_with("fixed_size_list<") {
        let k: i64 = obj.getattr("list_size")?.extract()?;
        // k == 1 is the one shape where the lane COUNT and the arrow SHAPE
        // disagree: width-1 unnamed binds as a plain scalar expression here
        // and registers as a scalar on DuckDB, so a value declared as a list
        // would cross as its element.
        if k < 2 {
            return Err(build_err(format!(
                "udf '{name}': a width-1 list return is a scalar — declare the \
                 element type rather than pa.list_(t, {k})"
            )));
        }
        let ty = arrow_ty(name, "returns", &obj.getattr("value_type")?)?;
        return Ok((Vec::new(), vec![ty; k as usize]));
    }
    if s.starts_with("list<") {
        return Err(build_err(format!(
            "udf '{name}': `returns` list must declare its width — \
             pa.list_(pa.float64(), k), not pa.list_(pa.float64())"
        )));
    }
    Ok((Vec::new(), vec![arrow_ty(name, "returns", obj)?]))
}

fn parse_udfs(py: Python<'_>, udfs: Vec<Py<PyAny>>) -> PyResult<(Vec<UdfDecl>, Vec<TreeDecl>)> {
    let mut out: Vec<UdfDecl> = Vec::new();
    let mut trees: Vec<TreeDecl> = Vec::new();
    // One namespace: a call site resolves a name against both lists, so a
    // collision between them would bind the wrong implementation silently.
    let mut names: Vec<String> = Vec::new();
    for obj in udfs {
        let b = obj.bind(py);
        let name: String = b
            .getattr("name")
            .and_then(|v| v.extract())
            .map_err(|_| build_err("every udf must declare a string `name`"))?;
        if let Some(other) = names.iter().find(|o| o.eq_ignore_ascii_case(&name)) {
            return Err(build_err(format!(
                "duplicate udf name '{other}' and '{name}'"
            )));
        }
        // A third namespace, and the binder consults it FIRST: a UDF named
        // after a builtin would be shadowed here while DuckDB binds the UDF.
        // See `frontend::BUILTIN_NAMES` for why this refuses rather than
        // letting either side win.
        if super::specializer::frontend::is_builtin(&name) {
            return Err(build_err(format!(
                "udf '{name}' collides with the builtin function '{name}' — \
                 rename it. The builtin binds first here, while DuckDB binds \
                 the udf, so the two engines would answer differently."
            )));
        }
        names.push(name.clone());
        let (_take_names, take_tys) = parse_takes(&name, &b.getattr("takes").map_err(|_| {
            build_err(format!("udf '{name}': `takes` must be a pyarrow Schema"))
        })?)?;
        // A UDF exposing `tree_tables()` is scored by the native kernel, so
        // it never becomes an extern: no callable, no GIL on the row path.
        // Its features bind positionally after the implicit instance id.
        if b.hasattr("tree_tables")? {
            if take_tys.is_empty() {
                return Err(build_err(format!(
                    "udf '{name}': a tree transform scores at least one feature"
                )));
            }
            trees.push(parse_tree_udf(py, name, take_tys, &b)?);
            continue;
        }
        let (ret_names, rets) = parse_returns(&name, &b.getattr("returns").map_err(|_| {
            build_err(format!("udf '{name}': `returns` must be a pyarrow DataType"))
        })?)?;
        let mut params = Vec::with_capacity(take_tys.len() + 1);
        // The implicit leading instance id (DRAFT-22): objects with an
        // `instances` attribute are fitted transformers whose first SQL
        // argument is the nullable i64 id — never written in `takes`.
        if b.hasattr("instances")? {
            params.push(Ty::I64);
        }
        params.extend(take_tys);
        // Field binding is ASCII-case-insensitive (here and in DuckDB's
        // struct keys) — colliding names would bind silently wrong.
        for (i, a) in ret_names.iter().enumerate() {
            if ret_names[..i].iter().any(|b| b.eq_ignore_ascii_case(a)) {
                return Err(build_err(format!(
                    "udf '{name}': return_names collide case-insensitively ('{a}')"
                )));
            }
        }
        // TASK-101: DuckDB's own flag, its default. Absent means false =
        // pure = bind-foldable; a PRESENT non-bool refuses rather than
        // failing open into executing user code at build (DuckDB's own
        // create_function rejects non-bools too).
        let side_effects = match b.getattr("side_effects") {
            Err(_) => false,
            Ok(v) => v.extract::<bool>().map_err(|_| {
                build_err(format!("udf '{name}': `side_effects` must be a bool"))
            })?,
        };
        out.push(UdfDecl {
            spec: ExternSpec {
                name,
                params,
                rets,
                ret_names,
                side_effects,
            },
            obj: obj.clone_ref(py),
        });
    }
    Ok((out, trees))
}

/// The engine-side implementations: one boxed trampoline per declared UDF.
/// Each call attaches to Python, converts NULL-masked scalars to Python
/// values, calls the object's scalar protocol (`__call__` returning a
/// tuple or None), and converts back under the declared return types. A
/// raised exception or a shape-violating result is a named trap (the
/// shared `call_extern` enforces shape again — belt and braces).
fn make_externs(py: Python<'_>, decls: &[UdfDecl]) -> Vec<ExternImpl> {
    decls
        .iter()
        .map(|d| {
            let obj = d.obj.clone_ref(py);
            let name = d.spec.name.clone();
            let rets = d.spec.rets.clone();
            ExternImpl {
                name: d.spec.name.clone(),
                fun: Box::new(move |args| {
                    Python::attach(|py| -> Result<Option<Vec<Option<ScalarVal>>>, String> {
                        use pyo3::IntoPyObjectExt;
                        let mut py_args = Vec::with_capacity(args.len());
                        for a in args {
                            py_args.push(match a {
                                None => py.None(),
                                Some(ScalarVal::I1(x)) => {
                                    x.into_py_any(py).map_err(|e| e.to_string())?
                                }
                                Some(ScalarVal::I64(x)) => {
                                    x.into_py_any(py).map_err(|e| e.to_string())?
                                }
                                Some(ScalarVal::F64(x)) => {
                                    x.into_py_any(py).map_err(|e| e.to_string())?
                                }
                                Some(ScalarVal::Str(x)) => {
                                    x.into_py_any(py).map_err(|e| e.to_string())?
                                }
                            });
                        }
                        let tuple = pyo3::types::PyTuple::new(py, py_args)
                            .map_err(|e| e.to_string())?;
                        let res = obj
                            .bind(py)
                            .call1(tuple)
                            .map_err(|e| format!("udf '{name}' raised: {e}"))?;
                        if res.is_none() {
                            return Ok(None);
                        }
                        let items: Vec<Bound<'_, PyAny>> = res
                            .try_iter()
                            .and_then(|it| it.collect())
                            .map_err(|_| {
                                format!(
                                    "udf '{name}' returned a non-sequence; the scalar \
                                     protocol returns a tuple or None"
                                )
                            })?;
                        if items.len() != rets.len() {
                            return Err(format!(
                                "udf '{name}' returned {} values, declared {}",
                                items.len(),
                                rets.len()
                            ));
                        }
                        let mut vals = Vec::with_capacity(rets.len());
                        for (item, ty) in items.iter().zip(&rets) {
                            if item.is_none() {
                                vals.push(None);
                                continue;
                            }
                            let bad = |_| {
                                format!(
                                    "udf '{name}' returned a value that is not the \
                                     declared {}",
                                    ty.name()
                                )
                            };
                            vals.push(Some(match ty {
                                Ty::I1 => ScalarVal::I1(item.extract().map_err(bad)?),
                                Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => {
                                    ScalarVal::I64(item.extract().map_err(bad)?)
                                }
                                Ty::F64 => ScalarVal::F64(item.extract().map_err(bad)?),
                                Ty::Str => ScalarVal::Str(item.extract().map_err(bad)?),
                            }));
                        }
                        Ok(Some(vals))
                    })
                }),
            }
        })
        .collect()
}

/// One map static from a pyarrow Table, per its `StaticSpec` recipe: rows
/// with a NULL `=` key are dropped (a NULL never equi-matches) while a NULL
/// IS NOT DISTINCT FROM key is an ordinary (false, default) key pair, a
/// NULL in a value column is an error, and an int column joined against a
/// float expression converts here (the declared key type is the
/// expression's).
fn materialize_map(
    py: Python<'_>,
    table: &Py<PyAny>,
    spec: &StaticSpec,
    key_tys: &[Ty],
    val_tys: &[Ty],
) -> PyResult<StaticData> {
    let rows = table.bind(py).call_method0("to_pylist")?;
    // TASK-90 / m-8 phase 1: `to_pylist` hands decimal.Decimal for DECIMAL
    // columns (the ordinary fit path — sum(BIGINT) params are
    // decimal128(38,0)), and a bare f64 extract SILENTLY rounds them:
    // 9007199254740993 served as ...992.0, off by one, violating "no third
    // mode". Exact-or-refuse: a Decimal that round-trips f64 exactly keeps
    // the conversion; one that does not refuses by name. Exact serving
    // arrives with the m-8 Dec lanes.
    let decimal_cls = PyModule::import(py, "decimal")?.getattr("Decimal")?;
    let exact_f64 = |v: &pyo3::Bound<'_, PyAny>, name: &str| -> PyResult<f64> {
        let f: f64 = v.extract()?;
        if v.is_instance(&decimal_cls)? {
            let back = decimal_cls.call1((f,))?;
            if !back.eq(v)? {
                return Err(build_err(format!(
                    "unsupported: static table '{}' column '{name}' holds the \
                     DECIMAL value {v} that f64 cannot represent exactly — \
                     serving it would round silently. CAST the fit-time \
                     aggregate to DOUBLE or BIGINT",
                    spec.table
                )));
            }
        }
        Ok(f)
    };
    let mut entries = Vec::new();
    'row: for item in rows.try_iter()? {
        let row = item?;
        let row = row.cast::<PyDict>().map_err(|_| {
            build_err(format!(
                "static table '{}': to_pylist row is not a dict",
                spec.table
            ))
        })?;
        // TASK-116: a struct column's lanes are named by their full ordered
        // path, so a dotted name walks the row's nested dicts. A NULL struct
        // anywhere on the way down makes every lane below it NULL, which is
        // exactly the nullability the catalogue derived for those leaves.
        let get = |name: &str| -> PyResult<pyo3::Bound<'_, PyAny>> {
            let missing = || {
                build_err(format!(
                    "static table '{}' row is missing column '{name}'",
                    spec.table
                ))
            };
            let mut cur = row.get_item(name.split('.').next().expect("non-empty"))?
                .ok_or_else(missing)?;
            for seg in name.split('.').skip(1) {
                if cur.is_none() {
                    return Ok(cur);
                }
                cur = cur
                    .cast::<PyDict>()
                    .map_err(|_| missing())?
                    .get_item(seg)?
                    .ok_or_else(missing)?;
            }
            Ok(cur)
        };
        let mut keys = Vec::with_capacity(key_tys.len());
        let mut kt = key_tys.iter();
        for (name, &indf) in spec.key_cols.iter().zip(&spec.key_indf) {
            let v = get(name)?;
            let convert = |v: &pyo3::Bound<'_, PyAny>, ty: Ty| -> PyResult<KeyBits> {
                Ok(match ty {
                    Ty::I1 => KeyBits::I1(v.extract()?),
                    Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => KeyBits::I64(v.extract().map_err(|_| {
                        build_err(format!(
                            "unsupported: static table '{}' key column '{name}' value \
                             outside int64 range (uint64/decimal128 payloads)",
                            spec.table
                        ))
                    })?),
                    Ty::F64 => KeyBits::F64(exact_f64(v, name)?.to_bits()),
                    Ty::Str => KeyBits::Str(v.extract()?),
                })
            };
            if indf {
                // IS NOT DISTINCT FROM key: (validity, payload) pair; NULL
                // is an ordinary key value, stored as (false, type default)
                // — exactly the probe side's masked encoding.
                let _validity_ty = kt.next();
                let ty = *kt.next().expect("payload type follows validity");
                if v.is_none() {
                    keys.push(KeyBits::I1(false));
                    keys.push(match ty {
                        Ty::I1 => KeyBits::I1(false),
                        Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => KeyBits::I64(0),
                        Ty::F64 => KeyBits::F64(0f64.to_bits()),
                        Ty::Str => KeyBits::Str(String::new()),
                    });
                } else {
                    keys.push(KeyBits::I1(true));
                    keys.push(convert(&v, ty)?);
                }
            } else {
                let ty = *kt.next().expect("one type per plain key column");
                if v.is_none() {
                    continue 'row;
                }
                keys.push(convert(&v, ty)?);
            }
        }
        let mut vals = Vec::with_capacity(val_tys.len());
        let mut vt = val_tys.iter();
        for (name, &nullable) in spec.val_cols.iter().zip(&spec.val_nullable) {
            let v = get(name)?;
            let convert = |v: &pyo3::Bound<'_, PyAny>, ty: Ty| -> PyResult<ScalarVal> {
                Ok(match ty {
                    Ty::I1 => ScalarVal::I1(v.extract()?),
                    Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => ScalarVal::I64(v.extract().map_err(
                        |_| {
                        build_err(format!(
                            "unsupported: static table '{}' value column '{name}' value \
                             outside int64 range (uint64/decimal128 payloads)",
                            spec.table
                        ))
                    })?),
                    Ty::F64 => ScalarVal::F64(exact_f64(v, name)?),
                    Ty::Str => ScalarVal::Str(v.extract()?),
                })
            };
            if nullable {
                // (validity, payload) pair per the flattened map layout
                // (TASK-55): NULL -> (false, typed default).
                let _validity_ty = vt.next();
                let ty = *vt.next().expect("payload type follows validity");
                if v.is_none() {
                    vals.push(ScalarVal::I1(false));
                    vals.push(match ty {
                        Ty::I1 => ScalarVal::I1(false),
                        Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => ScalarVal::I64(0),
                        Ty::F64 => ScalarVal::F64(0.0),
                        Ty::Str => ScalarVal::Str(String::new()),
                    });
                } else {
                    vals.push(ScalarVal::I1(true));
                    vals.push(convert(&v, ty)?);
                }
            } else {
                let ty = *vt.next().expect("one type per non-nullable column");
                if v.is_none() {
                    // Declared non-nullable yet NULL in the data — the
                    // original guard stays as a safety net.
                    return Err(build_err(format!(
                        "static table '{}' has a NULL in value column '{name}' — declared \
                         non-nullable",
                        spec.table
                    )));
                }
                vals.push(convert(&v, ty)?);
            }
        }
        entries.push((keys, vals));
    }
    Ok(StaticData::Map(entries))
}

/// A declared tree transform: the two Arrow tables its `tree_tables()` handed
/// over, plus the width and grid the binder needs. Held as Python objects
/// because the statics are materialized twice (the cranelift attempt consumes
/// them, the interpreter fallback rebuilds).
struct TreeDecl {
    name: String,
    nodes: Py<PyAny>,
    headers: Py<PyAny>,
    takes: Vec<Ty>,
    grid: super::specializer::plan::CompareGrid,
}

/// A declared UDF that exposes `tree_tables()` is served by the native kernel
/// rather than an ecall: `(nodes, models, compare_grid)` in, a `TreeDecl` out.
///
/// The grid is REQUIRED of the protocol, deliberately not defaulted: it says
/// which floating-point grid the thresholds were fitted on, and the packer
/// that would get it wrong is exactly the one that never thought about it. A
/// default would be the same trap with an extra step. See `plan::CompareGrid`.
fn parse_tree_udf(
    py: Python<'_>,
    name: String,
    takes: Vec<Ty>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<TreeDecl> {
    let got = obj
        .call_method0("tree_tables")
        .map_err(|e| build_err(format!("udf '{name}': tree_tables() raised: {e}")))?;
    let (nodes, headers, grid): (Py<PyAny>, Py<PyAny>, String) =
        got.extract().map_err(|_| {
            build_err(format!(
                "udf '{name}': tree_tables() must return (nodes, models, compare_grid)"
            ))
        })?;
    let grid = match grid.as_str() {
        "float32" => super::specializer::plan::CompareGrid::F32,
        "float64" => super::specializer::plan::CompareGrid::F64,
        other => {
            return Err(build_err(format!(
                "udf '{name}': compare_grid '{other}' is not 'float32' or 'float64'"
            )))
        }
    };
    let _ = py;
    Ok(TreeDecl {
        name,
        nodes,
        headers,
        takes,
        grid,
    })
}

/// Every static the program declares, in program order: one entry per join
/// (from its `StaticSpec` recipe) followed by one per referenced model set.
///
/// Called twice — the cranelift attempt CONSUMES the static data, so the
/// interpreter fallback rebuilds it. One function so the two paths cannot
/// drift.
fn materialize_statics(
    py: Python<'_>,
    prepared: &crate::specializer::Prepared,
    static_tables: &HashMap<String, Py<PyAny>>,
    trees: &[TreeDecl],
) -> PyResult<Vec<StaticData>> {
    let n_join = prepared.statics.len();
    if prepared.program.statics.len() != n_join + prepared.models.len() {
        return Err(build_err("internal: static count disagrees with the program"));
    }
    let mut data = Vec::with_capacity(prepared.program.statics.len());
    // Program statics and StaticSpecs are both indexed by join id.
    for (spec, sty) in prepared.statics.iter().zip(&prepared.program.statics) {
        if spec.batch {
            // Stage-B self-join: built per call by the executor.
            data.push(StaticData::Map(Vec::new()));
            continue;
        }
        let (StaticTy::Map { keys, values } | StaticTy::MultiMap { keys, values }) = sty else {
            return Err(build_err("internal: a join static lowered to a non-map"));
        };
        let table = static_tables
            .get(&spec.table)
            .expect("spec names come from the catalog");
        data.push(materialize_map(py, table, spec, keys, values)?);
    }
    // Model statics are appended AFTER every join static — that ordering is
    // what keeps existing probes' `@N` from shifting, and this zip asserts it.
    for (name, sty) in prepared
        .models
        .iter()
        .zip(&prepared.program.statics[n_join..])
    {
        let StaticTy::Model { n_features } = sty else {
            return Err(build_err(
                "internal: model statics must follow every join static",
            ));
        };
        let decl = trees
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(name))
            .expect("model names come from the catalog");
        data.push(StaticData::Model(Box::new(arrow::ensemble(
            decl.nodes.bind(py),
            decl.headers.bind(py),
            *n_features,
            name,
        )?)));
    }
    Ok(data)
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
) -> PyResult<(Vec<Py<PyAny>>, Py<PyAny>)> {
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
    let mut rows = Vec::new();
    for r in arrow.call_method0("to_pylist")?.try_iter()? {
        rows.push(r?.unbind());
    }
    Ok((rows, schema_obj))
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
    /// Ingest path per lane: one segment for a plain column, the dotted
    /// segments for a struct leaf (None at any level -> NULL lane).
    in_names: Vec<Vec<Py<PyString>>>,
    /// One entry per OUTPUT FIELD (not lane): plan + interned name.
    plan: Vec<EmitField>,
    /// Declared out-column types, indexed like the engine's out lanes —
    /// the row path enforces narrow widths with these (see narrow_check).
    out_tys: Vec<Ty>,
    out_names: Vec<Py<PyString>>,
    cols: Vec<ColData>,
    state: RunState,
}

impl Marshaller {
    fn build(
        py: Python<'_>,
        in_cols: &[Col],
        out_cols: &[Col],
        plan: &[EmitField],
        fun: &Backend,
    ) -> PyResult<Marshaller> {
        Ok(Marshaller {
            in_names: in_cols
                .iter()
                .map(|c| {
                    c.name
                        .split('.')
                        .map(|seg| PyString::intern(py, seg).unbind())
                        .collect()
                })
                .collect(),
            plan: plan.to_vec(),
            out_tys: out_cols.iter().map(|c| c.ty.ty).collect(),
            out_names: plan
                .iter()
                .map(|f| PyString::intern(py, f.name(out_cols)).unbind())
                .collect(),
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
            for ((c, path), col) in in_cols.iter().zip(&self.in_names).zip(&mut self.cols) {
                let mut attr = match dict {
                    Some(d) => d.get_item(path[0].bind(py))?.ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Row for table '{row_table}' is missing attribute '{}'",
                            c.name
                        ))
                    })?,
                    None => bound.getattr(path[0].bind(py)).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Row for table '{row_table}' is missing attribute '{}': {e}",
                            c.name
                        ))
                    })?,
                };
                // Struct leaf lanes walk the rest of the path; a None at
                // any level makes every leaf under it NULL.
                for seg in &path[1..] {
                    if attr.is_none() {
                        break;
                    }
                    attr = match attr.cast::<PyDict>() {
                        Ok(d) => d.get_item(seg.bind(py))?.ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Row for table '{row_table}' is missing attribute '{}'",
                                c.name
                            ))
                        })?,
                        Err(_) => attr.getattr(seg.bind(py)).map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Row for table '{row_table}' is missing attribute '{}': {e}",
                                c.name
                            ))
                        })?,
                    };
                }
                let null = attr.is_none();
                if null && !c.ty.nullable {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "column '{}' is not nullable but a row has None",
                        c.name
                    )));
                }
                push_input_cell(col, c, &attr, null)?;
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

        let mut out = Vec::with_capacity(self.state.emitted);
        for r in 0..self.state.emitted {
            let d = PyDict::new(py);
            for (name, field) in self.out_names.iter().zip(&self.plan) {
                let k = name.bind(py);
                match field {
                    EmitField::Scalar(i) => match &self.state.out[*i] {
                        OutCol::I1(v) => {
                            let (ok, x) = v[r];
                            d.set_item(k, ok.then_some(x))?;
                        }
                        OutCol::I64(v) => {
                            let (ok, x) = v[r];
                            if ok {
                                narrow_check(
                                    self.out_tys[*i],
                                    &k.to_string_lossy(),
                                    x,
                                )?;
                            }
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
                    },
                    EmitField::Wide {
                        valid,
                        first,
                        width,
                        names,
                        ..
                    } => {
                        d.set_item(
                            k,
                            wide_py(
                                py,
                                &self.state,
                                *valid,
                                *first,
                                *width,
                                names,
                                &self.out_tys[*first..*first + *width],
                                &k.to_string_lossy(),
                                r,
                            )?,
                        )?;
                    }
                }
            }
            out.push(d.unbind().into_any());
        }
        Ok(out)
    }
}

enum Engine {
    Compiled {
        fun: Backend,
        in_cols: Vec<Col>,
        out_cols: Vec<Col>,
        /// Output FIELDS in projection order (wide UDF lanes collapsed).
        plan: Vec<EmitField>,
        /// `None` when `SPECIALIZER_GENERIC_BOUNDARY` pinned the generic
        /// boundary at construction (the bench baseline). RefCell so infer
        /// stays `&self`: a reentrant call (a row property calling infer on
        /// the same object mid-marshal) finds the cell borrowed and falls
        /// through to the per-call generic path instead of erroring —
        /// master behavior (adversarial-review finding, 2026-07-26). The
        /// pyclass is unsendable, so single-threaded RefCell suffices.
        marsh: Option<RefCell<Marshaller>>,
    },
    /// Fixed row dicts from a static-only query (already dict-shaped), plus
    /// the result's `pa.Schema` for `output_schema`.
    Constant {
        rows: Vec<Py<PyAny>>,
        schema: Py<PyAny>,
    },
}

#[pyclass(unsendable)]
pub struct DuckDBInferFn {
    engine: Engine,
    row_table: String,
    /// 0 = filter, 1 = map, 2 = many (the declared row-shape contract).
    shape_kind: u8,
}

#[pymethods]
impl DuckDBInferFn {
    #[new]
    #[pyo3(signature = (sql, row_tables, static_tables, udfs=None, shape=None))]
    fn new(
        py: Python<'_>,
        sql: String,
        row_tables: HashMap<String, Py<PyAny>>,
        static_tables: HashMap<String, Py<PyAny>>,
        udfs: Option<Vec<Py<PyAny>>>,
        shape: Option<String>,
    ) -> PyResult<Self> {
        let (udf_decls, tree_decls) = parse_udfs(py, udfs.unwrap_or_default())?;
        // The row-shape contract (TASK-58): "filter" (default) is today's
        // 0..1 rows out per row in; "map" statically PROVES exactly-one
        // (out[i] <-> in[i]) or refuses at build; "many" is reserved for
        // join multiplicity (stage B) and is the only shape under which
        // those constructs will ever build.
        let many = shape.as_deref() == Some("many");
        let shape_kind: u8 = match shape.as_deref() {
            None | Some("filter") => 0,
            Some("map") => 1,
            Some("many") => 2,
            Some(_) => 0, // rejected below
        };
        let strict_map = match shape.as_deref() {
            None | Some("filter") => false,
            Some("map") => true,
            Some("many") => false, // multiplicity: `many` below
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "shape must be 'map', 'filter', or 'many', got '{other}'"
                )))
            }
        };
        let (row_table, row_schema) = match row_tables.len() {
            1 => row_tables.into_iter().next().unwrap(),
            n => {
                return Err(build_err(format!(
                    "unsupported: the specializer takes exactly one row table, got {n}"
                )))
            }
        };
        // Out-of-vocabulary row-column types reject only when REFERENCED
        // (the binder knows them as opaque, star expansion included) — an
        // unreferenced timestamp field must not block a scalar query.
        // Struct columns flatten to scalar leaf LANES appended after every
        // plain column (TASK-56); the dotted lane name doubles as the
        // ingest path (segments are field names without dots, so '.' is
        // unambiguous).
        let mut in_cols = Vec::new();
        let mut opaque: Vec<(usize, String)> = Vec::new();
        let mut struct_defs: Vec<(usize, String, bool, Vec<(String, schema::RowField)>)> =
            Vec::new();
        for (pos, (name, rf)) in schema::arrow_row_schema(py, &row_table, &row_schema)?
            .into_iter()
            .enumerate()
        {
            match rf {
                schema::RowField::Scalar { ty, nullable } => in_cols.push(Col {
                    name,
                    ty: ColTy { ty, nullable },
                }),
                schema::RowField::Struct { nullable, fields } => {
                    struct_defs.push((pos, name, nullable, fields))
                }
                schema::RowField::Opaque(_) => opaque.push((pos, name)),
            }
        }
        fn build_fields(
            in_cols: &mut Vec<Col>,
            prefix: &str,
            fields: &[(String, schema::RowField)],
            parent_nullable: bool,
        ) -> Vec<crate::specializer::plan::StructField> {
            use crate::specializer::plan::{StructField, StructNode};
            fields
                .iter()
                .map(|(fname, rf)| {
                    let node = if fname.contains('.') {
                        StructNode::Opaque // would break the path encoding
                    } else {
                        match rf {
                            schema::RowField::Struct { nullable, fields } => {
                                StructNode::Nested(build_fields(
                                    in_cols,
                                    &format!("{prefix}.{fname}"),
                                    fields,
                                    parent_nullable || *nullable,
                                ))
                            }
                            schema::RowField::Scalar { ty, nullable } => {
                                in_cols.push(Col {
                                    name: format!("{prefix}.{fname}"),
                                    ty: ColTy {
                                        ty: *ty,
                                        nullable: parent_nullable || *nullable,
                                    },
                                });
                                StructNode::Leaf((in_cols.len() - 1) as u32)
                            }
                            schema::RowField::Opaque(_) => StructNode::Opaque,
                        }
                    };
                    StructField {
                        name: fname.clone(),
                        node,
                    }
                })
                .collect()
        }
        let structs: Vec<crate::specializer::plan::StructCol> = struct_defs
            .into_iter()
            .map(|(pos, name, nullable, fields)| {
                let fields = build_fields(&mut in_cols, &name, &fields, nullable);
                crate::specializer::plan::StructCol { pos, name, fields }
            })
            .collect();

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
            // TASK-96: the SAME parser the row path uses, so a static column
            // types at its declared arrow width instead of collapsing every
            // integer to int64. A non-scalar column is still omitted rather
            // than rejected — unreferenced ones cost nothing.
            let mut cols = Vec::new();
            let mut opaque = Vec::new();
            for (cname, rf) in schema::arrow_static_schema(py, name, &schema_obj)? {
                match rf {
                    schema::RowField::Scalar { ty, nullable } => cols.push(Col {
                        name: cname,
                        ty: ColTy { ty, nullable },
                    }),
                    // Kept, not dropped — see StaticTable::opaque.
                    schema::RowField::Opaque(aty) => opaque.push((cname, aty)),
                    // TASK-116: a struct's scalar leaves ARE the lane set a
                    // static table already stores, so flatten them under
                    // their FULL ORDERED PATH ('w.mean', 'w.x.y.z.a') the
                    // way the row path does. The struct NAME stays opaque —
                    // `s.w` as a whole value, and `s.*`, are still unserved.
                    schema::RowField::Struct { nullable, fields } => {
                        flatten_static(&mut cols, &cname, &fields, nullable);
                        opaque.push((cname, "struct".to_string()));
                    }
                }
            }
            catalog.push(StaticTable {
                name: name.clone(),
                cols,
                opaque,
            });
        }

        use super::specializer::PrepareError;
        let extern_specs: Vec<ExternSpec> = udf_decls.iter().map(|d| d.spec.clone()).collect();
        let model_catalog: Vec<super::specializer::plan::ModelTable> = tree_decls
            .iter()
            .map(|t| super::specializer::plan::ModelTable {
                name: t.name.clone(),
                takes: t.takes.clone(),
                grid: t.grid,
            })
            .collect();
        // TASK-101: the same closures the runtime uses, handed to the
        // binder so a pure udf can constant-fold at build.
        let bind_impls = make_externs(py, &udf_decls);
        let prepared = match prepare_opaque(
            &sql,
            &row_table,
            &in_cols,
            &opaque,
            &structs,
            &catalog,
            many,
            &extern_specs,
            &model_catalog,
            &bind_impls,
        ) {
            Ok(p) => p,
            // With declared UDFs the constant-emitter fallback is off: DuckDB
            // cannot evaluate the udf calls, so surface the prepare error.
            Err(e) if !udf_decls.is_empty() || !tree_decls.is_empty() => {
                return Err(build_err(e.to_string()))
            }
            // Unsupported/unparseable SQL might still be a static-tables-only
            // query (static driving table, aggregation, ORDER BY, DuckDB
            // dialect beyond sqlparser): try the constant-emitter path. It
            // self-validates — a dynamic query references the row table,
            // which DuckDB does not know, so evaluation fails and the
            // original clean error surfaces unchanged. Bind errors stay hard.
            Err(e @ (PrepareError::Unsupported(_) | PrepareError::Parse(_))) => {
                match eval_static_only(py, &sql, &static_tables) {
                    Ok((rows, schema)) => {
                        if strict_map {
                            // Fixed rows regardless of input — the exact
                            // opposite of out[i] <-> in[i].
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "shape='map': a static-tables-only query emits fixed \
                                 rows unrelated to the input rows",
                            ));
                        }
                        return Ok(DuckDBInferFn {
                            engine: Engine::Constant { rows, schema },
                            row_table,
                            shape_kind,
                        });
                    }
                    Err(_) => return Err(build_err(e.to_string())),
                }
            }
            Err(e) => return Err(build_err(e.to_string())),
        };
        if strict_map {
            if let Some(blocker) = &prepared.one_row_blocker {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "shape='map': {blocker}"
                )));
            }
        }

        let data = materialize_statics(py, &prepared, &static_tables, &tree_decls)?;

        // SPECIALIZER_FORCE_INTERP pins the interpreter — the bench control
        // and a debugging escape hatch.
        let force_interp = std::env::var_os("SPECIALIZER_FORCE_INTERP").is_some();
        let fun = match (force_interp, data) {
            (true, data) => Backend::Interp(
                compile_ext(&prepared.program, data, make_externs(py, &udf_decls))
                    .map_err(|e| build_err(e.to_string()))?,
            ),
            (false, data) => {
                match cranelift::compile_ext(&prepared.program, data, make_externs(py, &udf_decls))
                {
                    Ok(f) => Backend::Cranelift(f),
                    // The failed attempt consumed the static data and the
                    // externs; rebuild both on this cold path and fall back
                    // to the interpreter.
                    Err(_) => Backend::Interp(
                        compile_ext(
                            &prepared.program,
                            materialize_statics(py, &prepared, &static_tables, &tree_decls)?,
                            make_externs(py, &udf_decls),
                        )
                        .map_err(|e| build_err(e.to_string()))?,
                    ),
                }
            }
        };
        let plan = emit_plan(&prepared.program.out_cols, &prepared.wide_outputs);
        // SPECIALIZER_GENERIC_BOUNDARY pins the pre-marshaller boundary —
        // the bench baseline, mirroring SPECIALIZER_FORCE_INTERP.
        let marsh = if std::env::var_os("SPECIALIZER_GENERIC_BOUNDARY").is_some() {
            None
        } else {
            Some(RefCell::new(Marshaller::build(
                py,
                &in_cols,
                &prepared.program.out_cols,
                &plan,
                &fun,
            )?))
        };
        Ok(DuckDBInferFn {
            engine: Engine::Compiled {
                fun,
                in_cols,
                out_cols: prepared.program.out_cols.clone(),
                plan,
                marsh,
            },
            row_table,
            shape_kind,
        })
    }

    /// The declared row-shape contract: "map", "filter", or "many".
    #[getter]
    fn shape(&self) -> &'static str {
        match self.shape_kind {
            1 => "map",
            2 => "many",
            _ => "filter",
        }
    }

    /// The output contract as a `pa.Schema`: field names, arrow types and
    /// order, exactly what `infer_arrow`'s output table carries (and the
    /// keys of every `infer_rows` dict).
    #[getter]
    fn output_schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.engine {
            Engine::Compiled { out_cols, plan, .. } => arrow::output_schema(py, out_cols, plan),
            Engine::Constant { schema, .. } => Ok(schema.clone_ref(py)),
        }
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

    /// The row path: dict-or-object rows in, dict rows out. A
    /// static-tables-only build emits fixed rows and cannot read input, so
    /// it REFUSES anything but `infer_rows([])` (TASK-110) rather than
    /// dropping what it was handed.
    fn infer_rows(&self, py: Python<'_>, rows: Vec<Py<PyAny>>) -> PyResult<Vec<Py<PyAny>>> {
        self.run_rows(py, &rows)
    }

    /// The columnar boundary (TASK-60): a single-chunk pa.Table or
    /// RecordBatch in, a pa.Table out — zero per-value Python objects on
    /// either side. Columns match the row model by NAME with strict
    /// dtypes (int64 / double / string / bool; cast first otherwise).
    /// Values are byte-identical to infer(); under shape='map' the output
    /// aligns positionally with the input.
    ///
    /// Refuses when the caller supplied an `output_model`: this path never
    /// builds Python rows, so it has nothing to run `model_validate` on, and
    /// silently skipping it made three documented entry points to one
    /// function give two different answers (TASK-71).
    fn infer_arrow(&self, py: Python<'_>, batch: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let (fun, in_cols, out_cols, plan) = match &self.engine {
            Engine::Compiled {
                fun,
                in_cols,
                out_cols,
                plan,
                ..
            } => (fun, in_cols, out_cols, plan),
            Engine::Constant { .. } => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "infer_arrow: a static-tables-only query emits fixed rows — \
                     use infer_rows([])",
                ))
            }
        };
        let input = arrow::ingest(py, &batch, in_cols)?;
        let mut st = fun.new_state();
        fun.run(&input, &mut st)
            .map_err(|t| PyErr::from(InterpError::Eval(t.0)))?;
        arrow::emit(py, out_cols, plan, &st)
    }
}

impl DuckDBInferFn {
    fn run_rows(&self, py: Python<'_>, rows: &[Py<PyAny>]) -> PyResult<Vec<Py<PyAny>>> {
        let (fun, in_cols, out_cols, plan, marsh) = match &self.engine {
            Engine::Compiled {
                fun,
                in_cols,
                out_cols,
                plan,
                marsh,
            } => (fun, in_cols, out_cols, plan, marsh),
            Engine::Constant { rows: fixed, .. } => {
                // TASK-110. This build reads only static tables, so it cannot
                // see input rows at all — and silently dropping them was the
                // one mistake at this boundary that did not refuse by name.
                // It hides a real caller bug: N request rows through a
                // function that structurally cannot read them returns 1 fixed
                // row, and the caller's positional assumption breaks
                // somewhere downstream instead of here.
                if !rows.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "this query reads only static tables, so it emits {} \
                         fixed row(s) and cannot see the {} row(s) given — \
                         call infer_rows([])",
                        fixed.len(),
                        rows.len(),
                    )));
                }
                let mut out = Vec::with_capacity(fixed.len());
                for r in fixed.iter() {
                    // A fresh copy per call: callers may mutate.
                    let d = r.bind(py).cast::<PyDict>().map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err(
                            "internal: constant rows are dicts",
                        )
                    })?;
                    out.push(d.copy()?.unbind().into_any());
                }
                return Ok(out);
            }
        };
        if let Some(cell) = marsh {
            // A reentrant call (row property re-entering infer mid-marshal)
            // finds the cell borrowed and takes the generic path below.
            if let Ok(mut m) = cell.try_borrow_mut() {
                return m.call(py, fun, in_cols, rows, &self.row_table);
            }
        }

        let n = rows.len();
        let mut cols: Vec<ColData> = in_cols
            .iter()
            .map(|c| match c.ty.ty {
                Ty::I1 => ColData::I1 {
                    valid: Vec::with_capacity(n),
                    data: Vec::with_capacity(n),
                },
                Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => ColData::I64 {
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
            // Dict rows are part of the API surface; the baseline path must
            // accept the same inputs as the marshaller, differing only in
            // cost (adversarial-review finding, 2026-07-26).
            let dict = bound.cast::<PyDict>().ok();
            for (c, col) in in_cols.iter().zip(&mut cols) {
                let mut segs = c.name.split('.');
                let first = segs.next().expect("split yields at least one");
                let mut attr = match dict {
                    Some(d) => d.get_item(first)?.ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Row for table '{}' is missing attribute '{}'",
                            self.row_table, c.name
                        ))
                    })?,
                    None => bound.getattr(first).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Row for table '{}' is missing attribute '{}': {e}",
                            self.row_table, c.name
                        ))
                    })?,
                };
                // Struct leaf lanes: walk the dotted path (None -> NULL).
                for seg in segs {
                    if attr.is_none() {
                        break;
                    }
                    attr = match attr.cast::<PyDict>() {
                        Ok(d) => d.get_item(seg)?.ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Row for table '{}' is missing attribute '{}'",
                                self.row_table, c.name
                            ))
                        })?,
                        Err(_) => attr.getattr(seg).map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Row for table '{}' is missing attribute '{}': {e}",
                                self.row_table, c.name
                            ))
                        })?,
                    };
                }
                let null = attr.is_none();
                if null && !c.ty.nullable {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "column '{}' is not nullable but a row has None",
                        c.name
                    )));
                }
                push_input_cell(col, c, &attr, null)?;
            }
        }

        let batch = Batch { rows: n, cols };
        let mut st = fun.new_state();
        fun.run(&batch, &mut st)
            .map_err(|t| PyErr::from(InterpError::Eval(t.0)))?;

        let mut out = Vec::with_capacity(st.emitted);
        for r in 0..st.emitted {
            let dict = PyDict::new(py);
            for field in plan {
                let name = field.name(out_cols);
                match field {
                    EmitField::Scalar(i) => match &st.out[*i] {
                        OutCol::I1(v) => {
                            let (ok, x) = v[r];
                            dict.set_item(name, ok.then_some(x))?;
                        }
                        OutCol::I64(v) => {
                            let (ok, x) = v[r];
                            if ok {
                                narrow_check(out_cols[*i].ty.ty, name, x)?;
                            }
                            dict.set_item(name, ok.then_some(x))?;
                        }
                        OutCol::F64(v) => {
                            let (ok, x) = v[r];
                            dict.set_item(name, ok.then_some(x))?;
                        }
                        OutCol::Str(v) => {
                            let (ok, s) = v[r];
                            dict.set_item(name, ok.then(|| st.arena.get(s)))?;
                        }
                    },
                    EmitField::Wide {
                        valid,
                        first,
                        width,
                        names,
                        ..
                    } => {
                        let tys: Vec<Ty> = out_cols[*first..*first + *width]
                            .iter()
                            .map(|c| c.ty.ty)
                            .collect();
                        dict.set_item(
                            name,
                            wide_py(py, &st, *valid, *first, *width, names, &tys, name, r)?,
                        )?;
                    }
                }
            }
            out.push(dict.unbind().into_any());
        }
        Ok(out)
    }
}
