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
use crate::specializer::exec::{
    self, Batch, ColData, ExternImpl, KeyBits, OutCol, ScalarVal, StaticData,
};
use crate::specializer::exec::{RunState, Trap};
use crate::specializer::ir::{Col, ColTy, ExternSpec, StaticTy, Ty};
use crate::specializer::plan::{self, StaticTable};
use crate::specializer::{prepare_opaque, StaticSpec, WideOut};

/// The declared type's spelling for boundary refusals — Arrow's, because
/// Arrow is what the caller wrote.
///
/// A refusal about a column the caller declared `pa.int32()` says `int32`,
/// not `INTEGER`, so the message quotes the declaration back instead of
/// making them translate it. Decided 2026-08-15. The DuckDB spellings live
/// on in `dialect/`, where they belong: that module emits SQL text.
pub(super) fn arrow_ty_name(t: Ty) -> std::borrow::Cow<'static, str> {
    use std::borrow::Cow;
    match t {
        Ty::I1 => Cow::Borrowed("bool"),
        Ty::I8 => Cow::Borrowed("int8"),
        Ty::I16 => Cow::Borrowed("int16"),
        Ty::I32 => Cow::Borrowed("int32"),
        Ty::I64 => Cow::Borrowed("int64"),
        Ty::F64 => Cow::Borrowed("double"),
        Ty::Str => Cow::Borrowed("string"),
        // pyarrow's own `str()` spelling, space and all, so a refusal
        // quotes back exactly what the caller can read off the schema.
        Ty::Dec(p, s) => Cow::Owned(format!("decimal128({p}, {s})")),
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
///
/// `name` and `ty` are exactly what this reads off the lane — the display
/// name for the two error messages, the declared type for `arrow_ty_name`
/// and `int_range`. Taken apart rather than as a `&Col` because this runs
/// ONCE PER CELL PER ROW on the engine's headline path against a ~200 ns/row
/// floor: `name: &str` borrows out of the lane, `ColTy` is `Copy`, and
/// rebuilding a `Col` per cell would allocate a `String` per cell.
fn push_input_cell(
    col: &mut ColData,
    name: &str,
    ty: ColTy,
    attr: &Bound<'_, PyAny>,
    null: bool,
) -> PyResult<()> {
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
            name,
            arrow_ty_name(ty.ty)
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
                        name,
                        arrow_ty_name(ty.ty)
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
                if let Some((lo, hi)) = ty.ty.int_range() {
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

/// An empty `ColData` for one input lane, optionally with capacity. Both row
/// boundaries build every lane through it and `arrow::ingest` builds its
/// string lanes here, so the mapping it fixes is the one every ingest path
/// obeys — and a PRESENCE lane is always a non-nullable `I1`, which is what
/// lets `ColData::push_present` assume an `I1`.
///
/// Not a constructor on `ColData`: that would put `exec` on `plan`, and the
/// import graph runs the other way (`plan` names `ir` and nothing else,
/// `exec` names `ir` and nothing else). The chooser lives at the BOUNDARY
/// instead, which is where every lane already is.
pub(crate) fn col_for_lane(lane: &plan::InputLane, cap: usize) -> ColData {
    let ty = match lane.kind {
        // Always non-nullable I1 — that is the whole payload of the kind.
        plan::LaneKind::Present => Ty::I1,
        plan::LaneKind::Value(ct) => ct.ty,
    };
    match ty {
        Ty::I1 => ColData::I1 {
            valid: Vec::with_capacity(cap),
            data: Vec::with_capacity(cap),
        },
        Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => ColData::I64 {
            valid: Vec::with_capacity(cap),
            data: Vec::with_capacity(cap),
        },
        Ty::F64 => ColData::F64 {
            valid: Vec::with_capacity(cap),
            data: Vec::with_capacity(cap),
        },
        Ty::Str => ColData::Str {
            valid: Vec::with_capacity(cap),
            buf: String::new(),
            spans: Vec::with_capacity(cap),
        },
        // A decimal ROW column is opaque (schema.rs, Policy::Row), so no
        // input lane is ever a Dec.
        Ty::Dec(..) => unreachable!("a decimal row column is opaque"),
    }
}

/// Append a static struct column's scalar leaves as input lanes and return
/// the field TREE that resolution walks. Each lane's `name` is its dotted
/// path, carried for display only — nothing resolves by building or
/// splitting that string. A field whose own name contains a '.' is skipped,
/// subtree and all, so it is unreachable here by any spelling.
fn flatten_static(
    cols: &mut Vec<Col>,
    prefix: &str,
    fields: &[(String, schema::RowField)],
    parent_nullable: bool,
) -> Vec<crate::specializer::plan::StructField> {
    use crate::specializer::plan::{StructField, StructNode};
    let mut tree = Vec::with_capacity(fields.len());
    for (fname, rf) in fields {
        // A retained choice, not something the encoding forces: lanes carry
        // a structured path, so a dotted segment is no longer ambiguous and
        // this skip could be lifted soundly. Until someone decides to,
        // dotted names stay opaque here. (The row path keeps such a field as
        // an `Opaque` node instead, so it refuses by name rather than going
        // missing.)
        if fname.contains('.') {
            continue;
        }
        let path = format!("{prefix}.{fname}");
        match rf {
            schema::RowField::Scalar { ty, nullable } => {
                cols.push(Col {
                    // display-only: resolution walks the tree
                    name: path,
                    ty: ColTy {
                        ty: *ty,
                        nullable: parent_nullable || *nullable,
                    },
                });
                tree.push(StructField {
                    name: fname.clone(),
                    node: StructNode::Leaf(cols.len() as u32 - 1),
                });
            }
            schema::RowField::Struct { nullable, fields } => {
                let n = flatten_static(cols, &path, fields, parent_nullable || *nullable);
                tree.push(StructField {
                    name: fname.clone(),
                    node: StructNode::Nested(n),
                });
            }
            // dropped, as before: an unservable leaf is invisible here
            schema::RowField::Opaque(_) => {}
        }
    }
    tree
}

/// A narrow out column's value must fit its declared width on EVERY
/// boundary — infer and infer_arrow answer identically or not at all
/// (fleet 2026-08-13: the row path served what the arrow path refused).
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

/// A scaled i128 back as `decimal.Decimal`, WITH THE DECLARED SCALE:
/// `Decimal('0.50')`, never `Decimal('0.5')`. Not cosmetic — the whole
/// differential harness compares by `repr`, and DuckDB's own `to_pylist`
/// preserves the trailing zeros. Built from the text spelling because the
/// string carries the exponent exactly; the row boundary is not the hot
/// lane (the arrow boundary writes the raw buffer instead).
pub(crate) fn dec_text(v: i128, scale: u8) -> String {
    if scale == 0 {
        return v.to_string();
    }
    let s = scale as usize;
    let neg = v < 0;
    let digits = v.unsigned_abs().to_string();
    let digits = if digits.len() <= s {
        format!("{}{digits}", "0".repeat(s + 1 - digits.len()))
    } else {
        digits
    };
    let (int, frac) = digits.split_at(digits.len() - s);
    format!("{}{int}.{frac}", if neg { "-" } else { "" })
}

/// A python `decimal.Decimal` as `(unscaled, scale)` at ITS OWN exponent,
/// exactly — no f64 anywhere on the path. `as_tuple()` rather than
/// `scaleb`/`int()`: the decimal context's 28-digit precision would round a
/// 38-digit payload, which is the whole class this task exists to serve.
fn decimal_parts(v: &Bound<'_, PyAny>, what: &str) -> PyResult<(i128, u8)> {
    let bad = || build_err(format!("{what} holds a DECIMAL this build cannot serve: {v}"));
    let t = v.call_method0("as_tuple").map_err(|_| bad())?;
    let sign: i32 = t.get_item(0)?.extract().map_err(|_| bad())?;
    let digits: Vec<u8> = t.get_item(1)?.extract().map_err(|_| bad())?;
    // NaN / Infinity spell their exponent as a string; arrow cannot carry
    // either, so this is a refusal rather than a special case.
    let exp: i32 = t.get_item(2)?.extract().map_err(|_| bad())?;
    let mut m: i128 = 0;
    for d in digits {
        m = m
            .checked_mul(10)
            .and_then(|x| x.checked_add(i128::from(d)))
            .ok_or_else(bad)?;
    }
    let (m, scale) = if exp >= 0 {
        (
            m.checked_mul(10i128.checked_pow(exp as u32).ok_or_else(bad)?)
                .ok_or_else(bad)?,
            0u8,
        )
    } else {
        let s = u8::try_from(-exp).map_err(|_| bad())?;
        if s > 38 {
            return Err(bad());
        }
        (m, s)
    };
    Ok((if sign == 1 { -m } else { m }, scale))
}

/// The same value's scaled integer AT `to`. `None` when it does not divide
/// down exactly — which for an integer key lane means the build row can
/// never equal any probe and drops.
fn rescale(m: i128, from: u8, to: u8) -> Option<i128> {
    if to >= from {
        m.checked_mul(10i128.checked_pow(u32::from(to - from))?)
    } else {
        let p = 10i128.checked_pow(u32::from(from - to))?;
        (m % p == 0).then_some(m / p)
    }
}

fn dec_py(py: Python<'_>, v: i128, ty: Ty) -> PyResult<Py<PyAny>> {
    let (_, s) = ty.dec().expect("a Dec lane carries a Dec type");
    Ok(PyModule::import(py, "decimal")?
        .getattr("Decimal")?
        .call1((dec_text(v, s),))?
        .unbind())
}

/// One field of the output boundary: a plain scalar lane, or a wide UDF
/// field assembled from its whole-validity lane plus k component lanes.
/// Empty `names` is the DRAFT-22 unnamed boundary (the field is
/// `list | None`); non-empty assembles a STRUCT keyed by the declared
/// names. Either way a NULL whole-validity is the NULL field —
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
fn lane_py(py: Python<'_>, st: &RunState, ty: Ty, lane: usize, r: usize) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    Ok(match &st.out[lane] {
        OutCol::Dec(v) => {
            let (ok, x) = v[r];
            if ok {
                dec_py(py, x, ty)?
            } else {
                py.None()
            }
        }
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
            lane_py(py, st, tys[j], l, r)
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
/// * `pa.struct([...])` — width-k with addressable field names,
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

/// Every declared UDF, split by how it will be SERVED: an ordinary extern
/// whose callable runs per row, or a tree transform the native kernel scores
/// without ever calling Python. The split is the object's own shape — a
/// `tree_tables()` method — not a flag the caller sets.
///
/// Both lists share ONE name space with each other and with the builtins,
/// because a call site resolves a name against all three.
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
        // DuckDB's own flag, and its default. Absent means false =
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
                                // A UDF over DECIMAL refuses at bind (m-8
                                // lattice phase 5).
                                Some(ScalarVal::Dec(..)) => {
                                    return Err(format!(
                                        "udf '{name}' was handed a DECIMAL argument, which \
                                         this build does not serve"
                                    ))
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
                                // A UDF over DECIMAL refuses at bind (m-8
                                // lattice phase 5), so no declaration
                                // reaches here carrying one.
                                Ty::Dec(..) => {
                                    return Err(format!(
                                        "udf '{name}' declares a DECIMAL return, which                                          this build does not serve"
                                    ))
                                }
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
/// NULL in a declared NON-NULLABLE value column is an error (a nullable one
/// rides as its own (false, default) pair), and an int column joined against a
/// float expression converts here (the declared key type is the
/// expression's).
fn materialize_map(py: Python<'_>, table: &Py<PyAny>, spec: &StaticSpec) -> PyResult<StaticData> {
    let rows = table.bind(py).call_method0("to_pylist")?;
    // `to_pylist` hands `decimal.Decimal` for a DECIMAL column (the ordinary
    // fit path — sum(BIGINT) params are decimal128(38,0)), and it becomes
    // the SCALED i128 exactly. No f64-exactness guard rides here on purpose:
    // the payload never touches f64, and a guard that refused every value
    // f64 could not hold refused ordinary ones — 2^63+1 is a plain
    // sum(BIGINT), referenced or not.
    let mut entries = Vec::new();
    'row: for item in rows.try_iter()? {
        let row = item?;
        let row = row.cast::<PyDict>().map_err(|_| {
            build_err(format!(
                "static table '{}': to_pylist row is not a dict",
                spec.table
            ))
        })?;
        // A struct leaf's SEGMENT PATH walks the row's nested dicts; a plain
        // column is one segment, dots and all — a name is never split. A
        // NULL struct anywhere on the way down makes every lane below it
        // NULL, which is exactly the nullability the catalogue derived for
        // those leaves.
        let get = |path: &[String]| -> PyResult<pyo3::Bound<'_, PyAny>> {
            let missing = || {
                build_err(format!(
                    "static table '{}' row is missing column '{}'",
                    spec.table,
                    path.join(".")
                ))
            };
            let mut cur = row
                .get_item(path.first().expect("non-empty").as_str())?
                .ok_or_else(missing)?;
            for seg in &path[1..] {
                if cur.is_none() {
                    return Ok(cur);
                }
                cur = cur
                    .cast::<PyDict>()
                    .map_err(|_| missing())?
                    .get_item(seg.as_str())?
                    .ok_or_else(missing)?;
            }
            Ok(cur)
        };
        let mut keys = Vec::with_capacity(spec.keys.len());
        for k in &spec.keys {
            let present = k.present;
            let name = k.path.join(".");
            let name = name.as_str();
            let v = get(&k.path)?;
            // `None` means "this build row can never equal any probe" — a
            // non-integral DECIMAL against an integer key lane. Dropping it
            // is the established, semantics-preserving move (a NULL `=` key
            // drops the same way, just below).
            let convert = |v: &pyo3::Bound<'_, PyAny>, ty: Ty| -> PyResult<Option<KeyBits>> {
                // A PRESENCE key reads the struct NODE, not a value:
                // reaching here at all means the node is non-NULL, and the
                // NULL case is handled by the plain / INDF arms below
                // exactly as it is for any other key.
                if present {
                    return Ok(Some(KeyBits::I1(true)));
                }
                let is_dec = |v: &pyo3::Bound<'_, PyAny>| -> PyResult<bool> {
                    Ok(v.get_type().name()?.to_string_lossy() == "Decimal")
                };
                Ok(Some(match ty {
                    Ty::I1 => KeyBits::I1(v.extract()?),
                    // A DECIMAL build key against an INTEGER probe: DuckDB
                    // casts the integer UP to the decimal, exactly
                    // (ImplicitCastBigint, cast_rules.cpp:96-107), so
                    // equality is decidable here — integral and in range,
                    // or the row cannot match.
                    Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 if is_dec(v)? => {
                        let (m, s) = decimal_parts(
                            v,
                            &format!("static table '{}' key column '{name}'", spec.table),
                        )?;
                        match rescale(m, s, 0).and_then(|x| i64::try_from(x).ok()) {
                            Some(i) => KeyBits::I64(i),
                            None => return Ok(None),
                        }
                    }
                    Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => KeyBits::I64(v.extract().map_err(|_| {
                        build_err(format!(
                            "unsupported: static table '{}' key column '{name}' value \
                             outside int64 range (uint64/decimal128 payloads)",
                            spec.table
                        ))
                    })?),
                    // A DECIMAL build key against a DOUBLE probe: only
                    // decimal->double is a legal implicit cast, so the
                    // DECIMAL side casts DOWN and the comparison is LOSSY.
                    // Reproduce the loss with DuckDB's own algorithm rather
                    // than hide it (cell D2).
                    Ty::F64 if is_dec(v)? => {
                        let (m, s) = decimal_parts(
                            v,
                            &format!("static table '{}' key column '{name}'", spec.table),
                        )?;
                        KeyBits::F64(
                            crate::specializer::exec::kernels::dec_to_f64(m, s).to_bits(),
                        )
                    }
                    Ty::F64 => KeyBits::F64(v.extract::<f64>()?.to_bits()),
                    Ty::Str => KeyBits::Str(v.extract()?),
                    // The key lane is the PROBE expression's type, and a
                    // probe expression is a ROW expression — decimal row
                    // columns are opaque, so nothing can produce one.
                    Ty::Dec(..) => unreachable!("a probe expression is never a decimal"),
                }))
            };
            let ty = k.map.ty;
            match k.map.cmp {
                // IS NOT DISTINCT FROM key: (validity, payload) pair; NULL
                // is an ordinary key value, stored as (false, type default)
                // — exactly the probe side's masked encoding.
                plan::KeyCmp::NotDistinct => {
                    if v.is_none() {
                        keys.extend(exec::null_key_slots(ty));
                    } else {
                        let Some(kb) = convert(&v, ty)? else {
                            continue 'row;
                        };
                        keys.push(KeyBits::I1(true));
                        keys.push(kb);
                    }
                }
                // A NULL `=` key never matches anything, so the row is not
                // in the map at all — the per-site half of the plain-key
                // rule, which the slot layout deliberately does not own.
                plan::KeyCmp::Eq => {
                    if v.is_none() {
                        continue 'row;
                    }
                    let Some(kb) = convert(&v, ty)? else {
                        continue 'row;
                    };
                    keys.push(kb);
                }
            }
        }
        let mut vals = Vec::with_capacity(spec.vals.len());
        for vc in &spec.vals {
            let name = vc.path.join(".");
            let name = name.as_str();
            let v = get(&vc.path)?;
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
                    Ty::F64 => ScalarVal::F64(v.extract()?),
                    Ty::Str => ScalarVal::Str(v.extract()?),
                    // The whole point: the scaled i128, exactly, no f64 on
                    // the path at all.
                    Ty::Dec(p, s) => {
                        let what = format!("static table '{}' value column '{name}'", spec.table);
                        let (m, vs) = decimal_parts(v, &what)?;
                        ScalarVal::Dec(
                            rescale(m, vs, s).ok_or_else(|| {
                                build_err(format!(
                                    "{what} holds {v}, which does not fit its declared \
                                     decimal128({p}, {s})"
                                ))
                            })?,
                            p,
                            s,
                        )
                    }
                })
            };
            let ty = vc.map.ty;
            if vc.map.nullable {
                // (validity, payload) pair per the flattened map layout:
                // NULL -> (false, typed default).
                if v.is_none() {
                    vals.extend(exec::null_val_slots(ty));
                } else {
                    vals.push(ScalarVal::I1(true));
                    vals.push(convert(&v, ty)?);
                }
            } else {
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
        // The recipe and the lowered type vector are two derivations of one
        // slot layout, and the materializer now takes its types off the
        // recipe alone — so this is where the two are compared.
        //
        // MEASURED, against the spec's claim that a disagreement would
        // otherwise be silent: it would not. `interp::prepare_statics`
        // type-checks EVERY entry's flat key and value vectors against the
        // declaration in release, for `Map` and `MultiMap` alike, and a
        // shortened build tuple fails there with "static data mismatch".
        // What this assert buys is WHERE and WHEN: it fires at the
        // recipe/declaration seam naming the two derivations, before any
        // row is read, instead of downstream naming one row's shape — and
        // it covers the zero-row table the entry loop walks vacuously.
        // Debug-only, so release behavior is untouched.
        debug_assert_eq!(
            spec.keys.iter().map(|k| k.map.slots().len()).sum::<usize>(),
            keys.len(),
            "StaticSpec and StaticTy disagree on key slot arity"
        );
        debug_assert_eq!(
            spec.vals.iter().map(|v| v.map.slots().len()).sum::<usize>(),
            values.len(),
            "StaticSpec and StaticTy disagree on value slot arity"
        );
        data.push(materialize_map(py, table, spec)?);
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

/// A frozen ORDER BY that leaves two rows tied: the query states no order
/// between them, so two builds of the same function could freeze different
/// sequences (goal.md exclusion: whole-relation-shapes).
const TIE_ORDER_REFUSAL: &str = "unsupported: tie-producing ORDER BY on a \
     static-tables-only query -- which of the tied rows comes first depends \
     on scan order, not the query";

/// The same refusal when the keys could not be measured at all: an unruled-out
/// tie is refused like a found one. `why` names what stopped the reading.
fn unmeasurable_order_refusal(why: &str) -> String {
    format!(
        "unsupported: {why} on a static-tables-only query -- a tie among its \
         rows could not be ruled out, and which of two tied rows comes first \
         depends on scan order, not the query"
    )
}

/// A clause that keeps only some of the rows. `clause` is the clause as the
/// statement spells it, so the message quotes the query back.
fn row_limit_refusal(clause: &str) -> String {
    format!(
        "unsupported: row limit ({clause}) on a static-tables-only query -- \
         which rows survive depends on scan order, not the query"
    )
}

/// A shape whose answer follows scan position: it picks one row out of a
/// group, or numbers rows, and the query says nothing about which row that
/// is. `shape` names it, so the refusal points at the clause to remove.
fn positional_refusal(shape: &str) -> String {
    format!(
        "unsupported: {shape} on a static-tables-only query -- what it picks \
         out of the rows depends on scan order, not the query"
    )
}

/// A statement DuckDB runs but will not hand back for inspection. Nothing
/// can be ruled out about a shape that cannot be read, so the query refuses
/// whatever it spells; `word` only names the clause that raised the question
/// when the tokens showed one.
fn unreadable_refusal(word: Option<&str>) -> String {
    let what = word.map_or_else(
        || "a statement".to_string(),
        |w| format!("{w} in a statement"),
    );
    format!(
        "unsupported: {what} DuckDB would not expose for \
         inspection -- what a static-tables-only query selects may be frozen \
         only when it is a function of the query, and this shape could not be \
         read"
    )
}

/// Several statements in one string. DuckDB exposes them all perfectly well
/// (measured), so the reading names the count, not the serialization: only
/// one of them is inspected and only the last one's rows come back — which
/// leaves every value rule asked of the wrong statement, whether or not any
/// clause word turned up.
fn multi_statement_refusal(word: Option<&str>) -> String {
    let what = word.map_or_else(
        || "a statement string".to_string(),
        |w| format!("{w} in a statement string"),
    );
    format!(
        "unsupported: {what} holding more than one \
         statement -- what a static-tables-only query selects may be frozen \
         only when it is a function of the query, and only one statement of \
         this string is read"
    )
}

/// A function whose value is not fixed by its arguments. DuckDB's own
/// catalogue says which: `duckdb_functions().stability` is VOLATILE for a
/// fresh draw per row (`random`, `gen_random_uuid`) and
/// CONSISTENT_WITHIN_QUERY for one reading of a clock or a session per query
/// (`now`, `current_date`). Either way the value belongs to the RUN, so
/// freezing one makes the answer a function of the build.
///
/// Two kinds the catalogue cannot answer for join them: a MACRO, whose
/// stability column is NULL, is read through its `macro_definition` instead,
/// and the names in `RUN_STATE_FUNCTIONS`, which the catalogue calls
/// CONSISTENT although their value comes from the machine or the moment.
fn nondeterministic_refusal(name: &str) -> String {
    format!(
        "unsupported: the non-deterministic function {name}() on a \
         static-tables-only query -- its value is drawn when the query runs, \
         not fixed by the query"
    )
}

/// An aggregate whose answer depends on the order its rows arrive in.
///
/// DuckDB classifies this itself and defaults to ORDER_DEPENDENT: an
/// aggregate is order-free only where the source calls
/// `SetOrderDependent(NOT_ORDER_DEPENDENT)`, which is `ORDER_FREE_AGGREGATES`
/// and nothing else. The default catches three kinds at once — the ones that
/// pick a row (`first`, `arg_max`, `mode`), the ones that build a sequence
/// (`list`, `string_agg`), and the ones that accumulate in floating point,
/// where association is not a law (`avg`, `stddev`).
///
/// An aggregate may carry its OWN `ORDER BY`, which DuckDB's optimizer uses
/// to make an order-dependent aggregate deterministic — but only when those
/// keys separate the rows WITHIN each group, and measuring that is a probe
/// this reading does not build. So the family refuses whole, and the message
/// says the inner ORDER BY was not read rather than leaving the user to
/// guess why it did not help.
fn order_sensitive_agg_refusal(name: &str) -> String {
    format!(
        "unsupported: order-sensitive aggregate {name} on a \
         static-tables-only query -- its answer follows scan order, and an \
         ORDER BY inside the aggregate is not read as a fix"
    )
}

/// What DuckDB's own parse of a statement says about the shapes a constant
/// fold may not freeze.
struct Shapes {
    /// The row-limiting clause, spelled the way the statement spells it.
    /// `None` when the statement has none, and when the one it has removes
    /// no row (`LIMIT ALL`, `OFFSET 0`).
    row_limit: Option<&'static str>,
    /// A shape whose frozen answer would not be a function of the query, as
    /// the whole refusal it earns.
    refusal: Option<String>,
}

/// The aggregate names DuckDB itself declares order-free for EVERY overload
/// it has (v1.5.5, `SetOrderDependent(AggregateOrderDependent::
/// NOT_ORDER_DEPENDENT)` in src/function/aggregate and
/// extension/core_functions/aggregate). Every other aggregate keeps the
/// AggregateFunction constructor's default, which is ORDER_DEPENDENT.
///
/// `sum` is absent from this list because the flag is not its whole name's:
/// DuckDB opts its BOOLEAN, integer, HUGEINT and DECIMAL overloads out and
/// leaves the DOUBLE one order-dependent, where association is not a law
/// (measured, `sum` over doubles arriving in a hash order answered six ways).
/// Which overload a call bound is not in DuckDB's parse, so `exact_sum` asks
/// DuckDB's binder for it instead and this list gains `sum` for that
/// statement when the answer is an exact overload.
///
/// `sum_no_overflow` is opted out at both its overloads and is still absent,
/// because no query can name it: it is in the catalogue but binding one is
/// `sum_no_overflow is for internal use only!` (measured). A name that
/// cannot be called cannot be over-refused.
const ORDER_FREE_AGGREGATES: &str = "'count','count_star','min','max',\
     'bool_and','bool_or','mad','median','quantile','quantile_cont',\
     'quantile_disc'";

/// The pure WINDOW functions, which DuckDB's catalogue also types
/// `aggregate` because its window machinery reuses the aggregate registry.
/// They are answered by the window readings instead — the row-position ones
/// by name, the rest by their frame — and letting the aggregate rule reach
/// them would refuse the `rank` family, whose value is a function of the key
/// and is the same under every setting (measured).
const WINDOW_ONLY_FUNCTIONS: &str = "'row_number','ntile','lead','lag',\
     'first_value','last_value','nth_value','rank','rank_dense','dense_rank',\
     'percent_rank','cume_dist'";

/// The clock spellings DuckDB's parser accepts as bare words rather than
/// calls, so they arrive in the parse as a column reference and the
/// catalogue lookup never sees them.
///
/// The list stops at the clock on purpose. A bare name is matched against
/// THIS list and not against the catalogue, because a static table may
/// legitimately have a column called `uuid`, `today` or `error` — DuckDB
/// binds the column, and matching the catalogue would refuse a query with
/// nothing wrong with it. Measured, no other non-CONSISTENT name binds bare
/// at all: `random`, `now` and the rest are a binder error without their
/// parentheses. That leaves a static column actually named `current_date`
/// as the one false refusal this costs, and two builds reading two clocks as
/// what it buys.
const CLOCK_KEYWORDS: &str =
    "'current_date','current_time','current_timestamp','localtime','localtimestamp'";

/// The functions whose value is a reading of the machine or the moment while
/// DuckDB's catalogue calls them CONSISTENT — the one place this reading
/// keeps names of its own, because DuckDB has no flag that answers the
/// question. `stability` says "constant within one query"; the rule here is
/// "a function of the query text and the statics", and these four are not.
///
/// The two clocks are DuckDB's own inconsistency, not a judgement call: its
/// binder maps the bare words `localtime` and `localtimestamp` onto
/// `current_localtime`/`current_localtimestamp`
/// (`bind_columnref_expression.cpp`), which ICU registers with no stability
/// at all and so inherits CONSISTENT (`extension/icu/icu-timezone.cpp`).
/// Measured, the value moves between two connections milliseconds apart, and
/// the bare spellings already refuse. `version` and `current_setting` freeze
/// the build's wheel and the build machine's settings — two machines, two
/// frozen answers for one query.
///
/// A list, so not a proof of exhaustion: a TABLE function that reads the
/// machine (`duckdb_settings()`) is the same class and is not covered, which
/// known-limitations.md names.
const RUN_STATE_FUNCTIONS: &str =
    "'current_localtime','current_localtimestamp','version','current_setting'";

/// The shapes of a statement, read off DuckDB's OWN parse of it.
///
/// A static-tables-only query is evaluated once at build and frozen, and what
/// a whole-relation construct selects may be frozen only when it is a
/// function of the query (goal.md exclusion: whole-relation-shapes). Measured
/// over a 60k-row static table fed through a tying `GROUP BY`, the same
/// statement under five DuckDB settings a build machine picks for itself
/// (default, `threads` 1/2/8, `preserve_insertion_order=false`) answered: a
/// `LIMIT` in a derived table four ways, in a CTE five, `DISTINCT ON` five,
/// `QUALIFY` over `row_number()` five, `row_number()` over tied keys five,
/// and `USING SAMPLE` differently on all twelve of twelve fresh connections.
/// None of those is a function of the query, so all of them refuse.
///
/// The reading is DuckDB's because this carve-out exists FOR DuckDB syntax:
/// `json_serialize_sql` is the parser that will run the query, so no clause
/// can hide behind dialect another parser has never seen. Its output is
/// minified and JSON-escaped, so the markers below can only come from real
/// structure — a string literal spelling one out arrives with its quotes
/// escaped and cannot forge one (measured).
///
/// When DuckDB will not serialize a statement it still runs — `PIVOT` is one
/// — the reading falls back to DuckDB's TOKENS, and that fallback refuses
/// rather than falling silent: a shape nobody could read is a shape nobody
/// ruled out.
///
/// Three of the readings are about a VALUE rather than a row's position, and
/// they answer the same question: a function whose value is a draw or a
/// clock, an aggregate whose value follows the arrival order, and a window
/// frame counted in ROWS rather than in key peers. Each is asked of DuckDB's
/// own catalogue or its own frame flavour, not of a list kept here — with
/// the two exceptions the catalogue cannot answer, which say so where they
/// are written (`RUN_STATE_FUNCTIONS`, and a macro read through its
/// definition).
///
/// Asked of the connection the statement has ALREADY run on, so the
/// catalogue is the one that bound it: `json_serialize_sql` only parses, and
/// a function DuckDB autoloads an extension for at BIND time is in no
/// catalogue before then.
fn read_shapes(py: Python<'_>, con: &Bound<'_, PyAny>, sql: &str) -> PyResult<Shapes> {
    /// One round trip: the parse status, and every marker the scan turns on,
    /// gathered from anywhere in the tree by JSON recursive descent.
    ///
    /// The two name columns pick the ALPHABETICALLY FIRST offender so the
    /// message a query earns is a function of the query. Calls are matched
    /// against the catalogue; a BARE name is matched only against the clock
    /// keywords and `rowid`, because a static column may legitimately be
    /// called `uuid` or `today` and DuckDB binds the column, not the
    /// function.
    ///
    /// A row limit that removes no row is not a row limit, and the walk
    /// distinguishes the two ways a limit node can fail to be one. DuckDB's
    /// LIMIT modifier always carries BOTH fields, so the side the query did
    /// not spell arrives as the JSON literal `null` — that one is absent, and
    /// absent removes no row. The side it did spell has to READ as a no-op:
    /// `LIMIT ALL` is a NULL-typed constant and `OFFSET 0` a zero. A limit
    /// that is any other node — an arithmetic expression, a CAST, a subquery
    /// — is one this reading cannot evaluate, so it counts as the real limit
    /// it is rather than defaulting to a no-op.
    const SHAPE_SQL: &str = "WITH a(j) AS (SELECT json_serialize_sql(?)), \
         f(n) AS (SELECT lower(unnest(coalesce( \
             json_extract_string(j, '$..function_name'), []))) FROM a), \
         cn(x) AS (SELECT unnest(coalesce(CAST(json_extract(j, \
             '$..column_names') AS JSON[]), [])) FROM a), \
         c(n) AS (SELECT lower(json_extract_string(x, '$[0]')) FROM cn \
             WHERE json_array_length(x) = 1), \
         t(n) AS (SELECT lower(json_extract_string(x, \
             '$[' || (json_array_length(x) - 1) || ']')) FROM cn), \
         lim(x) AS (SELECT unnest(coalesce(CAST(json_extract(j, '$..limit') \
             AS JSON[]), [])) FROM a), \
         off(x) AS (SELECT unnest(coalesce(CAST(json_extract(j, '$..offset') \
             AS JSON[]), [])) FROM a), \
         ag(n) AS (SELECT n FROM f WHERE n IN (SELECT lower(function_name) \
             FROM duckdb_functions() WHERE function_type = 'aggregate' \
                 AND lower(function_name) NOT IN (__ORDER_FREE__))) \
         SELECT json_extract(j, '$.error')::VARCHAR, \
         json_array_length(json_extract(j, '$.statements')), \
         json_extract(j, '$..type')::VARCHAR, \
         json_extract(j, '$..qualify')::VARCHAR, \
         json_extract(j, '$..sample')::VARCHAR, \
         json_extract(j, '$..distinct_on_targets')::VARCHAR, \
         json_extract(j, '$..ref_type')::VARCHAR, \
         json_extract(j, '$..start')::VARCHAR || \
             json_extract(j, '$..end')::VARCHAR, \
         (SELECT min(n) FROM ( \
             SELECT n FROM f WHERE n IN (SELECT lower(function_name) \
                 FROM duckdb_functions() \
                 WHERE stability IN ('VOLATILE', 'CONSISTENT_WITHIN_QUERY')) \
             UNION ALL SELECT n FROM f WHERE n IN (__RUN_STATE__) \
             UNION ALL SELECT f.n FROM f, duckdb_functions() d \
                 WHERE lower(d.function_name) = f.n \
                     AND d.macro_definition IS NOT NULL \
                     AND regexp_matches(lower(d.macro_definition), \
                         (SELECT '\\b(' || string_agg(DISTINCT v, '|') || ')\\b' \
                          FROM (SELECT lower(function_name) AS v \
                                    FROM duckdb_functions() \
                                    WHERE stability IN ('VOLATILE', \
                                        'CONSISTENT_WITHIN_QUERY') \
                                        AND lower(function_name) <> 'error' \
                                UNION SELECT unnest([__CLOCK__])))) \
             UNION ALL SELECT n FROM c WHERE n IN (__CLOCK__))), \
         [(SELECT min(n) FROM ag), (SELECT min(n) FROM ag WHERE n <> 'sum')], \
         coalesce((SELECT bool_and(json_type(x) = 'NULL' OR coalesce( \
             json_extract_string(x, '$.value.is_null'), 'false') = 'true') \
             FROM lim), true) \
         AND coalesce((SELECT bool_and(json_type(x) = 'NULL' OR coalesce( \
             json_extract_string(x, '$.value.value'), '') = '0') \
             FROM off), true), \
         (SELECT count(*) FROM t WHERE n = 'rowid') > 0 \
         FROM a";
    /// Window functions whose value is a row's POSITION among its peers.
    /// `rank` and its family are functions of the key and stay out; a plain
    /// aggregate over a whole partition is a set and stays out too.
    const POSITIONAL_WINDOWS: [&str; 7] = [
        "\"WINDOW_ROW_NUMBER\"",
        "\"WINDOW_NTILE\"",
        "\"WINDOW_FIRST_VALUE\"",
        "\"WINDOW_LAST_VALUE\"",
        "\"WINDOW_NTH_VALUE\"",
        "\"WINDOW_LEAD\"",
        "\"WINDOW_LAG\"",
    ];

    /// A frame bound counted in ROWS: which rows are in the frame is then
    /// the arrival order of the key's peers, not the key. RANGE and GROUPS
    /// bounds move by peer group and are functions of the key, and the two
    /// UNBOUNDED bounds carry no flavour at all because the whole partition
    /// is the whole partition either way.
    const ROW_FRAME_BOUNDS: [&str; 3] = [
        "\"CURRENT_ROW_ROWS\"",
        "\"EXPR_PRECEDING_ROWS\"",
        "\"EXPR_FOLLOWING_ROWS\"",
    ];

    // The clause NAME always comes from the tokens: DuckDB's parse folds
    // LIMIT, OFFSET and FETCH into one node, and the message quotes back the
    // clause the query actually wrote.
    let (limit_word, other_word) = ordering_words(py, sql)?;
    let shape_sql = SHAPE_SQL
        .replace("__CLOCK__", CLOCK_KEYWORDS)
        .replace("__RUN_STATE__", RUN_STATE_FUNCTIONS)
        .replace(
            "__ORDER_FREE__",
            &format!("{ORDER_FREE_AGGREGATES},{WINDOW_ONLY_FUNCTIONS}"),
        );
    let row = con
        .call_method1("execute", (shape_sql, (sql,)))?
        .call_method0("fetchone")?;
    let (
        error,
        n_statements,
        types,
        qualify,
        sample,
        distinct_on,
        joins,
        frame,
        drawn,
        aggs,
        no_op_limit,
        rowid,
    ): (
        Option<String>,
        Option<i64>,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
        Vec<Option<String>>,
        bool,
        bool,
    ) = row.extract()?;
    // A serialized value object opens with a brace; the absence of one is
    // DuckDB's `null` for that field, and an empty DISTINCT list is plain
    // DISTINCT, which collapses a set rather than picking out of a group.
    let present = |field: &Option<String>| field.as_deref().is_some_and(|s| s.contains('{'));
    // Both of these return before a single value rule above has been read, so
    // neither may depend on a clause word turning up: the word only sharpens
    // the message. A statement nobody could read is a statement in which
    // nothing was ruled out — no draw, no order-sensitive aggregate, no frame
    // — and the same goes for the statements of the string that were skipped.
    if error.as_deref() != Some("false") {
        return Ok(Shapes {
            row_limit: limit_word,
            refusal: Some(unreadable_refusal(other_word)),
        });
    }
    if n_statements != Some(1) {
        return Ok(Shapes {
            row_limit: limit_word,
            refusal: Some(multi_statement_refusal(other_word)),
        });
    }
    let types = types.unwrap_or_default();
    let row_limit = (types.contains("\"LIMIT_") && !no_op_limit)
        .then(|| limit_word.unwrap_or("LIMIT/OFFSET"));
    let frame = frame.unwrap_or_default();
    // The offender the message names, and the one it names once `sum` is not
    // one: `sum`'s order-dependence is its OVERLOAD's rather than its name's,
    // so it leaves the set when DuckDB bound an exact overload. Only ever
    // asked when `sum` is the name the message would carry -- where another
    // offender sorts before it, that one refuses either way.
    let mut aggs = aggs.into_iter();
    let agg = aggs.next().flatten();
    let agg = if agg.as_deref() == Some("sum") && exact_sum(con, sql)? {
        aggs.next().flatten()
    } else {
        agg
    };
    let refusal = if let Some(name) = drawn {
        Some(nondeterministic_refusal(&name))
    } else if let Some(name) = agg {
        Some(order_sensitive_agg_refusal(&name))
    } else if ROW_FRAME_BOUNDS.iter().any(|b| frame.contains(b)) {
        Some(positional_refusal(
            "a row-based window frame (ROWS PRECEDING/FOLLOWING/CURRENT ROW)",
        ))
    } else if present(&sample) {
        // A sample keeps a share of the rows, which is a row limit — but it
        // is answered on the far side of the fold, because the row path has
        // its own name for the clause and owes the caller that one.
        Some(row_limit_refusal("USING SAMPLE"))
    } else if present(&distinct_on) {
        Some(positional_refusal("DISTINCT ON"))
    } else if present(&qualify) {
        Some(positional_refusal("QUALIFY"))
    } else if POSITIONAL_WINDOWS.iter().any(|w| types.contains(w)) {
        Some(positional_refusal(
            "a row-position window function (row_number/ntile/lead/lag/first_value/last_value/nth_value)",
        ))
    } else if joins.is_some_and(|j| j.contains("\"POSITIONAL\"")) {
        // A POSITIONAL JOIN pairs row i of one relation with row i of the
        // other and asks nothing of their values, so which rows meet is the
        // scan order itself. Measured over a 300k static fed through a tying
        // GROUP BY, one such join answered five ways across the five settings
        // -- as a SCALAR count, not only as a sequence.
        Some(positional_refusal("POSITIONAL JOIN"))
    } else if rowid {
        Some(positional_refusal("the rowid pseudo-column"))
    } else {
        None
    };
    Ok(Shapes { row_limit, refusal })
}

/// Whether every `sum` this statement names bound one of DuckDB's EXACT
/// overloads — the ones it declares NOT_ORDER_DEPENDENT — rather than the
/// DOUBLE one that accumulates in floating point.
///
/// The question is answered by the RETURN TYPE, because that is what names
/// the overload: DuckDB's exact sums return HUGEINT (from BOOLEAN, SMALLINT,
/// INTEGER, BIGINT and HUGEINT) or DECIMAL, and only the order-dependent
/// overloads return anything else (DOUBLE from FLOAT and DOUBLE, and — this
/// is why the argument's own integrality does not answer it — DOUBLE from
/// UHUGEINT, which has no exact overload to cast to; BIGNUM from BIGNUM).
///
/// DuckDB's parse does not carry types, so its BINDER is asked: the statement
/// is re-projected onto its bare `sum` outputs with its modifiers dropped —
/// DuckDB's own parse in, DuckDB's own SQL out, as the tie probe does — and
/// DESCRIBEd, which binds without running. A sum the re-projection cannot
/// carry is one whose overload goes unread: nested in a larger expression, in
/// a HAVING, in a set operation or a subquery, or called as a window. Those
/// keep refusing, which is what the count check enforces — every `sum` the
/// parse names has to be one of the bare outputs, or the answer is no.
fn exact_sum(con: &Bound<'_, PyAny>, sql: &str) -> PyResult<bool> {
    const SUM_SQL: &str = "WITH a(j) AS (SELECT json_serialize_sql(?)), \
         q(j, node, sums) AS (SELECT j, json_extract(j, '$.statements[0].node'), \
             list_filter(coalesce(CAST(json_extract(j, \
                 '$.statements[0].node.select_list') AS JSON[]), []), \
                 x -> json_extract_string(x, '$.class') = 'FUNCTION' \
                     AND lower(json_extract_string(x, '$.function_name')) = 'sum') \
             FROM a), \
         f(n) AS (SELECT lower(unnest(coalesce( \
             json_extract_string(j, '$..function_name'), []))) FROM a) \
         SELECT json_deserialize_sql(json_merge_patch(j, json_object('statements', \
             json_array(json_merge_patch(json_extract(j, '$.statements[0]'), \
                 json_object('node', json_object('select_list', to_json(sums), \
                     'modifiers', to_json([]::JSON[])))))))), \
             len(sums) > 0 AND len(sums) = (SELECT count(*) FROM f WHERE n = 'sum') \
         FROM q";
    let (projected, all_bare): (Option<String>, bool) = con
        .call_method1("execute", (SUM_SQL, (sql,)))?
        .call_method0("fetchone")?
        .extract()?;
    let Some(projected) = projected.filter(|_| all_bare) else {
        return Ok(false);
    };
    let described = con.call_method1(
        "execute",
        (format!("SELECT column_type FROM (DESCRIBE {projected})"),),
    );
    let Ok(cur) = described else { return Ok(false) };
    let types: Vec<(String,)> = cur.call_method0("fetchall")?.extract()?;
    Ok(types
        .iter()
        .all(|(ty,)| ty == "HUGEINT" || ty.starts_with("DECIMAL")))
}

/// The clause words a statement spells, from DuckDB's own tokenizer: the
/// first row-limiting one, and the first that raises a question about order.
///
/// The tokenizer, not the text: a string literal or a quoted identifier
/// keeps its quotes in the token's own span, so `'limit'` and `"limit"`
/// cannot be mistaken for the clause.
///
/// `TOP` is deliberately not among them, although a row path elsewhere names
/// `SELECT TOP` as a limit. DuckDB has no such clause, so it never runs one —
/// and this reading is only ever reached AFTER DuckDB has run the statement,
/// which leaves `top` as an identifier every time it arrives here. Reading it
/// as a clause reported a row limit for a static column's name.
fn ordering_words(
    py: Python<'_>,
    sql: &str,
) -> PyResult<(Option<&'static str>, Option<&'static str>)> {
    let toks: Vec<(usize, Bound<'_, PyAny>)> = PyModule::import(py, "duckdb")?
        .call_method1("tokenize", (sql,))?
        .extract()?;
    let mut limit = None;
    let mut other = None;
    for (i, (start, kind)) in toks.iter().enumerate() {
        // 2 is duckdb.token_type.string_const, whose span swallows any
        // trailing comment; every other token starts with its own word.
        if kind.getattr("value")?.extract::<i64>()? == 2 {
            continue;
        }
        let end = toks.get(i + 1).map_or(sql.len(), |(s, _)| *s);
        let Some(word) = sql.get(*start..end).and_then(|s| s.split_whitespace().next()) else {
            continue;
        };
        let word = word.to_ascii_uppercase();
        let named = match word.as_str() {
            "FETCH" => Some("FETCH FIRST/NEXT"),
            "LIMIT" | "OFFSET" => Some("LIMIT/OFFSET"),
            _ => None,
        };
        if named.is_some() {
            limit = limit.or(named);
            continue;
        }
        other = other.or(match word.as_str() {
            "ORDER" => Some("ORDER BY"),
            "DISTINCT" => Some("DISTINCT"),
            "QUALIFY" => Some("QUALIFY"),
            "OVER" => Some("OVER"),
            "SAMPLE" | "TABLESAMPLE" => Some("USING SAMPLE"),
            _ => None,
        });
    }
    Ok((limit, other))
}

/// The top-level select list, in the two fields an ORDER BY NAME resolves
/// against: each entry's EXPLICIT alias (empty where it has none, and a star
/// never has one) and whether the entry is a star.
struct SelectList {
    aliases: Vec<String>,
    stars: Vec<bool>,
}

/// Which column DuckDB's binder answers an ORDER BY NAME with.
enum NameBinding {
    /// A column of the frozen result, by position.
    Out(usize),
    /// Not the output at all: the binder falls through to the query's own
    /// input, where the key has to be computed beside the rows.
    Input,
    /// An alias whose output position this reading cannot place.
    Unplaceable,
}

/// Where DuckDB's binder sends `ORDER BY <name>`, read off its own source.
///
/// A SELECT fills its alias map in select-list order and lets a later entry
/// OVERWRITE an earlier one (`bind_select_node.cpp`), so a name two columns
/// answer to binds the LAST one ALIASED to it. Only aliases are in that map:
/// a name that matches because a column happens to carry it goes to the
/// expression list instead, which binds it only when exactly one column
/// matches and otherwise leaves the term to the input (`order_binder.cpp`,
/// `TryGetProjectionReference`). Measured, `SELECT g AS c, v AS c FROM s
/// ORDER BY c DESC` comes back sorted by v.
///
/// A SET OPERATION is a different binder: it gathers the names of its
/// children's OUTPUT columns, aliased or not, and keeps the FIRST of a repeat
/// (`bind_setop_node.cpp`). It has no select list of its own, which is how
/// this reading tells the two apart.
///
/// An alias sits at its own place in the select list, which is its output
/// position until a star stands in front of it — a star expands to whatever
/// the result has left over, and one star's expansion is that count exactly.
/// Two stars split an unknown count two ways, so an alias behind them cannot
/// be placed, and an unplaceable key refuses rather than guessing.
fn resolve_name(names: &[String], sel: Option<&SelectList>, name: &str) -> NameBinding {
    let sole_match = || {
        let mut it = names
            .iter()
            .enumerate()
            .filter(|(_, n)| n.eq_ignore_ascii_case(name));
        match (it.next(), it.next()) {
            (Some((p, _)), None) => NameBinding::Out(p),
            _ => NameBinding::Input,
        }
    };
    let Some(sel) = sel else {
        return names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(name))
            .map_or(NameBinding::Input, NameBinding::Out);
    };
    let Some(i) = sel.aliases.iter().rposition(|a| a.eq_ignore_ascii_case(name)) else {
        return sole_match();
    };
    let star = sel.stars.iter().position(|s| *s);
    match star {
        None if sel.aliases.len() == names.len() => NameBinding::Out(i),
        Some(k) if sel.stars[k + 1..].iter().all(|s| !*s) && names.len() + 1 >= sel.aliases.len() => {
            let expanded = names.len() + 1 - sel.aliases.len();
            NameBinding::Out(if i < k { i } else { i - 1 + expanded })
        }
        _ => NameBinding::Unplaceable,
    }
}

/// Where one sort key can be read: off the frozen result, or only by
/// computing it where the ORDER BY computes it.
enum OrderKey {
    /// A column of the frozen result, by POSITION — so two output columns
    /// sharing a name are still two distinct, measurable keys.
    Out(usize),
    /// A key the result does not carry, named by its place in the ORDER BY.
    Hidden(usize),
}

/// The query whose rows carry every sort key, and the keys to group them by:
/// the probe that asks DuckDB whether the top-level ORDER BY leaves two rows
/// in an order the query does not state. `None` when the query sorts nothing;
/// `Err` names what stopped the reading, and an unruled-out tie refuses like
/// a found one.
///
/// Keys resolve as DuckDB resolves them, off DuckDB's own reading: an integer
/// literal and a bare `#N` are both a position in the OUTPUT; `ALL` is every
/// output column (and arrives as the star it is, so a column actually NAMED
/// `all` is not mistaken for it); a bare name goes where DuckDB's binder
/// sends it, which `resolve_name` reads off that binder's own source.
/// Anything else is computed over the query's own input, so it is added to
/// that query's projection — by rewriting DuckDB's parse of the statement and
/// asking DuckDB to print it back, which is why syntax no other parser reads
/// still measures. Its resolution is unchanged by the move: a bare name that
/// is not an output alias, and a `#N` written inside a larger expression,
/// both mean the input column in either place.
///
/// An ORDER BY below the top is not read at all: it orders nothing that this
/// path promises. Row order on the constant path is not part of the contract
/// — the differential compares static-only results as an unordered multiset
/// (`fuzz/oracle.py` compare_mode `constant-unordered`, and the oracle spec's
/// nondeterminism chapter) — so an inner sort changes only a sequence nobody
/// was promised. Measured, it left the SET identical under every setting.
fn tie_probe(
    con: &Bound<'_, PyAny>,
    sql: &str,
    names: &[String],
) -> PyResult<Result<Option<String>, &'static str>> {
    /// The top-level sort terms, one row each, reduced to the fields the
    /// reading turns on. The two spellings of a position share a column:
    /// an integer literal carries it in `value.value` and DuckDB's `#N`
    /// positional reference in `index`, and only one of the two is ever set.
    const ORDER_SQL: &str = "WITH a(j) AS (SELECT json_serialize_sql(?)), \
         m(mo) AS (SELECT unnest(CAST(json_extract(j, '$.statements[0].node.modifiers') \
             AS JSON[])) FROM a), \
         o(x) AS (SELECT unnest(CAST(json_extract(mo, '$.orders') AS JSON[])) FROM m \
             WHERE json_extract_string(mo, '$.type') = 'ORDER_MODIFIER') \
         SELECT json_extract_string(x, '$.expression.type'), \
             json_array_length(json_extract(x, '$.expression.column_names')), \
             json_extract_string(x, '$.expression.column_names[0]'), \
             coalesce(json_extract_string(x, '$.expression.value.value'), \
                 json_extract_string(x, '$.expression.index')), \
             json_extract_string(x, '$.expression.expr') IS NOT NULL \
                 OR coalesce(json_array_length( \
                     json_extract(x, '$.expression.exclude_list')), 0) > 0 \
                 OR coalesce(json_array_length( \
                     json_extract(x, '$.expression.replace_list')), 0) > 0 \
         FROM o";
    /// The statement with its top-level ORDER BY dropped and the named sort
    /// terms appended to its projection — DuckDB's own parse in, DuckDB's own
    /// SQL out. Also reports the two shapes an appended column would change
    /// the meaning of: a DISTINCT collapses on a different tuple, and a set
    /// operation has no projection to append to at all.
    const KEYED_SQL: &str = "WITH a(j) AS (SELECT json_serialize_sql(?)), \
         n(j, node) AS (SELECT j, json_extract(j, '$.statements[0].node') FROM a), \
         m(j, node, mods) AS ( \
             SELECT j, node, CAST(json_extract(node, '$.modifiers') AS JSON[]) FROM n), \
         k(j, node, kept, orders) AS ( \
             SELECT j, node, \
                 list_filter(mods, x -> json_extract_string(x, '$.type') <> 'ORDER_MODIFIER'), \
                 flatten(list_transform( \
                     list_filter(mods, x -> json_extract_string(x, '$.type') = 'ORDER_MODIFIER'), \
                     x -> CAST(json_extract(x, '$.orders') AS JSON[]))) \
             FROM m) \
         SELECT json_deserialize_sql(json_merge_patch(j, json_object('statements', \
                 json_array(json_merge_patch(json_extract(j, '$.statements[0]'), \
                     json_object('node', json_object( \
                         'modifiers', to_json(kept), \
                         'select_list', to_json(list_concat( \
                             CAST(json_extract(node, '$.select_list') AS JSON[]), \
                             list_transform( \
                                 list_filter( \
                                     list_transform(orders, (o, i) -> {'o': o, 'i': i - 1}), \
                                     p -> list_contains(CAST(? AS BIGINT[]), p.i)), \
                                 p -> json_merge_patch(json_extract(p.o, '$.expression'), \
                                     json_object('alias', '__confit_k' || p.i)))))))))))), \
             list_contains(list_transform(kept, x -> json_extract_string(x, '$.type')), \
                 'DISTINCT_MODIFIER') \
                 OR json_extract(node, '$.select_list') IS NULL \
         FROM k";
    /// The select list an ORDER BY NAME resolves against, entry by entry.
    /// NULL for a node that has none, which is the set operations.
    const SELECT_SQL: &str = "WITH a(j) AS (SELECT json_serialize_sql(?)), \
         n(sl) AS (SELECT CAST(json_extract(j, '$.statements[0].node.select_list') \
             AS JSON[]) FROM a) \
         SELECT list_transform(sl, x -> json_extract_string(x, '$.alias')), \
             list_transform(sl, x -> json_extract_string(x, '$.class') = 'STAR') \
         FROM n";
    type Term = (Option<String>, Option<i64>, Option<String>, Option<String>, bool);
    let terms: Vec<Term> = con
        .call_method1("execute", (ORDER_SQL, (sql,)))?
        .call_method0("fetchall")?
        .extract()?;
    let n_terms = terms.len();
    if n_terms == 0 {
        return Ok(Ok(None));
    }
    let (aliases, stars): (Option<Vec<String>>, Option<Vec<bool>>) = con
        .call_method1("execute", (SELECT_SQL, (sql,)))?
        .call_method0("fetchone")?
        .extract()?;
    let select_list = aliases
        .zip(stars)
        .map(|(aliases, stars)| SelectList { aliases, stars });
    let mut keys = Vec::with_capacity(n_terms);
    for (i, (ty, n_names, first_name, position, narrowed)) in terms.into_iter().enumerate() {
        match ty.as_deref() {
            // A star sorts by every OUTPUT column, so a tie under it is a
            // repeated output row — but only in the shape DuckDB itself reads
            // that way: the sole sort term, with no COLUMNS(...) expression
            // and no EXCLUDE/REPLACE list. `ORDER BY ALL` and a bare
            // `ORDER BY *` are both that shape. Anything else narrows the
            // star, and DuckDB then expands it over the FROM's INPUT columns
            // instead — different keys, which the frozen result does not
            // carry and this reading will not guess at.
            Some("STAR") if !narrowed && n_terms == 1 => {
                keys.clear();
                keys.extend((0..names.len()).map(OrderKey::Out));
                break;
            }
            Some("STAR") => return Ok(Err("a star sort key this reading cannot expand")),
            // A position in the OUTPUT, in either of DuckDB's two spellings:
            // an integer literal, and `#N`. Both resolve the same way in an
            // ORDER BY — its binder answers them from one function, which
            // indexes the projection (`order_binder.cpp`,
            // `TryGetProjectionReference`) — and the distinction matters
            // because `#N` means something ELSE in a select list, where it is
            // the Nth column of the FROM. So a `#N` sort key must be read
            // here rather than computed beside the rows, which would measure
            // a different column under the same name.
            //
            // Only a BARE one. Inside a larger expression DuckDB's binder
            // never reaches that function, and `#N` there IS the input
            // column — the same column the `Hidden` arm computes it in.
            //
            // Any other literal is a constant key, which orders nothing and
            // therefore ties everything — so it takes the `Hidden` arm below
            // and is measured as the tie it is. Out of range never arrives:
            // DuckDB refuses `ORDER BY 3` and `ORDER BY #3` over two output
            // columns at bind, before this statement has run.
            Some("VALUE_CONSTANT") | Some("POSITIONAL_REFERENCE")
                if position
                    .as_deref()
                    .and_then(|t| t.parse::<usize>().ok())
                    .is_some_and(|p| (1..=names.len()).contains(&p)) =>
            {
                let p: usize = position.unwrap_or_default().parse().expect("just parsed");
                keys.push(OrderKey::Out(p - 1));
            }
            // A bare name is an output column where DuckDB's binder makes it
            // one -- `resolve_name` is that rule, and a name two columns
            // answer to is where the two readings differ.
            Some("COLUMN_REF")
                if n_names == Some(1)
                    && names
                        .iter()
                        .any(|n| n.eq_ignore_ascii_case(first_name.as_deref().unwrap_or_default())) =>
            {
                let name = first_name.unwrap_or_default();
                match resolve_name(names, select_list.as_ref(), &name) {
                    NameBinding::Out(p) => keys.push(OrderKey::Out(p)),
                    NameBinding::Input => keys.push(OrderKey::Hidden(i)),
                    NameBinding::Unplaceable => {
                        return Ok(Err("a repeated output name this reading cannot place"))
                    }
                }
            }
            _ => keys.push(OrderKey::Hidden(i)),
        }
    }
    if keys.is_empty() {
        return Ok(Ok(None));
    }
    let hidden: Vec<i64> = keys
        .iter()
        .filter_map(|k| match k {
            OrderKey::Hidden(i) => Some(*i as i64),
            OrderKey::Out(_) => None,
        })
        .collect();
    // The wrapper renames the inner result POSITIONALLY: the output columns
    // first, then one column per key the projection had to grow.
    let mut cols: Vec<String> = (0..names.len()).map(|i| format!("__confit_c{i}")).collect();
    cols.extend(hidden.iter().map(|i| format!("__confit_k{i}")));
    // ALWAYS DuckDB's own round trip, even when nothing has to be appended:
    // the probe wraps this statement in a subquery, and the raw text carries
    // whatever the caller wrote after it. A trailing `;` or a trailing line
    // comment made that wrapper a syntax error, which arrived as a refusal of
    // a query with nothing wrong with it. Printing the parse back drops both.
    // The two blocks below are about the APPENDED column, so they only apply
    // when there is one: a DISTINCT would then collapse on a different tuple,
    // and a set operation has no projection to append to.
    let (rewritten, blocked): (Option<String>, Option<bool>) = con
        .call_method1("execute", (KEYED_SQL, (sql, &hidden)))?
        .call_method0("fetchone")?
        .extract()?;
    let inner = match rewritten {
        Some(s) if hidden.is_empty() || blocked != Some(true) => s,
        _ => return Ok(Err("a sort key its projection cannot carry")),
    };
    let group: Vec<&str> = keys
        .iter()
        .map(|k| match k {
            OrderKey::Out(p) => cols[*p].as_str(),
            OrderKey::Hidden(i) => cols[names.len()
                + hidden
                    .iter()
                    .position(|h| *h == *i as i64)
                    .expect("every hidden key is listed")]
            .as_str(),
        })
        .collect();
    Ok(Ok(Some(format!(
        "SELECT 1 FROM ({inner}) AS __confit_t({}) GROUP BY {} HAVING count(*) > 1",
        cols.join(", "),
        group.join(", ")
    ))))
}

/// The constant emitter: a static-tables-only query is evaluated ONCE, here
/// at build time, by DuckDB itself — nothing dynamic remains and no IR is
/// built at all. Statics materialize as native tables (duckdb's
/// registered-arrow scan path has divergent filter semantics — see the
/// builtin-pins spec). Returns the fixed row dicts plus the result schema, or
/// the refusal message for a frozen shape the query does not fix.
fn eval_static_only(
    py: Python<'_>,
    sql: &str,
    static_tables: &HashMap<String, Py<PyAny>>,
) -> PyResult<Result<(Vec<Py<PyAny>>, Py<PyAny>), String>> {
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
    // Every refusal below is answered only AFTER the query has run, because
    // a refusal that names the static-tables-only path may be pinned only on
    // a query that IS one. A dynamic query reaches here too — the fallback
    // tries this path whenever the row path could not lower the SQL — and it
    // fails on the row table DuckDB does not know, which is what carries the
    // row path's own error back to the caller untouched.
    let arrow = con
        .call_method1("execute", (sql,))?
        .call_method0("to_arrow_table")?;
    let schema = arrow.getattr("schema")?;
    // AFTER the statement has bound, so the catalogue the shapes are read
    // against is the one that answered it — an extension DuckDB autoloads for
    // an unknown function name is loaded by this execute and by nothing
    // earlier.
    let shapes = read_shapes(py, &con, sql)?;
    if let Some(clause) = shapes.row_limit {
        return Ok(Err(row_limit_refusal(clause)));
    }
    if let Some(msg) = shapes.refusal {
        return Ok(Err(msg));
    }
    let mut rows = Vec::new();
    for r in arrow.call_method0("to_pylist")?.try_iter()? {
        rows.push(r?.unbind());
    }
    // Ties are measured, not assumed: the question goes to DuckDB itself, on
    // this connection, over the result it has just produced. Fewer than two
    // rows cannot tie, and a statement whose parse could not be read has
    // already refused above.
    if rows.len() > 1 {
        let names: Vec<String> = schema.getattr("names")?.extract()?;
        let probe = match tie_probe(&con, sql, &names)? {
            Ok(p) => p,
            Err(why) => return Ok(Err(unmeasurable_order_refusal(why))),
        };
        if let Some(probe) = probe {
            match con.call_method1("execute", (probe,)) {
                // One group of two is one pair of rows in no stated order.
                Ok(cur) => {
                    if !cur.call_method0("fetchone")?.is_none() {
                        return Ok(Err(TIE_ORDER_REFUSAL.to_string()));
                    }
                }
                // The probe is DuckDB's own statement re-projected, so the
                // only way it does not run is a key that cannot be computed
                // beside the rows it orders — which is a tie left unruled-out.
                Err(e) => {
                    return Ok(Err(format!(
                        "{} ({e})",
                        unmeasurable_order_refusal("a sort key its projection cannot carry")
                    )))
                }
            }
        }
    }
    Ok(Ok((rows, schema.unbind())))
}

/// The execution backend: cranelift when it compiles, the interpreter as
/// the always-available fallback — an uncovered op must not fail prepare.
/// Both agree byte-for-byte by the 500-seed differential.
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
        lanes: &[plan::InputLane],
        out_cols: &[Col],
        plan: &[EmitField],
        fun: &Backend,
    ) -> PyResult<Marshaller> {
        Ok(Marshaller {
            in_names: lanes
                .iter()
                .map(|l| {
                    l.path
                        .iter()
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
            cols: lanes.iter().map(|l| col_for_lane(l, 0)).collect(),
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
        lanes: &[plan::InputLane],
        rows: &[Py<PyAny>],
        row_table: &str,
    ) -> PyResult<Vec<Py<PyAny>>> {
        for col in &mut self.cols {
            col.clear();
        }
        for row_obj in rows {
            let bound = row_obj.bind(py);
            let dict = bound.cast::<PyDict>().ok();
            for ((lane, path), col) in
                lanes.iter().zip(&self.in_names).zip(&mut self.cols)
            {
                let mut attr = match dict {
                    Some(d) => d.get_item(path[0].bind(py))?.ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Row for table '{row_table}' is missing attribute '{}'",
                            lane.name
                        ))
                    })?,
                    None => bound.getattr(path[0].bind(py)).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Row for table '{row_table}' is missing attribute '{}': {e}",
                            lane.name
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
                                lane.name
                            ))
                        })?,
                        Err(_) => attr.getattr(seg.bind(py)).map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Row for table '{row_table}' is missing attribute '{}': {e}",
                                lane.name
                            ))
                        })?,
                    };
                }
                let null = attr.is_none();
                match lane.kind {
                    // A PRESENCE lane's VALUE is that validity: its path
                    // walked to a struct NODE, not to a scalar.
                    plan::LaneKind::Present => col.push_present(!null),
                    plan::LaneKind::Value(ct) => {
                        if null && !ct.nullable {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "column '{}' is not nullable but a row has None",
                                lane.name
                            )));
                        }
                        push_input_cell(col, &lane.name, ct, &attr, null)?;
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
                        OutCol::Dec(v) => {
                            let (ok, x) = v[r];
                            d.set_item(
                                k,
                                if ok {
                                    dec_py(py, x, self.out_tys[*i])?
                                } else {
                                    py.None()
                                },
                            )?;
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

/// What a successful build produced. The two share only the Python
/// boundary: `Compiled` runs a program over the request rows, while a query
/// that reads nothing but static tables has no program at all — DuckDB
/// already answered it, so `Constant` just hands the answer back.
enum Engine {
    Compiled {
        fun: Backend,
        /// The program's row input, in IR order: name, SEGMENT path (the
        /// boundaries walk these, never split names) and KIND. Taken
        /// verbatim off `Prepared`, which is the only place it is built;
        /// the boundary does not assemble a lane list of its own.
        lanes: Vec<plan::InputLane>,
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
        // The row-shape contract: "filter" (default) is today's
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
        // plain column; the dotted lane name doubles as the ingest path
        // (segments are field names without dots, so '.' is unambiguous).
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
        // The IN-side duplicate check belongs HERE rather than in the IR
        // verifier: here the names are still IDENTIFIERS — the struct leaf
        // lanes, whose dotted display names are not, get appended below — so
        // a repeat is a real collision and refuses by name instead of
        // surfacing later as an internal verifier bug on every query over
        // the table.
        for (i, c) in in_cols.iter().enumerate() {
            if in_cols[..i].iter().any(|p| p.name == c.name) {
                return Err(build_err(format!(
                    "row table '{row_table}' has two columns named '{}'",
                    c.name
                )));
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
                        StructNode::Opaque // dotted names stay unreachable
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
            // The SAME parser the row path uses, so a static column types at
            // its declared arrow width instead of collapsing every integer
            // to int64. A non-scalar column is still omitted rather than
            // rejected — unreferenced ones cost nothing.
            let mut cols = Vec::new();
            let mut opaque = Vec::new();
            // Declared order, one entry per schema column: what a star sees.
            // A struct is ONE opaque star entry; its flattened leaves are
            // addressable by path but never expand under a star.
            let mut star = Vec::new();
            let mut structs = Vec::new();
            for (pos, (cname, rf)) in schema::arrow_static_schema(py, name, &schema_obj)?
                .into_iter()
                .enumerate()
            {
                match rf {
                    schema::RowField::Scalar { ty, nullable } => {
                        star.push(crate::specializer::plan::StarCol::Real(cols.len() as u32));
                        cols.push(Col {
                            name: cname,
                            ty: ColTy { ty, nullable },
                        })
                    }
                    // Kept, not dropped — see StaticTable::opaque.
                    schema::RowField::Opaque(aty) => {
                        star.push(crate::specializer::plan::StarCol::Opaque(cname.clone()));
                        opaque.push((cname, aty))
                    }
                    // A struct's scalar leaves ARE the lane set a static
                    // table already stores. The lanes keep their dotted
                    // names for DISPLAY; resolution walks the TREE pushed
                    // here. Under a star the struct is one opaque entry:
                    // EXCLUDE it or the query refuses by name.
                    schema::RowField::Struct { nullable, fields } => {
                        star.push(crate::specializer::plan::StarCol::Opaque(cname.clone()));
                        let tree = flatten_static(&mut cols, &cname, &fields, nullable);
                        structs.push(crate::specializer::plan::StructCol {
                            pos,
                            name: cname,
                            fields: tree,
                        });
                    }
                }
            }
            catalog.push(StaticTable {
                name: name.clone(),
                cols,
                opaque,
                star,
                structs,
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
        // The same closures the runtime uses, handed to the binder so a pure
        // udf can constant-fold at build.
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
                    // A refusal the fold itself decided: it names the frozen
                    // query, not the prepare error that got us here.
                    Ok(Err(msg)) => return Err(build_err(msg)),
                    Ok(Ok((rows, schema))) => {
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
        let lanes = prepared.input_lanes().to_vec();
        let marsh = if std::env::var_os("SPECIALIZER_GENERIC_BOUNDARY").is_some() {
            None
        } else {
            Some(RefCell::new(Marshaller::build(
                py,
                &lanes,
                &prepared.program.out_cols,
                &plan,
                &fun,
            )?))
        };
        Ok(DuckDBInferFn {
            engine: Engine::Compiled {
                fun,
                lanes,
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
    /// it REFUSES anything but `infer_rows([])` rather than dropping what it
    /// was handed.
    fn infer_rows(&self, py: Python<'_>, rows: Vec<Py<PyAny>>) -> PyResult<Vec<Py<PyAny>>> {
        self.run_rows(py, &rows)
    }

    /// The columnar boundary: a single-chunk pa.Table or RecordBatch in, a
    /// pa.Table out — zero per-value Python objects on either side. Columns
    /// match the row schema by name — a struct leaf by its path through the
    /// struct — at their exact declared arrow types, never a widening (cast
    /// first otherwise). Values are byte-identical to infer_rows(); under
    /// shape='map' the output aligns positionally with the input.
    fn infer_arrow(&self, py: Python<'_>, batch: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let (fun, lanes, out_cols, plan) = match &self.engine {
            Engine::Compiled {
                fun,
                lanes,
                out_cols,
                plan,
                ..
            } => (fun, lanes, out_cols, plan),
            Engine::Constant { .. } => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "infer_arrow: a static-tables-only query emits fixed rows — \
                     use infer_rows([])",
                ))
            }
        };
        let input = arrow::ingest(py, &batch, lanes)?;
        let mut st = fun.new_state();
        fun.run(&input, &mut st)
            .map_err(|t| PyErr::from(InterpError::Eval(t.0)))?;
        arrow::emit(py, out_cols, plan, &st)
    }
}

impl DuckDBInferFn {
    fn run_rows(&self, py: Python<'_>, rows: &[Py<PyAny>]) -> PyResult<Vec<Py<PyAny>>> {
        let (fun, lanes, out_cols, plan, marsh) = match &self.engine {
            Engine::Compiled {
                fun,
                lanes,
                out_cols,
                plan,
                marsh,
            } => (fun, lanes, out_cols, plan, marsh),
            Engine::Constant { rows: fixed, .. } => {
                // This build reads only static tables, so it cannot see
                // input rows at all — and silently dropping them was the one
                // mistake at this boundary that did not refuse by name. It
                // hides a real caller bug: N request rows through a
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
                return m.call(py, fun, lanes, rows, &self.row_table);
            }
        }

        let n = rows.len();
        let mut cols: Vec<ColData> = lanes.iter().map(|l| col_for_lane(l, n)).collect();
        for row_obj in rows {
            let bound = row_obj.bind(py);
            // Dict rows are part of the API surface; the baseline path must
            // accept the same inputs as the marshaller, differing only in
            // cost (adversarial-review finding, 2026-07-26).
            let dict = bound.cast::<PyDict>().ok();
            for (lane, col) in lanes.iter().zip(&mut cols) {
                let mut segs = lane.path.iter().map(|s| s.as_str());
                let first = segs.next().expect("a path is never empty");
                let mut attr = match dict {
                    Some(d) => d.get_item(first)?.ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Row for table '{}' is missing attribute '{}'",
                            self.row_table, lane.name
                        ))
                    })?,
                    None => bound.getattr(first).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Row for table '{}' is missing attribute '{}': {e}",
                            self.row_table, lane.name
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
                                self.row_table, lane.name
                            ))
                        })?,
                        Err(_) => attr.getattr(seg).map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Row for table '{}' is missing attribute '{}': {e}",
                                self.row_table, lane.name
                            ))
                        })?,
                    };
                }
                let null = attr.is_none();
                match lane.kind {
                    // A PRESENCE lane's VALUE is that validity.
                    plan::LaneKind::Present => col.push_present(!null),
                    plan::LaneKind::Value(ct) => {
                        if null && !ct.nullable {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "column '{}' is not nullable but a row has None",
                                lane.name
                            )));
                        }
                        push_input_cell(col, &lane.name, ct, &attr, null)?;
                    }
                }
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
                        OutCol::Dec(v) => {
                            let (ok, x) = v[r];
                            dict.set_item(
                                name,
                                if ok {
                                    dec_py(py, x, out_cols[*i].ty.ty)?
                                } else {
                                    py.None()
                                },
                            )?;
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
