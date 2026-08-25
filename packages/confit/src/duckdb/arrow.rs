//! The columnar boundary (TASK-60): `infer_arrow(pa.Table) -> pa.Table`.
//!
//! Ingest walks pyarrow buffers directly (address + size via the Python
//! buffer API — no arrow-rs dependency) into the engine's `ColData` lanes;
//! emit builds `pa.Array.from_buffers` per OUTPUT COLUMN from rust-built
//! buffers. Zero per-value Python objects on either side — the measured
//! ~1.4 µs/row of boxing at 31-column width simply disappears. v1 copies
//! buffers (the proposal's copy-first recommendation); zero-copy lanes are
//! a measured follow-up, not assumed.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

use crate::error::InterpError;
use crate::specializer::exec::tree_ensemble::{ModelRows, NodeRows, TreeEnsemble};
use crate::specializer::exec::{Batch, ColData, OutCol, RunState};
use crate::specializer::ir::{Col, Ty};

fn err(msg: impl Into<String>) -> PyErr {
    InterpError::Build(msg.into()).into()
}

/// One arrow array's raw view: offset-adjusted validity + buffers.
struct RawArray<'py> {
    /// Kept alive for the duration of the buffer reads.
    _arr: Bound<'py, PyAny>,
    offset: usize,
    len: usize,
    bufs: Vec<Option<(usize, usize)>>, // (address, size) per buffer slot
}

impl RawArray<'_> {
    fn valid(&self, i: usize) -> bool {
        match self.bufs.first().copied().flatten() {
            None => true,
            Some((addr, size)) => {
                let bit = self.offset + i;
                let byte = bit / 8;
                debug_assert!(byte < size);
                let b = unsafe { *(addr as *const u8).add(byte) };
                (b >> (bit % 8)) & 1 == 1
            }
        }
    }

    /// Typed data pointer starting at the array's offset. Unaligned by
    /// contract — see [`Unaligned`].
    unsafe fn data<T: Copy>(&self, slot: usize) -> Unaligned<T> {
        let (addr, _) = self.bufs[slot].expect("data buffer present");
        Unaligned(unsafe { (addr as *const T).add(self.offset) })
    }
}

/// A pointer into an Arrow value buffer, which carries NO alignment
/// guarantee (TASK-67).
///
/// Arrow does not promise naturally aligned value buffers, and pyarrow
/// zero-copies one that is not: `np.frombuffer(blob, np.float64, offset=4)`
/// — a packed binary record, an mmap with a header — goes straight through
/// `pa.array()` and pyarrow computes over it perfectly. Dereferencing that
/// as `*const f64` is UB. A debug build traps it as a NON-UNWINDING panic,
/// so a single request batch ends the serving process rather than raising;
/// release merely happens not to fault today, which is a property of this
/// week's codegen and not a guarantee (nothing stops LLVM turning the row
/// loop into aligned vector moves).
///
/// Hence a newtype rather than a raw `*const T`: `get` is the only way to
/// read one, so the unchecked deref is not spellable. On x86-64
/// `read_unaligned` lowers to the same `mov` as an aligned load for a
/// scalar — what it costs is exactly the autovectorisation that would have
/// introduced the fault.
#[derive(Clone, Copy)]
struct Unaligned<T>(*const T);

impl<T: Copy> Unaligned<T> {
    /// # Safety
    /// `i` must be in bounds of the buffer this pointer came from.
    #[inline]
    unsafe fn get(self, i: usize) -> T {
        unsafe { self.0.add(i).read_unaligned() }
    }
}

/// A string array's character bytes (buffer slot 2).
///
/// `from_raw_parts` requires a non-null, aligned pointer EVEN AT LEN 0, so a
/// missing or empty data buffer cannot go through it — `unwrap_or((0, 0))`
/// built a null slice, which is UB in its own right. pyarrow refuses to
/// construct a string array without a data buffer today (`ArrowInvalid: Value
/// data buffer is null`), so this is hardening rather than a live bug, but it
/// is the same "a raw address is not a Rust reference" mistake as TASK-67.
fn str_bytes<'a>(raw: &'a RawArray<'_>) -> &'a [u8] {
    match raw.bufs.get(2).copied().flatten() {
        Some((addr, size)) if size > 0 => unsafe {
            std::slice::from_raw_parts(addr as *const u8, size)
        },
        _ => &[],
    }
}

fn raw_array<'py>(arr: Bound<'py, PyAny>) -> PyResult<RawArray<'py>> {
    let offset: usize = arr.getattr("offset")?.extract()?;
    let len: usize = arr.call_method0("__len__")?.extract()?;
    let mut bufs = Vec::new();
    for b in arr.call_method0("buffers")?.try_iter()? {
        let b = b?;
        if b.is_none() {
            bufs.push(None);
        } else {
            let addr: usize = b.getattr("address")?.extract()?;
            let size: usize = b.getattr("size")?.extract()?;
            bufs.push(Some((addr, size)));
        }
    }
    Ok(RawArray {
        _arr: arr,
        offset,
        len,
        bufs,
    })
}

/// A single-chunk column by NAME from a Table or RecordBatch. `what` names
/// the caller's table in every error ("infer_arrow", "model set 'trees'
/// nodes") so the message points at the input the user actually passed.
fn column<'py>(
    batch: &Bound<'py, PyAny>,
    name: &str,
    what: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let schema = batch.getattr("schema")?;
    let idx: i64 = schema
        .call_method1("get_field_index", (name,))?
        .extract()?;
    if idx < 0 {
        return Err(err(format!("{what}: missing column '{name}'")));
    }
    let col = batch.call_method1("column", (idx,))?;
    // Table columns are ChunkedArrays; RecordBatch columns are Arrays.
    if col.hasattr("num_chunks")? {
        let n: usize = col.getattr("num_chunks")?.extract()?;
        match n {
            1 => col.call_method1("chunk", (0,)),
            0 => col.call_method1("combine_chunks", ()),
            _ => Err(err(format!(
                "{what}: column '{name}' has {n} chunks — call \
                 combine_chunks() on the table first"
            ))),
        }
    } else {
        Ok(col)
    }
}

/// The leaf array for one lane, plus every struct ancestor along its path
/// (TASK-114). A path is the lane's SEGMENT list (TASK-132): `[name]` for a
/// plain column — dots included, a name is not a path — and the struct walk
/// for a leaf lane.
///
/// The ancestors come back because DuckDB folds a struct's validity into
/// each child at ARROW INGEST (src/function/table/arrow_conversion.cpp,
/// `ColumnArrowToDuckDB`, the STRUCT case: it copies the child's own mask
/// and then explicitly invalidates every slot the parent invalidated, then
/// recurses with that mask). `StructExtractFunction`
/// (src/function/scalar/struct/struct_extract.cpp) is three statements and
/// folds nothing — it only references the child vector. So the fold belongs
/// HERE, at our own arrow ingest, not at the extract.
///
/// Each ancestor keeps its OWN `RawArray` because each carries its own
/// offset: pyarrow composes a child's offset with its parent's
/// (`GetEffectiveOffset` = array.offset + parent_offset + chunk_offset), so
/// a sliced struct hands back ancestors at different offsets and every
/// bitmap must be read at the one it came with.
fn walk_lane<'py>(
    batch: &Bound<'py, PyAny>,
    path: &[String],
    display: &str,
) -> PyResult<(Bound<'py, PyAny>, Vec<RawArray<'py>>)> {
    let mut arr = column(batch, &path[0], "infer_arrow")?;
    let mut parents = Vec::new();
    for seg in &path[1..] {
        let ty = arr.getattr("type")?;
        // `get_field_index` exists only on a struct type, and returns -1 for
        // a missing AND for an ambiguous name — one refusal covers both,
        // where `.field(name)` would raise the same KeyError for either.
        let idx: i64 = match ty.call_method1("get_field_index", (seg.as_str(),)) {
            Ok(v) => v.extract()?,
            Err(_) => {
                return Err(err(format!(
                    "infer_arrow: column '{display}': '{}' is {}, the schema \
                     declares a struct — cast first",
                    path[0],
                    ty.str()?
                )))
            }
        };
        if idx < 0 {
            return Err(err(format!(
                "infer_arrow: column '{display}': the batch's struct '{}' has \
                 no field '{seg}'",
                path[0]
            )));
        }
        let child = arr.call_method1("field", (idx,))?;
        parents.push(raw_array(arr)?);
        arr = child;
    }
    Ok((arr, parents))
}

/// pyarrow batch -> engine Batch, matching `in_cols` by their lane PATH with
/// strict dtypes (int64 / double / string|large_string / bool).
pub fn ingest<'py>(
    py: Python<'_>,
    batch: &Bound<'py, PyAny>,
    in_cols: &[Col],
    in_paths: &[Vec<String>],
) -> PyResult<Batch> {
    let _ = py;
    let rows: usize = batch.call_method0("__len__")?.extract()?;
    let mut cols = Vec::with_capacity(in_cols.len());
    for (c, path) in in_cols.iter().zip(in_paths) {
        let (arr, parents) = walk_lane(batch, path, &c.name)?;
        let dtype: String = arr.getattr("type")?.str()?.extract()?;
        let ok = matches!(
            (c.ty.ty, dtype.as_str()),
            (Ty::I8, "int8")
                | (Ty::I16, "int16")
                | (Ty::I32, "int32")
                | (Ty::I64, "int64")
                | (Ty::F64, "double")
                | (Ty::I1, "bool")
                | (Ty::Str, "string")
                | (Ty::Str, "large_string")
        );
        if !ok {
            return Err(err(format!(
                "infer_arrow: column '{}' is {dtype}, the schema declares {} — cast first",
                c.name,
                c.ty.ty.name()
            )));
        }
        let large = dtype == "large_string";
        let raw = raw_array(arr)?;
        // The null lane is the OR of EVERY level: a NOT NULL child under a
        // nullable parent has no validity buffer of its own, and its data
        // buffer under a null parent holds an arbitrary live value (measured:
        // a `.field()` walk without this fold serves 1000000 where DuckDB
        // serves NULL). On the all-scalar path `parents` is an empty Vec,
        // which does not allocate, and `[].iter().all(..)` is a length
        // compare. Everything else below reads the LEAF, whose own offset is
        // already the composed one.
        let valid_at = |i: usize| parents.iter().all(|p| p.valid(i)) && raw.valid(i);
        if raw.len != rows {
            return Err(err(format!(
                "infer_arrow: column '{}' has {} rows, the batch {rows}",
                c.name, raw.len
            )));
        }
        let mut null_seen = false;
        let col_data = match c.ty.ty {
            // Narrow widths widen into the engine's i64 lane; the declared
            // arrow type already proved every value fits.
            Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => {
                let mut valid = Vec::with_capacity(rows);
                let mut data = Vec::with_capacity(rows);
                for i in 0..rows {
                    let v = valid_at(i);
                    null_seen |= !v;
                    valid.push(v);
                    data.push(if v {
                        match c.ty.ty {
                            Ty::I8 => unsafe { raw.data::<i8>(1).get(i) as i64 },
                            Ty::I16 => unsafe { raw.data::<i16>(1).get(i) as i64 },
                            Ty::I32 => unsafe { raw.data::<i32>(1).get(i) as i64 },
                            _ => unsafe { raw.data::<i64>(1).get(i) },
                        }
                    } else {
                        0
                    });
                }
                ColData::I64 { valid, data }
            }
            Ty::F64 => {
                let mut valid = Vec::with_capacity(rows);
                let mut data = Vec::with_capacity(rows);
                let p = unsafe { raw.data::<f64>(1) };
                for i in 0..rows {
                    let v = valid_at(i);
                    null_seen |= !v;
                    valid.push(v);
                    data.push(if v { unsafe { p.get(i) } } else { 0.0 });
                }
                ColData::F64 { valid, data }
            }
            Ty::I1 => {
                let mut valid = Vec::with_capacity(rows);
                let mut data = Vec::with_capacity(rows);
                let (addr, _) = raw.bufs[1].expect("bool data buffer");
                for i in 0..rows {
                    let v = valid_at(i);
                    null_seen |= !v;
                    valid.push(v);
                    let bit = raw.offset + i;
                    let b = unsafe { *(addr as *const u8).add(bit / 8) };
                    data.push(v && (b >> (bit % 8)) & 1 == 1);
                }
                ColData::I1 { valid, data }
            }
            Ty::Str => {
                let mut col = ColData::Str {
                    valid: Vec::with_capacity(rows),
                    buf: String::new(),
                    spans: Vec::with_capacity(rows),
                };
                let bytes = str_bytes(&raw);
                for i in 0..rows {
                    let v = valid_at(i);
                    null_seen |= !v;
                    if !v {
                        col.push_str_cell(false, "");
                        continue;
                    }
                    let (lo, hi) = if large {
                        let p = unsafe { raw.data::<i64>(1) };
                        unsafe { (p.get(i) as usize, p.get(i + 1) as usize) }
                    } else {
                        let p = unsafe { raw.data::<i32>(1) };
                        unsafe { (p.get(i) as usize, p.get(i + 1) as usize) }
                    };
                    let s = std::str::from_utf8(&bytes[lo..hi])
                        .map_err(|_| err("infer_arrow: invalid UTF-8 in string column"))?;
                    col.push_str_cell(true, s);
                }
                col
            }
            // A decimal ROW column is opaque (schema.rs, Policy::Row), so
            // the arrow INPUT path needs no decimal arm at all.
            Ty::Dec(..) => unreachable!("a decimal row column is opaque"),
        };
        if null_seen && !c.ty.nullable {
            return Err(err(format!(
                "column '{}' is not nullable but the batch has NULLs",
                c.name
            )));
        }
        cols.push(col_data);
    }
    Ok(Batch { rows, cols })
}

// --------------------------------------------------------- model sets --

/// One column of a model-set table, null-checked. A NULL anywhere in these
/// nine columns is a malformed extract, not a value the kernel can score, so
/// it refuses here with the row index rather than reaching `TreeEnsemble`.
fn model_col<'py>(
    table: &Bound<'py, PyAny>,
    what: &str,
    name: &str,
) -> PyResult<(RawArray<'py>, String)> {
    let arr = column(table, name, what)?;
    let dtype: String = arr.getattr("type")?.str()?.extract()?;
    let raw = raw_array(arr)?;
    for i in 0..raw.len {
        if !raw.valid(i) {
            return Err(err(format!(
                "{what}: column '{name}' has a NULL at row {i}"
            )));
        }
    }
    Ok((raw, dtype))
}

/// int32 and int64 both read as i64 — `pa.array([1, 2])` defaults to int64,
/// and refusing that would be a papercut with no safety value.
fn ints(table: &Bound<'_, PyAny>, what: &str, name: &str) -> PyResult<Vec<i64>> {
    let (raw, dtype) = model_col(table, what, name)?;
    match dtype.as_str() {
        "int64" => Ok((0..raw.len)
            .map(|i| unsafe { raw.data::<i64>(1).get(i) })
            .collect()),
        "int32" => Ok((0..raw.len)
            .map(|i| unsafe { raw.data::<i32>(1).get(i) as i64 })
            .collect()),
        d => Err(err(format!(
            "{what}: column '{name}' is {d}, want int32 or int64"
        ))),
    }
}

/// [`ints`] narrowed: `feature`, `left` and `right` are i32 in the kernel.
fn small_ints(table: &Bound<'_, PyAny>, what: &str, name: &str) -> PyResult<Vec<i32>> {
    ints(table, what, name)?
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            i32::try_from(v).map_err(|_| {
                err(format!(
                    "{what}: column '{name}' row {i} is {v}, outside the int32 range"
                ))
            })
        })
        .collect()
}

fn doubles(table: &Bound<'_, PyAny>, what: &str, name: &str) -> PyResult<Vec<f64>> {
    let (raw, dtype) = model_col(table, what, name)?;
    if dtype != "double" {
        return Err(err(format!(
            "{what}: column '{name}' is {dtype}, want double"
        )));
    }
    Ok((0..raw.len)
        .map(|i| unsafe { raw.data::<f64>(1).get(i) })
        .collect())
}

fn bools(table: &Bound<'_, PyAny>, what: &str, name: &str) -> PyResult<Vec<bool>> {
    let (raw, dtype) = model_col(table, what, name)?;
    if dtype != "bool" {
        return Err(err(format!(
            "{what}: column '{name}' is {dtype}, want bool"
        )));
    }
    let (addr, _) = raw.bufs[1].expect("bool data buffer");
    Ok((0..raw.len)
        .map(|i| {
            let bit = raw.offset + i;
            let b = unsafe { *(addr as *const u8).add(bit / 8) };
            (b >> (bit % 8)) & 1 == 1
        })
        .collect())
}

fn strings(table: &Bound<'_, PyAny>, what: &str, name: &str) -> PyResult<Vec<String>> {
    let (raw, dtype) = model_col(table, what, name)?;
    let large = match dtype.as_str() {
        "string" => false,
        "large_string" => true,
        d => {
            return Err(err(format!(
                "{what}: column '{name}' is {d}, want string"
            )))
        }
    };
    let bytes = str_bytes(&raw);
    (0..raw.len)
        .map(|i| {
            let (lo, hi) = if large {
                let p = unsafe { raw.data::<i64>(1) };
                unsafe { (p.get(i) as usize, p.get(i + 1) as usize) }
            } else {
                let p = unsafe { raw.data::<i32>(1) };
                unsafe { (p.get(i) as usize, p.get(i + 1) as usize) }
            };
            std::str::from_utf8(&bytes[lo..hi])
                .map(str::to_owned)
                .map_err(|_| err(format!("{what}: invalid UTF-8 in column '{name}'")))
        })
        .collect()
}

/// The `models=` boundary: a node table and a header table in, a validated
/// [`TreeEnsemble`] out. Nothing but Arrow crosses — no estimator object, no
/// pickle, no live model reference. Column layout is the spec's:
///
/// ```text
/// nodes:  model_id | tree_id | node_id | feature (-1 = leaf) | threshold
///                  | left | right | missing_left | value
/// models: model_id | base | agg ('sum'|'mean') | link ('identity'|'sigmoid')
/// ```
pub fn ensemble(
    nodes: &Bound<'_, PyAny>,
    headers: &Bound<'_, PyAny>,
    n_features: u32,
    set: &str,
) -> PyResult<TreeEnsemble> {
    let nw = format!("model set '{set}' nodes");
    let hw = format!("model set '{set}' models");
    let (model_id, tree_id, node_id) = (
        ints(nodes, &nw, "model_id")?,
        ints(nodes, &nw, "tree_id")?,
        ints(nodes, &nw, "node_id")?,
    );
    let feature = small_ints(nodes, &nw, "feature")?;
    let threshold = doubles(nodes, &nw, "threshold")?;
    let left = small_ints(nodes, &nw, "left")?;
    let right = small_ints(nodes, &nw, "right")?;
    let missing_left = bools(nodes, &nw, "missing_left")?;
    let value = doubles(nodes, &nw, "value")?;

    let h_model_id = ints(headers, &hw, "model_id")?;
    let base = doubles(headers, &hw, "base")?;
    let agg = strings(headers, &hw, "agg")?;
    let link = strings(headers, &hw, "link")?;
    let agg: Vec<&str> = agg.iter().map(String::as_str).collect();
    let link: Vec<&str> = link.iter().map(String::as_str).collect();

    TreeEnsemble::new(
        &NodeRows {
            model_id: &model_id,
            tree_id: &tree_id,
            node_id: &node_id,
            feature: &feature,
            threshold: &threshold,
            left: &left,
            right: &right,
            missing_left: &missing_left,
            value: &value,
        },
        &ModelRows {
            model_id: &h_model_id,
            base: &base,
            agg: &agg,
            link: &link,
        },
        n_features,
    )
    .map_err(|e| err(format!("model set '{set}': {e}")))
}

fn bitmap(oks: impl Iterator<Item = bool>, n: usize) -> (Vec<u8>, usize) {
    let mut bits = vec![0u8; n.div_ceil(8)];
    let mut nulls = 0usize;
    for (i, ok) in oks.enumerate() {
        if ok {
            bits[i / 8] |= 1 << (i % 8);
        } else {
            nulls += 1;
        }
    }
    (bits, nulls)
}

/// The raw little-endian bytes of a numeric buffer, for `pa.py_buffer`.
fn cast_bytes<T>(data: &[T], size: usize) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * size) }.to_vec()
}

fn pa_ty<'py>(pa: &Bound<'py, PyModule>, t: Ty) -> PyResult<Bound<'py, PyAny>> {
    match t {
        Ty::I1 => pa.call_method0("bool_"),
        Ty::I8 => pa.call_method0("int8"),
        Ty::I16 => pa.call_method0("int16"),
        Ty::I32 => pa.call_method0("int32"),
        Ty::I64 => pa.call_method0("int64"),
        Ty::F64 => pa.call_method0("float64"),
        Ty::Str => pa.call_method0("string"),
        // decimal128 at EVERY tier, because that is what DuckDB exports:
        // `SetArrowFormat` writes bit_width 128 regardless of the internal
        // int16/int32/int64/int128 storage under the default
        // ArrowFormatVersion::V1_0 (arrow_converter.cpp:236-262).
        Ty::Dec(p, s) => pa.call_method1("decimal128", (p, s)),
    }
}

/// The output contract as a `pa.Schema` — field for field what [`emit`]'s
/// `from_arrays` table carries (same declared widths, `list_`/`struct` for
/// wide UDF fields, default-nullable like DuckDB's own `.arrow()` export).
pub fn output_schema(
    py: Python<'_>,
    out_cols: &[Col],
    plan: &[super::EmitField],
) -> PyResult<Py<PyAny>> {
    let pa = PyModule::import(py, "pyarrow")?;
    let mut fields = Vec::with_capacity(plan.len());
    for field in plan {
        let (name, ty) = match field {
            super::EmitField::Scalar(i) => {
                let c = &out_cols[*i];
                (c.name.clone(), pa_ty(&pa, c.ty.ty)?)
            }
            super::EmitField::Wide {
                name,
                first,
                width,
                names: field_names,
                ..
            } => {
                let ty = if field_names.is_empty() {
                    pa.call_method1("list_", (pa_ty(&pa, out_cols[*first].ty.ty)?,))?
                } else {
                    let members = field_names
                        .iter()
                        .zip(&out_cols[*first..*first + *width])
                        .map(|(fname, c)| {
                            pa.call_method1("field", (fname.as_str(), pa_ty(&pa, c.ty.ty)?))
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    pa.call_method1("struct", (PyList::new(py, members)?,))?
                };
                (name.clone(), ty)
            }
        };
        fields.push(pa.call_method1("field", (name, ty))?);
    }
    Ok(pa
        .call_method1("schema", (PyList::new(py, fields)?,))?
        .unbind())
}

/// Engine output -> pa.Table, one `Array.from_buffers` per scalar field; a
/// wide UDF field assembles as `pa.array` of `None | [components]` (the
/// python-list path — the wide boundary is transformer output, not the
/// scalar hot lane).
pub fn emit(
    py: Python<'_>,
    out_cols: &[Col],
    plan: &[super::EmitField],
    st: &RunState,
) -> PyResult<Py<PyAny>> {
    let pa = PyModule::import(py, "pyarrow")?;
    let from_buffers = pa.getattr("Array")?.getattr("from_buffers")?;
    let py_buffer = pa.getattr("py_buffer")?;
    let n = st.emitted;
    let mut arrays = Vec::with_capacity(plan.len());
    let mut names = Vec::with_capacity(plan.len());
    for field in plan {
        let (c, oc) = match field {
            super::EmitField::Scalar(i) => (&out_cols[*i], &st.out[*i]),
            super::EmitField::Wide {
                name,
                valid,
                first,
                width,
                names: field_names,
            } => {
                names.push(name.clone());
                let lane_ty = |t: crate::specializer::ir::Ty| match t {
                    crate::specializer::ir::Ty::I1 => pa.call_method0("bool_"),
                    // struct_pack fields are arbitrary expressions, so the
                    // width is real here (m-8 phase-2 campaign, 423 schema
                    // findings: ord() inside a struct is int32 on DuckDB).
                    // wide_py refuses out-of-range children by name before
                    // pa.array ever sees the values.
                    crate::specializer::ir::Ty::I8 => pa.call_method0("int8"),
                    crate::specializer::ir::Ty::I16 => pa.call_method0("int16"),
                    crate::specializer::ir::Ty::I32 => pa.call_method0("int32"),
                    crate::specializer::ir::Ty::I64 => pa.call_method0("int64"),
                    crate::specializer::ir::Ty::F64 => pa.call_method0("float64"),
                    // `string`, not `large_string` — see the scalar lane
                    // below and TASK-72.
                    crate::specializer::ir::Ty::Str => pa.call_method0("string"),
                    // A struct_pack / UDF child over a DECIMAL refuses at
                    // bind (m-8 lattice phase 5).
                    crate::specializer::ir::Ty::Dec(..) => {
                        unreachable!("a wide field child is never a decimal")
                    }
                };
                // Unnamed extern: list<elem>; named extern: struct keyed by
                // the declared names (slice 5), matching DuckDB's output.
                let out_ty = if field_names.is_empty() {
                    pa.call_method1("list_", (lane_ty(out_cols[*first].ty.ty)?,))?
                } else {
                    let members = field_names
                        .iter()
                        .zip(&out_cols[*first..*first + *width])
                        .map(|(fname, c)| {
                            pa.call_method1("field", (fname.as_str(), lane_ty(c.ty.ty)?))
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    pa.call_method1("struct", (PyList::new(py, members)?,))?
                };
                let child_tys: Vec<crate::specializer::ir::Ty> = out_cols
                    [*first..*first + *width]
                    .iter()
                    .map(|c| c.ty.ty)
                    .collect();
                let mut values = Vec::with_capacity(n);
                for r in 0..n {
                    values.push(super::wide_py(
                        py,
                        st,
                        *valid,
                        *first,
                        *width,
                        field_names,
                        &child_tys,
                        name,
                        r,
                    )?);
                }
                let vals = PyList::new(py, values)?;
                let kw = pyo3::types::PyDict::new(py);
                kw.set_item("type", out_ty)?;
                arrays.push(pa.call_method("array", (vals,), Some(&kw))?);
                continue;
            }
        };
        names.push(c.name.clone());
        let (dtype, validity, bufs): (Bound<'_, PyAny>, (Vec<u8>, usize), Vec<Py<PyAny>>) =
            match oc {
                OutCol::I64(v) => {
                    // The column's declared width narrows the emitted buffer
                    // (values compute in the i64 lane). Out of range refuses
                    // by name: every such input DuckDB itself traps on, and
                    // our matching runtime trap is m-8 phase 3.
                    let vb = bitmap(v.iter().map(|(ok, _)| *ok), n);
                    let ty = c.ty.ty;
                    let narrow_err = |x: i64| {
                        let ty_name = super::arrow_ty_name(ty);
                        err(format!(
                            "infer_arrow: column '{}' value {x} is outside its \
                             {ty_name} range — the {ty_name} overflow trap lands \
                             with m-8 phase 3",
                            c.name,
                        ))
                    };
                    let (dtype, raw): (_, Vec<u8>) = match ty {
                        Ty::I32 => {
                            let mut data: Vec<i32> = Vec::with_capacity(v.len());
                            for (ok, x) in v.iter() {
                                data.push(if *ok {
                                    i32::try_from(*x).map_err(|_| narrow_err(*x))?
                                } else {
                                    0
                                });
                            }
                            (pa.call_method0("int32")?, cast_bytes(&data, 4))
                        }
                        Ty::I16 => {
                            let mut data: Vec<i16> = Vec::with_capacity(v.len());
                            for (ok, x) in v.iter() {
                                data.push(if *ok {
                                    i16::try_from(*x).map_err(|_| narrow_err(*x))?
                                } else {
                                    0
                                });
                            }
                            (pa.call_method0("int16")?, cast_bytes(&data, 2))
                        }
                        Ty::I8 => {
                            let mut data: Vec<i8> = Vec::with_capacity(v.len());
                            for (ok, x) in v.iter() {
                                data.push(if *ok {
                                    i8::try_from(*x).map_err(|_| narrow_err(*x))?
                                } else {
                                    0
                                });
                            }
                            (pa.call_method0("int8")?, cast_bytes(&data, 1))
                        }
                        _ => {
                            let data: Vec<i64> = v.iter().map(|(_, x)| *x).collect();
                            (pa.call_method0("int64")?, cast_bytes(&data, 8))
                        }
                    };
                    (
                        dtype,
                        vb,
                        vec![py_buffer.call1((PyBytes::new(py, &raw),))?.unbind()],
                    )
                }
                // x86-64 little-endian i128 IS arrow's decimal128 layout,
                // so the payload slice goes straight into the buffer — the
                // same `cast_bytes` shape the i64 lane uses.
                OutCol::Dec(v) => {
                    let vb = bitmap(v.iter().map(|(ok, _)| *ok), n);
                    let data: Vec<i128> = v.iter().map(|(_, x)| *x).collect();
                    (
                        pa_ty(&pa, c.ty.ty)?,
                        vb,
                        vec![py_buffer
                            .call1((PyBytes::new(py, &cast_bytes(&data, 16)),))?
                            .unbind()],
                    )
                }
                OutCol::F64(v) => {
                    let vb = bitmap(v.iter().map(|(ok, _)| *ok), n);
                    let data: Vec<f64> = v.iter().map(|(_, x)| *x).collect();
                    let raw = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
                    };
                    (
                        pa.call_method0("float64")?,
                        vb,
                        vec![py_buffer.call1((PyBytes::new(py, raw),))?.unbind()],
                    )
                }
                OutCol::I1(v) => {
                    let vb = bitmap(v.iter().map(|(ok, _)| *ok), n);
                    let (data_bits, _) = bitmap(v.iter().map(|(_, x)| *x), n);
                    (
                        pa.call_method0("bool_")?,
                        vb,
                        vec![py_buffer
                            .call1((PyBytes::new(py, &data_bits),))?
                            .unbind()],
                    )
                }
                // `pa.string()`, 32-bit offsets, because that is what
                // DuckDB's own `.arrow()` returns for a VARCHAR column
                // (TASK-72). Emitting `large_string` gave byte-identical
                // VALUES under a schema that would not stack:
                // `pa.concat_tables([duck_out, ours])` raised, and so did any
                // pinned-schema writer.
                OutCol::Str(v) => {
                    let vb = bitmap(v.iter().map(|(ok, _)| *ok), n);
                    let mut offsets: Vec<i32> = Vec::with_capacity(n + 1);
                    let mut bytes: Vec<u8> = Vec::new();
                    offsets.push(0);
                    for (ok, r) in v.iter() {
                        if *ok {
                            bytes.extend_from_slice(st.arena.get(*r).as_bytes());
                        }
                        // 32-bit offsets are the whole point, so the 2 GiB
                        // ceiling is real. Refuse by name rather than wrap:
                        // DuckDB splits such a result across record batches,
                        // and we emit a single chunk.
                        let end = i32::try_from(bytes.len()).map_err(|_| {
                            err(
                                "infer_arrow: string column exceeds 2 GiB in one \
                                 batch — split the batch",
                            )
                        })?;
                        offsets.push(end);
                    }
                    let off_raw = unsafe {
                        std::slice::from_raw_parts(
                            offsets.as_ptr() as *const u8,
                            offsets.len() * 4,
                        )
                    };
                    (
                        pa.call_method0("string")?,
                        vb,
                        vec![
                            py_buffer.call1((PyBytes::new(py, off_raw),))?.unbind(),
                            py_buffer.call1((PyBytes::new(py, &bytes),))?.unbind(),
                        ],
                    )
                }
            };
        let (vbits, nulls) = validity;
        let vbuf: Py<PyAny> = if nulls == 0 {
            py.None()
        } else {
            py_buffer.call1((PyBytes::new(py, &vbits),))?.unbind()
        };
        let buf_list = PyList::new(py, std::iter::once(vbuf).chain(bufs))?;
        let arr = from_buffers.call((dtype, n, buf_list, nulls), None)?;
        arrays.push(arr);
    }
    let table = pa
        .getattr("Table")?
        .call_method1("from_arrays", (arrays, names))?;
    Ok(table.unbind())
}
