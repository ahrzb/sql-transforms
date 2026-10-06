//! The columnar boundary: `infer_arrow(pa.Table) -> pa.Table`.
//!
//! Ingest walks pyarrow buffers directly (address + size via the Python
//! buffer API — no arrow-rs dependency) into the engine's `ColData` lanes;
//! emit builds `pa.Array.from_buffers` per OUTPUT COLUMN from rust-built
//! buffers. Zero per-value Python objects on either side, which avoids the
//! measured ~1.4 µs/row of boxing at 31-column width. Buffers are copied;
//! there are no zero-copy lanes.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

use crate::error::InterpError;
use crate::specializer::exec::tree_ensemble::{ModelRows, NodeRows, TreeEnsemble};
use crate::specializer::exec::{Batch, ColData, OutCol, RunState};
use crate::specializer::ir::{Col, Ty};
use crate::specializer::plan::{InputLane, LaneKind};

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
/// guarantee.
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
/// is the same "a raw address is not a Rust reference" mistake [`Unaligned`]
/// exists to prevent.
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

/// The leaf array for one lane, plus every struct ancestor along its path.
/// A path is the lane's SEGMENT list: `[name]` for a plain column — dots
/// included, a name is not a path — and the struct walk for a leaf lane.
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

/// pyarrow batch -> engine Batch, matching each lane by its PATH with strict
/// dtypes: the column's arrow type must be the declared one exactly
/// (int8/int16/int32/int64 / double / string|large_string / bool), never a
/// widening.
pub fn ingest<'py>(
    py: Python<'_>,
    batch: &Bound<'py, PyAny>,
    lanes: &[InputLane],
) -> PyResult<Batch> {
    let _ = py;
    let rows: usize = batch.call_method0("__len__")?.extract()?;
    let mut cols = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let (arr, parents) = walk_lane(batch, &lane.path, &lane.name)?;
        // A PRESENCE lane stops at a struct NODE and reads that node's own
        // validity, folded with every parent's — the same fold the leaf
        // lanes below use. Its arrow type is a struct, so the scalar dtype
        // check does not apply to it.
        let ct = match lane.kind {
            LaneKind::Present => {
                // The node must BE a struct: a struct with no leaf lanes has
                // no other lane whose walk would refuse a wrong column.
                let ty: String = arr.getattr("type")?.str()?.extract()?;
                if !ty.starts_with("struct<") {
                    return Err(err(format!(
                        "infer_arrow: column '{}' is {ty}, the schema declares a struct",
                        lane.name
                    )));
                }
                let raw = raw_array(arr)?;
                if raw.len != rows {
                    return Err(err(format!(
                        "infer_arrow: column '{}' has {} rows, the batch {rows}",
                        lane.name, raw.len
                    )));
                }
                let data: Vec<bool> = (0..rows)
                    .map(|r| parents.iter().all(|p| p.valid(r)) && raw.valid(r))
                    .collect();
                cols.push(ColData::I1 {
                    valid: vec![true; rows],
                    data,
                });
                continue;
            }
            LaneKind::Value(ct) => ct,
        };
        let dtype: String = arr.getattr("type")?.str()?.extract()?;
        // A DECIMAL lane takes any of the tiers DuckDB itself exports, at
        // exactly the declared (p, s).
        let dec_width = match ct.ty {
            Ty::Dec(p, s) if crate::schema::decimal_ps(&dtype) == Some((p, s)) => {
                match dtype.split('(').next() {
                    Some("decimal32") => Some(4),
                    Some("decimal64") => Some(8),
                    Some("decimal128" | "decimal") => Some(16),
                    _ => None,
                }
            }
            _ => None,
        };
        let ok = dec_width.is_some()
            || matches!(
            (ct.ty, dtype.as_str()),
            (Ty::I8, "int8")
                | (Ty::I16, "int16")
                | (Ty::I32, "int32")
                | (Ty::I64, "int64")
                | (Ty::U8, "uint8")
                | (Ty::U16, "uint16")
                | (Ty::U32, "uint32")
                | (Ty::U64, "uint64")
                | (Ty::F64, "double")
                | (Ty::I1, "bool")
                | (Ty::Str, "string")
                | (Ty::Str, "large_string")
        );
        if !ok {
            return Err(err(format!(
                "infer_arrow: column '{}' is {dtype}, the schema declares {} — cast first",
                lane.name,
                ct.ty.name()
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
                lane.name, raw.len
            )));
        }
        let mut null_seen = false;
        let col_data = match ct.ty {
            // Narrow widths widen into the engine's i64 lane; the declared
            // arrow type already proved every value fits.
            Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::U8 | Ty::U16 | Ty::U32 => {
                let mut valid = Vec::with_capacity(rows);
                let mut data = Vec::with_capacity(rows);
                for i in 0..rows {
                    let v = valid_at(i);
                    null_seen |= !v;
                    valid.push(v);
                    data.push(if v {
                        match ct.ty {
                            Ty::I8 => unsafe { raw.data::<i8>(1).get(i) as i64 },
                            Ty::I16 => unsafe { raw.data::<i16>(1).get(i) as i64 },
                            Ty::I32 => unsafe { raw.data::<i32>(1).get(i) as i64 },
                            Ty::U8 => unsafe { raw.data::<u8>(1).get(i) as i64 },
                            Ty::U16 => unsafe { raw.data::<u16>(1).get(i) as i64 },
                            Ty::U32 => unsafe { raw.data::<u32>(1).get(i) as i64 },
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
                let mut col = super::col_for_lane(lane, rows);
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
            Ty::Dec(p, s) => {
                let mut valid = Vec::with_capacity(rows);
                let mut data = Vec::with_capacity(rows);
                for i in 0..rows {
                    let v = valid_at(i);
                    null_seen |= !v;
                    valid.push(v);
                    data.push(if !v {
                        0
                    } else {
                        // Little-endian two's complement, as Arrow lays it
                        // out at every width.
                        match dec_width {
                            Some(4) => unsafe { raw.data::<i32>(1).get(i) as i128 },
                            Some(8) => unsafe { raw.data::<i64>(1).get(i) as i128 },
                            _ => unsafe { raw.data::<i128>(1).get(i) },
                        }
                    });
                }
                ColData::Dec { p, s, valid, data }
            }
            // UBIGINT widens into the i128 lane. No arrow type reads as
            // HUGEINT (decimal128(38, 0) is a DECIMAL), so the check above
            // has already refused one.
            Ty::U64 | Ty::I128 => {
                let mut valid = Vec::with_capacity(rows);
                let mut data = Vec::with_capacity(rows);
                for i in 0..rows {
                    let v = valid_at(i);
                    null_seen |= !v;
                    valid.push(v);
                    data.push(if v { unsafe { raw.data::<u64>(1).get(i) as i128 } } else { 0 });
                }
                ColData::I128 { valid, data }
            }
        };
        if null_seen && !ct.nullable {
            return Err(err(format!(
                "column '{}' is not nullable but the batch has NULLs",
                lane.name
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
/// pickle, no live model reference. Column layout:
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

/// Arrow's validity encoding for `n` flags: LSB-first bits (set = valid),
/// plus the NULL count `Array.from_buffers` takes separately. `oks` yields
/// exactly `n` items — it is the same row range the buffer covers.
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
/// `size` must be `size_of::<T>()`: it is what turns the element count into
/// the byte length, so a larger value reads past the end of `data`.
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
        Ty::U8 => pa.call_method0("uint8"),
        Ty::U16 => pa.call_method0("uint16"),
        Ty::U32 => pa.call_method0("uint32"),
        Ty::F64 => pa.call_method0("float64"),
        Ty::Str => pa.call_method0("string"),
        // decimal128 at EVERY tier, because that is what DuckDB exports:
        // `SetArrowFormat` writes bit_width 128 regardless of the internal
        // int16/int32/int64/int128 storage under the default
        // ArrowFormatVersion::V1_0 (arrow_converter.cpp:236-262).
        Ty::Dec(p, s) => pa.call_method1("decimal128", (p, s)),
        // DuckDB exports HUGEINT as decimal128(38, 0) whatever the value,
        // past 38 digits too (measured: 2^127 - 1 comes back as a 39-digit
        // Decimal), and UBIGINT as uint64. Such an array fails pyarrow's
        // `validate(full=True)`, on DuckDB's export and here alike.
        Ty::I128 => pa.call_method1("decimal128", (38, 0)),
        Ty::U64 => pa.call_method0("uint64"),
    }
}

/// A wide node's arrow type: `list_` of its element type, or a `struct`
/// keyed by its field names, nested as the node is.
fn node_type<'py>(
    pa: &Bound<'py, PyModule>,
    out_cols: &[Col],
    node: &super::WideNode,
) -> PyResult<Bound<'py, PyAny>> {
    match node {
        super::WideNode::Lane(l, _) => pa_ty(pa, out_cols[*l].ty.ty),
        // Every element lane carries the list's one element type.
        super::WideNode::List { items, .. } => {
            let first = items
                .first()
                .ok_or_else(|| err("internal: a list field has no element lanes"))?;
            pa.call_method1("list_", (node_type(pa, out_cols, first)?,))
        }
        super::WideNode::Struct { fields, .. } => {
            let members = fields
                .iter()
                .map(|(n, f)| pa.call_method1("field", (n.as_str(), node_type(pa, out_cols, f)?)))
                .collect::<PyResult<Vec<_>>>()?;
            pa.call_method1("struct", (PyList::new(pa.py(), members)?,))
        }
    }
}

/// The output contract as a `pa.Schema` — field for field what [`emit`]'s
/// `from_arrays` table carries (same declared widths, `list_`/`struct` for
/// wide fields, default-nullable like DuckDB's own `.arrow()` export).
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
                (c.name.as_str(), pa_ty(&pa, c.ty.ty)?)
            }
            super::EmitField::Wide { name, node } => {
                (name.as_str(), node_type(&pa, out_cols, node)?)
            }
        };
        fields.push(pa.call_method1("field", (name, ty))?);
    }
    Ok(pa
        .call_method1("schema", (PyList::new(py, fields)?,))?
        .unbind())
}

/// Engine output -> pa.Table. Each scalar field, and each struct with its
/// children, is `Array.from_buffers` over rust-built buffers; a list field
/// assembles as `pa.array` of `None | [elements]` (the python-list path —
/// lists are transformer output, not the scalar hot lane).
pub fn emit(
    py: Python<'_>,
    out_cols: &[Col],
    plan: &[super::EmitField],
    st: &RunState,
) -> PyResult<Py<PyAny>> {
    let pa = PyModule::import(py, "pyarrow")?;
    let e = Emitter {
        py,
        from_buffers: pa.getattr("Array")?.getattr("from_buffers")?,
        py_buffer: pa.getattr("py_buffer")?,
        pa,
        out_cols,
        st,
        n: st.emitted,
        tys: out_cols.iter().map(|c| c.ty.ty).collect(),
    };
    let mut arrays = Vec::with_capacity(plan.len());
    let mut names = Vec::with_capacity(plan.len());
    for field in plan {
        match field {
            super::EmitField::Scalar(i) => {
                let c = &out_cols[*i];
                names.push(c.name.clone());
                arrays.push(e.lane(*i, &c.name, None)?);
            }
            super::EmitField::Wide { name, node } => {
                names.push(name.clone());
                arrays.push(e.node(node, None)?);
            }
        }
    }
    let table = e
        .pa
        .getattr("Table")?
        .call_method1("from_arrays", (arrays, names))?;
    Ok(table.unbind())
}

/// What [`emit`] builds each array from.
struct Emitter<'a, 'py> {
    py: Python<'py>,
    pa: Bound<'py, PyModule>,
    from_buffers: Bound<'py, PyAny>,
    py_buffer: Bound<'py, PyAny>,
    out_cols: &'a [Col],
    st: &'a RunState,
    n: usize,
    /// The out lanes' declared types, for [`super::wide_py`].
    tys: Vec<Ty>,
}

/// The i64 lane's values at the declared width `T`. A live value out of
/// range refuses by name: every such input DuckDB itself traps on.
fn narrowed<T: TryFrom<i64> + Default>(
    v: &[(bool, i64)],
    live: &[bool],
    refuse: impl Fn(i64) -> PyErr,
) -> PyResult<Vec<T>> {
    v.iter()
        .zip(live)
        .map(|((_, x), l)| {
            if *l {
                T::try_from(*x).map_err(|_| refuse(*x))
            } else {
                Ok(T::default())
            }
        })
        .collect()
}

impl<'py> Emitter<'_, 'py> {
    fn buffer(&self, raw: &[u8]) -> PyResult<Py<PyAny>> {
        Ok(self.py_buffer.call1((PyBytes::new(self.py, raw),))?.unbind())
    }

    /// `Array.from_buffers` with `live` as the validity bitmap, then `bufs`
    /// (and `children`, for a struct).
    fn array(
        &self,
        dtype: Bound<'py, PyAny>,
        live: &[bool],
        bufs: Vec<Py<PyAny>>,
        children: Option<Vec<Bound<'py, PyAny>>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (vbits, nulls) = bitmap(live.iter().copied(), self.n);
        let vbuf: Py<PyAny> = if nulls == 0 {
            self.py.None()
        } else {
            self.buffer(&vbits)?
        };
        let buf_list = PyList::new(self.py, std::iter::once(vbuf).chain(bufs))?;
        match children {
            None => self.from_buffers.call1((dtype, self.n, buf_list, nulls)),
            Some(ch) => self.from_buffers.call1((
                dtype,
                self.n,
                buf_list,
                nulls,
                0,
                PyList::new(self.py, ch)?,
            )),
        }
    }

    /// Out lane `i` as an arrow array. `name` is what a range refusal
    /// names. `mask`, when given, NULLs each row where an enclosing struct
    /// is NULL and skips that row's range check: a child is NULL wherever
    /// its parent is, as in DuckDB's own export, and a value no reader can
    /// reach never refuses (the row boundary does not look at it either).
    fn lane(&self, i: usize, name: &str, mask: Option<&[bool]>) -> PyResult<Bound<'py, PyAny>> {
        let (n, pa) = (self.n, &self.pa);
        let ty = self.out_cols[i].ty.ty;
        let live_of = |oks: &mut dyn Iterator<Item = bool>| -> Vec<bool> {
            oks.enumerate()
                .map(|(r, ok)| ok && mask.is_none_or(|m| m[r]))
                .collect()
        };
        let (dtype, live, bufs): (Bound<'_, PyAny>, Vec<bool>, Vec<Py<PyAny>>) =
            match &self.st.out[i] {
                OutCol::I64(v) => {
                    // The column's declared width narrows the emitted buffer
                    // (values compute in the i64 lane).
                    let live = live_of(&mut v.iter().map(|(ok, _)| *ok));
                    let refuse = |x: i64| {
                        err(format!(
                            "infer_arrow: column '{name}' value {x} is outside its {} range",
                            super::arrow_ty_name(ty),
                        ))
                    };
                    let (dtype, raw): (_, Vec<u8>) = match ty {
                        Ty::I32 => (
                            pa.call_method0("int32")?,
                            cast_bytes(&narrowed::<i32>(v, &live, refuse)?, 4),
                        ),
                        Ty::I16 => (
                            pa.call_method0("int16")?,
                            cast_bytes(&narrowed::<i16>(v, &live, refuse)?, 2),
                        ),
                        Ty::I8 => (
                            pa.call_method0("int8")?,
                            cast_bytes(&narrowed::<i8>(v, &live, refuse)?, 1),
                        ),
                        Ty::U32 => (
                            pa.call_method0("uint32")?,
                            cast_bytes(&narrowed::<u32>(v, &live, refuse)?, 4),
                        ),
                        Ty::U16 => (
                            pa.call_method0("uint16")?,
                            cast_bytes(&narrowed::<u16>(v, &live, refuse)?, 2),
                        ),
                        Ty::U8 => (
                            pa.call_method0("uint8")?,
                            cast_bytes(&narrowed::<u8>(v, &live, refuse)?, 1),
                        ),
                        _ => {
                            let data: Vec<i64> = v.iter().map(|(_, x)| *x).collect();
                            (pa.call_method0("int64")?, cast_bytes(&data, 8))
                        }
                    };
                    (dtype, live, vec![self.buffer(&raw)?])
                }
                // x86-64 little-endian i128 IS arrow's decimal128 layout,
                // so the payload slice goes straight into the buffer — the
                // same `cast_bytes` shape the i64 lane uses.
                // UBIGINT computes in the i128 lane and emits as uint64; the
                // lowering's range trap already proved every value fits.
                OutCol::Dec(v) if ty == Ty::U64 => {
                    let live = live_of(&mut v.iter().map(|(ok, _)| *ok));
                    let data: Vec<u64> = v.iter().map(|(_, x)| *x as u64).collect();
                    (pa_ty(pa, Ty::U64)?, live, vec![self.buffer(&cast_bytes(&data, 8))?])
                }
                OutCol::Dec(v) => {
                    let live = live_of(&mut v.iter().map(|(ok, _)| *ok));
                    let data: Vec<i128> = v.iter().map(|(_, x)| *x).collect();
                    (pa_ty(pa, ty)?, live, vec![self.buffer(&cast_bytes(&data, 16))?])
                }
                OutCol::F64(v) => {
                    let live = live_of(&mut v.iter().map(|(ok, _)| *ok));
                    let data: Vec<f64> = v.iter().map(|(_, x)| *x).collect();
                    (
                        pa.call_method0("float64")?,
                        live,
                        vec![self.buffer(&cast_bytes(&data, 8))?],
                    )
                }
                OutCol::I1(v) => {
                    let live = live_of(&mut v.iter().map(|(ok, _)| *ok));
                    let (data_bits, _) = bitmap(v.iter().map(|(_, x)| *x), n);
                    (pa.call_method0("bool_")?, live, vec![self.buffer(&data_bits)?])
                }
                // `pa.string()`, 32-bit offsets, because that is what
                // DuckDB's own `.arrow()` returns for a VARCHAR column.
                // Emitting `large_string` gave byte-identical VALUES under
                // a schema that would not stack:
                // `pa.concat_tables([duck_out, ours])` raised, and so did any
                // pinned-schema writer.
                OutCol::Str(v) => {
                    let live = live_of(&mut v.iter().map(|(ok, _)| *ok));
                    let mut offsets: Vec<i32> = Vec::with_capacity(n + 1);
                    let mut bytes: Vec<u8> = Vec::new();
                    offsets.push(0);
                    for ((_, s), l) in v.iter().zip(&live) {
                        if *l {
                            bytes.extend_from_slice(self.st.arena.get(*s).as_bytes());
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
                    (
                        pa.call_method0("string")?,
                        live,
                        vec![
                            self.buffer(&cast_bytes(&offsets, 4))?,
                            self.buffer(&bytes)?,
                        ],
                    )
                }
            };
        self.array(dtype, &live, bufs, None)
    }

    /// A field's array: a lane as above; a struct over its children's
    /// arrays, each built the same way and NULL wherever the struct is;
    /// a list through the python-list path.
    fn node(&self, node: &super::WideNode, mask: Option<&[bool]>) -> PyResult<Bound<'py, PyAny>> {
        match node {
            super::WideNode::Lane(l, at) => self.lane(*l, at, mask),
            super::WideNode::Struct { valid, fields } => {
                let here = (0..self.n)
                    .map(|r| Ok(super::valid_at(self.st, *valid, r)? && mask.is_none_or(|m| m[r])))
                    .collect::<PyResult<Vec<bool>>>()?;
                let children = fields
                    .iter()
                    .map(|(_, f)| self.node(f, Some(&here)))
                    .collect::<PyResult<Vec<_>>>()?;
                let dtype = node_type(&self.pa, self.out_cols, node)?;
                self.array(dtype, &here, Vec::new(), Some(children))
            }
            super::WideNode::List { items, .. } => self.list(node, items, mask),
        }
    }

    /// A list field as `pa.array` over each row's `None | [elements]`.
    fn list(
        &self,
        node: &super::WideNode,
        items: &[super::WideNode],
        mask: Option<&[bool]>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (py, pa) = (self.py, &self.pa);
        let mut values = Vec::with_capacity(self.n);
        for r in 0..self.n {
            values.push(if mask.is_none_or(|m| m[r]) {
                super::wide_py(py, self.st, node, &self.tys, r)?
            } else {
                py.None()
            });
        }
        let vals = PyList::new(py, values)?;
        let out_ty = node_type(pa, self.out_cols, node)?;
        let kw = pyo3::types::PyDict::new(py);
        // A HUGEINT element past 38 digits leaves as DuckDB exports it: the
        // raw i128 in a decimal128(38,0), which pa.array refuses to build
        // from a Decimal. Build at decimal256(39,0) and cast down unchecked,
        // keeping the raw value.
        let huge = items
            .iter()
            .any(|i| matches!(i, super::WideNode::Lane(l, _) if self.tys[*l] == Ty::I128));
        if huge {
            let wide_ty = pa.call_method1("list_", (pa.call_method1("decimal256", (39, 0))?,))?;
            kw.set_item("type", wide_ty)?;
            let built = pa.call_method("array", (vals,), Some(&kw))?;
            let cast_kw = pyo3::types::PyDict::new(py);
            cast_kw.set_item("safe", false)?;
            return built.call_method("cast", (out_ty,), Some(&cast_kw));
        }
        kw.set_item("type", out_ty)?;
        pa.call_method("array", (vals,), Some(&kw))
    }
}
