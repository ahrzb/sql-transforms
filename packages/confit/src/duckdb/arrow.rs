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

    /// Typed data slice starting at the array's offset.
    unsafe fn data<T: Copy>(&self, slot: usize) -> *const T {
        let (addr, _) = self.bufs[slot].expect("data buffer present");
        unsafe { (addr as *const T).add(self.offset) }
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

/// A single-chunk column by NAME from a Table or RecordBatch.
fn column<'py>(batch: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let schema = batch.getattr("schema")?;
    let idx: i64 = schema
        .call_method1("get_field_index", (name,))?
        .extract()?;
    if idx < 0 {
        return Err(err(format!(
            "infer_arrow: the batch is missing column '{name}'"
        )));
    }
    let col = batch.call_method1("column", (idx,))?;
    // Table columns are ChunkedArrays; RecordBatch columns are Arrays.
    if col.hasattr("num_chunks")? {
        let n: usize = col.getattr("num_chunks")?.extract()?;
        match n {
            1 => col.call_method1("chunk", (0,)),
            0 => col.call_method1("combine_chunks", ()),
            _ => Err(err(format!(
                "infer_arrow: column '{name}' has {n} chunks — call \
                 combine_chunks() on the table first"
            ))),
        }
    } else {
        Ok(col)
    }
}

/// pyarrow batch -> engine Batch, matching `in_cols` by name with strict
/// dtypes (int64 / double / string|large_string / bool).
pub fn ingest(py: Python<'_>, batch: &Bound<'_, PyAny>, in_cols: &[Col]) -> PyResult<Batch> {
    let _ = py;
    if in_cols.iter().any(|c| c.name.contains('.')) {
        return Err(err(
            "infer_arrow requires an all-scalar row model (struct columns: use infer())",
        ));
    }
    let rows: usize = batch.call_method0("__len__")?.extract()?;
    let mut cols = Vec::with_capacity(in_cols.len());
    for c in in_cols {
        let arr = column(batch, &c.name)?;
        let dtype: String = arr.getattr("type")?.str()?.extract()?;
        let ok = matches!(
            (c.ty.ty, dtype.as_str()),
            (Ty::I64, "int64")
                | (Ty::F64, "double")
                | (Ty::I1, "bool")
                | (Ty::Str, "string")
                | (Ty::Str, "large_string")
        );
        if !ok {
            return Err(err(format!(
                "infer_arrow: column '{}' is {dtype}, the model wants {} — cast first",
                c.name,
                c.ty.ty.name()
            )));
        }
        let large = dtype == "large_string";
        let raw = raw_array(arr)?;
        if raw.len != rows {
            return Err(err(format!(
                "infer_arrow: column '{}' has {} rows, the batch {rows}",
                c.name, raw.len
            )));
        }
        let mut null_seen = false;
        let col_data = match c.ty.ty {
            Ty::I64 => {
                let mut valid = Vec::with_capacity(rows);
                let mut data = Vec::with_capacity(rows);
                let p = unsafe { raw.data::<i64>(1) };
                for i in 0..rows {
                    let v = raw.valid(i);
                    null_seen |= !v;
                    valid.push(v);
                    data.push(if v { unsafe { *p.add(i) } } else { 0 });
                }
                ColData::I64 { valid, data }
            }
            Ty::F64 => {
                let mut valid = Vec::with_capacity(rows);
                let mut data = Vec::with_capacity(rows);
                let p = unsafe { raw.data::<f64>(1) };
                for i in 0..rows {
                    let v = raw.valid(i);
                    null_seen |= !v;
                    valid.push(v);
                    data.push(if v { unsafe { *p.add(i) } } else { 0.0 });
                }
                ColData::F64 { valid, data }
            }
            Ty::I1 => {
                let mut valid = Vec::with_capacity(rows);
                let mut data = Vec::with_capacity(rows);
                let (addr, _) = raw.bufs[1].expect("bool data buffer");
                for i in 0..rows {
                    let v = raw.valid(i);
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
                let (daddr, dsize) = raw.bufs[2].unwrap_or((0, 0));
                let bytes =
                    unsafe { std::slice::from_raw_parts(daddr as *const u8, dsize) };
                for i in 0..rows {
                    let v = raw.valid(i);
                    null_seen |= !v;
                    if !v {
                        col.push_str_cell(false, "");
                        continue;
                    }
                    let (lo, hi) = if large {
                        let p = unsafe { raw.data::<i64>(1) };
                        unsafe { (*p.add(i) as usize, *p.add(i + 1) as usize) }
                    } else {
                        let p = unsafe { raw.data::<i32>(1) };
                        unsafe { (*p.add(i) as usize, *p.add(i + 1) as usize) }
                    };
                    let s = std::str::from_utf8(&bytes[lo..hi])
                        .map_err(|_| err("infer_arrow: invalid UTF-8 in string column"))?;
                    col.push_str_cell(true, s);
                }
                col
            }
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

/// Engine output -> pa.Table, one `Array.from_buffers` per column.
pub fn emit(py: Python<'_>, out_cols: &[Col], st: &RunState) -> PyResult<Py<PyAny>> {
    let pa = PyModule::import(py, "pyarrow")?;
    let from_buffers = pa.getattr("Array")?.getattr("from_buffers")?;
    let py_buffer = pa.getattr("py_buffer")?;
    let n = st.emitted;
    let mut arrays = Vec::with_capacity(out_cols.len());
    let mut names = Vec::with_capacity(out_cols.len());
    for (c, oc) in out_cols.iter().zip(&st.out) {
        names.push(c.name.clone());
        let (dtype, validity, bufs): (Bound<'_, PyAny>, (Vec<u8>, usize), Vec<Py<PyAny>>) =
            match oc {
                OutCol::I64(v) => {
                    let vb = bitmap(v.iter().map(|(ok, _)| *ok), n);
                    let data: Vec<i64> = v.iter().map(|(_, x)| *x).collect();
                    let raw = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
                    };
                    (
                        pa.call_method0("int64")?,
                        vb,
                        vec![py_buffer.call1((PyBytes::new(py, raw),))?.unbind()],
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
                OutCol::Str(v) => {
                    let vb = bitmap(v.iter().map(|(ok, _)| *ok), n);
                    let mut offsets: Vec<i64> = Vec::with_capacity(n + 1);
                    let mut bytes: Vec<u8> = Vec::new();
                    offsets.push(0);
                    for (ok, r) in v.iter() {
                        if *ok {
                            bytes.extend_from_slice(st.arena.get(*r).as_bytes());
                        }
                        offsets.push(bytes.len() as i64);
                    }
                    let off_raw = unsafe {
                        std::slice::from_raw_parts(
                            offsets.as_ptr() as *const u8,
                            offsets.len() * 8,
                        )
                    };
                    (
                        pa.call_method0("large_string")?,
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
