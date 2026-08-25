//! Execution substrate shared by the interpreter backend (and, at
//! M-cranelift, the codegen backend): batches, the bump arena, prepared
//! static structures, and the run-state buffers. Everything here obeys the
//! Stage-2 discipline: the only growable memory is the arena and the
//! pre-reserved output builders, both owned by [`RunState`] and reused
//! across calls — see [`RunState`]'s docs for the precise zero-allocation
//! contract.
//!
//! Lives under `exec/` rather than a separate `runtime/` until the Cranelift
//! backend actually shares it — one home per concept until two consumers
//! exist.

pub mod casemap;
pub mod cranelift;
pub mod interp;
pub mod kernels;
mod pow10;
mod strip_accents;
pub mod tree_ensemble;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub mod testutil;

use super::ir::Ty;

/// A runtime error that aborts the whole call (division by zero, integer
/// overflow, CAST failure routed to `trap`, input-shape mismatch).
/// Constructing one allocates — that is fine, it is the error path.
#[derive(Debug, PartialEq, Eq)]
pub struct Trap(pub String);

impl std::fmt::Display for Trap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "trap: {}", self.0)
    }
}

/// Columnar input batch. `valid` is meaningful only for nullable columns —
/// for non-nullable columns the interpreter ignores it entirely. The payload
/// slot of an invalid row may hold anything; readers normalize it to the
/// type's default so downstream behavior never depends on garbage.
pub struct Batch {
    pub rows: usize,
    pub cols: Vec<ColData>,
}

pub enum ColData {
    I1 {
        valid: Vec<bool>,
        data: Vec<bool>,
    },
    I64 {
        valid: Vec<bool>,
        data: Vec<i64>,
    },
    F64 {
        valid: Vec<bool>,
        data: Vec<f64>,
    },
    /// Strings live flat in one shared buffer with per-row spans — no
    /// per-cell String, so a reused column refills without allocating once
    /// capacity is warm (the marshaller's zero-alloc contract).
    Str {
        valid: Vec<bool>,
        buf: String,
        spans: Vec<StrRef>,
    },
}

impl ColData {
    /// Empty column of type `t`, ready for `push_*` fills.
    pub fn new(t: Ty) -> ColData {
        match t {
            Ty::I1 => ColData::I1 {
                valid: Vec::new(),
                data: Vec::new(),
            },
            // Narrow widths live in headers only; payloads are the i64 lane.
            Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => ColData::I64 {
                valid: Vec::new(),
                data: Vec::new(),
            },
            Ty::F64 => ColData::F64 {
                valid: Vec::new(),
                data: Vec::new(),
            },
            Ty::Str => ColData::Str {
                valid: Vec::new(),
                buf: String::new(),
                spans: Vec::new(),
            },
            // Input columns only. A decimal ROW column is outside the row
            // schema vocabulary and stays opaque (schema.rs, `Policy::Row`),
            // so a Dec never reaches an INPUT lane — only a static's value
            // lane and the output boundary, neither of which is ColData.
            Ty::Dec(..) => unreachable!("a decimal row column is opaque, never a ColData"),
        }
    }

    /// Capacity-preserving clear, for reuse across calls.
    pub fn clear(&mut self) {
        match self {
            ColData::I1 { valid, data } => {
                valid.clear();
                data.clear();
            }
            ColData::I64 { valid, data } => {
                valid.clear();
                data.clear();
            }
            ColData::F64 { valid, data } => {
                valid.clear();
                data.clear();
            }
            ColData::Str { valid, buf, spans } => {
                valid.clear();
                buf.clear();
                spans.clear();
            }
        }
    }

    /// Append one non-NULL cell to an I1 column — the struct-node PRESENCE
    /// lanes both row boundaries fill (TASK-133).
    pub fn push_present(&mut self, v: bool) {
        let ColData::I1 { valid, data } = self else {
            unreachable!("a presence lane is always I1");
        };
        valid.push(true);
        data.push(v);
    }

    /// Append one cell to a Str column (`""` for a NULL payload).
    pub fn push_str_cell(&mut self, ok: bool, s: &str) {
        let ColData::Str { valid, buf, spans } = self else {
            unreachable!("push_str_cell on a non-Str column");
        };
        valid.push(ok);
        let off = buf.len();
        buf.push_str(s);
        spans.push(StrRef { off, len: s.len() });
    }

    /// The string payload of row `row` of a Str column.
    pub fn str_at(&self, row: usize) -> &str {
        let ColData::Str { buf, spans, .. } = self else {
            unreachable!("str_at on a non-Str column");
        };
        let StrRef { off, len } = spans[row];
        &buf[off..off + len]
    }

    pub fn ty(&self) -> Ty {
        match self {
            ColData::I1 { .. } => Ty::I1,
            ColData::I64 { .. } => Ty::I64,
            ColData::F64 { .. } => Ty::F64,
            ColData::Str { .. } => Ty::Str,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ColData::I1 { data, .. } => data.len(),
            ColData::I64 { data, .. } => data.len(),
            ColData::F64 { data, .. } => data.len(),
            ColData::Str { spans, .. } => spans.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A span into the run arena. All `str`-typed register values are arena
/// spans — input/const/static strings are copied in on read, so registers
/// stay `Copy` and nothing borrows across rows. Offsets are usize: a single
/// call can legitimately accumulate more than 4 GiB of varlen data, and a
/// u32 span silently aliased old data past the wrap (adversarial finding).
#[derive(Clone, Copy, Debug)]
pub struct StrRef {
    pub off: usize,
    pub len: usize,
}

/// Bump arena for varlen values, reset per call. Byte-backed so
/// `extend_from_within` (sconcat) works without intermediate allocation;
/// only whole UTF-8 strings are ever appended, so spans are always valid
/// UTF-8.
#[derive(Default)]
pub struct Arena(pub Vec<u8>);

impl Arena {
    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn push_str(&mut self, s: &str) -> StrRef {
        let off = self.0.len();
        self.0.extend_from_slice(s.as_bytes());
        StrRef { off, len: s.len() }
    }

    pub fn concat(&mut self, a: StrRef, b: StrRef) -> StrRef {
        let off = self.0.len();
        self.0.extend_from_within(a.off..a.off + a.len);
        self.0.extend_from_within(b.off..b.off + b.len);
        StrRef {
            off,
            len: a.len + b.len,
        }
    }

    pub fn get(&self, r: StrRef) -> &str {
        // Spans only ever come from push_str/concat of whole &strs, so this
        // never fails; checked conversion keeps the module unsafe-free.
        std::str::from_utf8(&self.0[r.off..r.off + r.len])
            .expect("arena spans are always whole UTF-8 strings")
    }

    /// Format directly into the arena — no intermediate String.
    pub fn push_fmt(&mut self, args: std::fmt::Arguments<'_>) -> StrRef {
        use std::fmt::Write;
        let off = self.0.len();
        let _ = ArenaWriter(&mut self.0).write_fmt(args);
        StrRef {
            off,
            len: self.0.len() - off,
        }
    }

    /// Char-by-char 1:1 case map of `r` into a fresh span. Decodes one char
    /// at a time (width from the leading byte, O(1) validation) so no temp
    /// String is needed while the arena grows under the read span — offsets
    /// stay valid across reallocation where borrows would not.
    pub fn case_map(&mut self, r: StrRef, map: fn(char) -> char) -> StrRef {
        let off = self.0.len();
        let mut pos = r.off;
        let end = r.off + r.len;
        let mut buf = [0u8; 4];
        while pos < end {
            let w = match self.0[pos] {
                0x00..=0x7f => 1,
                0xc0..=0xdf => 2,
                0xe0..=0xef => 3,
                _ => 4,
            };
            let ch = std::str::from_utf8(&self.0[pos..pos + w])
                .expect("arena spans are always whole UTF-8 strings")
                .chars()
                .next()
                .expect("non-empty UTF-8 sequence");
            pos += w;
            self.0
                .extend_from_slice(map(ch).encode_utf8(&mut buf).as_bytes());
        }
        StrRef {
            off,
            len: self.0.len() - off,
        }
    }
}

/// Byte sink over the arena so `write!` formats without intermediate
/// allocation.
struct ArenaWriter<'a>(&'a mut Vec<u8>);

impl std::fmt::Write for ArenaWriter<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

/// An owned scalar, used by static structures and test expectations.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarVal {
    I1(bool),
    I64(i64),
    F64(f64),
    Str(String),
    /// A DECIMAL as its SCALED integer plus the (p, s) it is scaled at.
    /// The (p, s) rides along because [`ScalarVal::ty`] is what prepare
    /// checks against the declared value type, and unlike the narrow ints
    /// a Dec's scale does not erase.
    Dec(i128, u8, u8),
}

impl ScalarVal {
    pub fn ty(&self) -> Ty {
        match self {
            ScalarVal::I1(_) => Ty::I1,
            ScalarVal::I64(_) => Ty::I64,
            ScalarVal::F64(_) => Ty::F64,
            ScalarVal::Str(_) => Ty::Str,
            ScalarVal::Dec(_, p, s) => Ty::Dec(*p, *s),
        }
    }
}

/// One component of a map-static key. F64 keys compare and match by
/// *canonical* bit pattern: compile rewrites build-side keys and the probe
/// canonicalizes the searched value with [`canon_f64_bits`], so `-0.0`
/// matches `0.0` and every NaN is one key class — which is what DuckDB's
/// `=` does for doubles (NaN equals itself there).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum KeyBits {
    I1(bool),
    I64(i64),
    F64(u64),
    Str(String),
}

impl KeyBits {
    pub fn ty(&self) -> Ty {
        match self {
            KeyBits::I1(_) => Ty::I1,
            KeyBits::I64(_) => Ty::I64,
            KeyBits::F64(_) => Ty::F64,
            KeyBits::Str(_) => Ty::Str,
        }
    }
}

/// DuckDB's DOUBLE comparison order (measured 1.5.5): IEEE except that NaN
/// equals NaN and sorts above everything (`nan > inf` is TRUE); zeros are
/// equal (`-0.0 = 0.0`). NOT Rust `total_cmp` (which orders -0.0 < 0.0).
pub fn duck_fcmp(x: f64, y: f64) -> std::cmp::Ordering {
    match (x.is_nan(), y.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => x.partial_cmp(&y).expect("no NaN on either side"),
    }
}

/// The canonical key bits of an f64: all NaNs collapse to the one Rust
/// `f64::NAN` payload, `-0.0` collapses to `+0.0`. Everything else is
/// already unique per bit pattern.
pub fn canon_f64_bits(x: f64) -> u64 {
    if x.is_nan() {
        f64::NAN.to_bits()
    } else if x == 0.0 {
        0f64.to_bits()
    } else {
        x.to_bits()
    }
}

/// One extern (UDF) implementation, supplied to `compile_ext` alongside the
/// program — one per declared [`super::ir::ExternSpec`], name-checked.
///
/// The callable takes one `Option<ScalarVal>` per declared param (`None` =
/// NULL) and returns:
/// * `Ok(None)` — the whole call is NULL (every output NULL; at the width-k
///   output boundary this is the NULL list, distinct from a list of NULLs);
/// * `Ok(Some(vals))` — one `Option<ScalarVal>` per declared return;
/// * `Err(msg)` — trap the whole call (a raised Python exception).
///
/// Both backends route through [`interp::call_extern`], which enforces the
/// declared return shape (wrong length/type -> named trap) — the shared-code
/// rule that keeps the backends from drifting.
pub struct ExternImpl {
    pub name: String,
    #[allow(clippy::type_complexity)]
    pub fun: Box<dyn Fn(&[Option<ScalarVal>]) -> Result<Option<Vec<Option<ScalarVal>>>, String>>,
}

/// Runtime payload for a static structure, supplied to `compile` alongside
/// the program and type-checked against its `StaticTy` declaration.
pub enum StaticData {
    Scalar {
        valid: bool,
        val: ScalarVal,
    },
    /// Entries are sorted + deduped at compile into a probe table; the
    /// binary-search probe is allocation-free (design doc: how a map is
    /// materialized is a backend decision — the oracle picks the simplest
    /// correct structure, perfect hashing is the codegen backend's game).
    Map(Vec<(Vec<KeyBits>, Vec<ScalarVal>)>),
    /// A fitted ensemble, already structurally validated by
    /// [`tree_ensemble::TreeEnsemble::new`]; compile only checks that its
    /// width matches the declaration. Boxed so one model does not widen
    /// every static in the vector — statics are prepared once, so the
    /// indirection never shows up per row.
    Model(Box<tree_ensemble::TreeEnsemble>),
}

/// Output column builder: `(valid, value)` pairs; strings are arena spans.
pub enum OutCol {
    I1(Vec<(bool, bool)>),
    I64(Vec<(bool, i64)>),
    F64(Vec<(bool, f64)>),
    Str(Vec<(bool, StrRef)>),
    /// Scaled i128 payloads. Little-endian i128 IS arrow's decimal128
    /// layout on this target, so the emit writes the slice straight out.
    Dec(Vec<(bool, i128)>),
}

impl OutCol {
    fn clear(&mut self) {
        match self {
            OutCol::I1(v) => v.clear(),
            OutCol::I64(v) => v.clear(),
            OutCol::F64(v) => v.clear(),
            OutCol::Str(v) => v.clear(),
            OutCol::Dec(v) => v.clear(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            OutCol::I1(v) => v.len(),
            OutCol::I64(v) => v.len(),
            OutCol::F64(v) => v.len(),
            OutCol::Str(v) => v.len(),
            OutCol::Dec(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A register: always a bare scalar, mirroring the IR's type system.
#[derive(Clone, Copy)]
pub enum RegVal {
    I1(bool),
    I64(i64),
    F64(f64),
    Str(StrRef),
    /// A DECIMAL's scaled integer. The (p, s) is static — it lives in the
    /// program's types, not in the register — so only the payload rides
    /// here. The enum grows from 24 to 32 bytes; the register frame is
    /// per-CALL and tens of slots wide, so this is noise.
    Dec(i128),
}

/// Reusable per-call buffers: registers, arena, output builders. Create once
/// with [`interp::InterpFn::new_state`], reuse across calls — `run` clears
/// (capacity-preserving) and refills them.
///
/// Zero-allocation contract, stated precisely (the naive "after one warm
/// call" claim was refuted by adversarial testing): a run allocates only
/// when the input's *arena footprint* — branch paths taken, probe hits,
/// non-NULL varlen values, output row count — exceeds every previous run's
/// high-water mark. Such growth is one-time and monotone: repeating any
/// content profile already seen allocates nothing. Traps may allocate
/// (error path).
pub struct RunState {
    pub regs: Vec<RegVal>,
    pub arena: Arena,
    pub out: Vec<OutCol>,
    /// Rows emitted by the last `run` — the row count even when the
    /// program has zero output columns.
    pub emitted: usize,
}
