//! Execution substrate shared by the interpreter backend (and, at
//! M-cranelift, the codegen backend): batches, the bump arena, prepared
//! static structures, and the run-state buffers. Everything here obeys the
//! Stage-2 discipline — steady-state execution allocates nothing on the
//! heap; the only growable memory is the arena and the pre-reserved output
//! builders, both owned by [`RunState`] and reused across calls.
//!
//! Lives under `exec/` rather than a separate `runtime/` until the Cranelift
//! backend actually shares it — one home per concept until two consumers
//! exist.

pub mod interp;

#[cfg(test)]
mod tests;

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
    I1 { valid: Vec<bool>, data: Vec<bool> },
    I64 { valid: Vec<bool>, data: Vec<i64> },
    F64 { valid: Vec<bool>, data: Vec<f64> },
    Str { valid: Vec<bool>, data: Vec<String> },
}

impl ColData {
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
            ColData::Str { data, .. } => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A span into the run arena. All `str`-typed register values are arena
/// spans — input/const/static strings are copied in on read, so registers
/// stay `Copy` and nothing borrows across rows.
#[derive(Clone, Copy, Debug)]
pub struct StrRef {
    pub off: u32,
    pub len: u32,
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
        let off = self.0.len() as u32;
        self.0.extend_from_slice(s.as_bytes());
        StrRef { off, len: s.len() as u32 }
    }

    pub fn concat(&mut self, a: StrRef, b: StrRef) -> StrRef {
        let off = self.0.len() as u32;
        self.0.extend_from_within(a.off as usize..(a.off + a.len) as usize);
        self.0.extend_from_within(b.off as usize..(b.off + b.len) as usize);
        StrRef { off, len: a.len + b.len }
    }

    pub fn get(&self, r: StrRef) -> &str {
        // Spans only ever come from push_str/concat of whole &strs, so this
        // never fails; checked conversion keeps the module unsafe-free.
        std::str::from_utf8(&self.0[r.off as usize..(r.off + r.len) as usize])
            .expect("arena spans are always whole UTF-8 strings")
    }
}

/// An owned scalar, used by static structures and test expectations.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarVal {
    I1(bool),
    I64(i64),
    F64(f64),
    Str(String),
}

impl ScalarVal {
    pub fn ty(&self) -> Ty {
        match self {
            ScalarVal::I1(_) => Ty::I1,
            ScalarVal::I64(_) => Ty::I64,
            ScalarVal::F64(_) => Ty::F64,
            ScalarVal::Str(_) => Ty::Str,
        }
    }
}

/// One component of a map-static key. F64 keys compare and match by bit
/// pattern (total, deterministic; NaN == NaN by bits) — an internal
/// convention of the prepared structure, not a SQL semantics statement.
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

/// Runtime payload for a static structure, supplied to `compile` alongside
/// the program and type-checked against its `StaticTy` declaration.
pub enum StaticData {
    Scalar { valid: bool, val: ScalarVal },
    /// Entries are sorted + deduped at compile into a probe table; the
    /// binary-search probe is allocation-free (design doc: how a map is
    /// materialized is a backend decision — the oracle picks the simplest
    /// correct structure, perfect hashing is the codegen backend's game).
    Map(Vec<(Vec<KeyBits>, Vec<ScalarVal>)>),
}

/// Output column builder: `(valid, value)` pairs; strings are arena spans.
pub enum OutCol {
    I1(Vec<(bool, bool)>),
    I64(Vec<(bool, i64)>),
    F64(Vec<(bool, f64)>),
    Str(Vec<(bool, StrRef)>),
}

impl OutCol {
    fn clear(&mut self) {
        match self {
            OutCol::I1(v) => v.clear(),
            OutCol::I64(v) => v.clear(),
            OutCol::F64(v) => v.clear(),
            OutCol::Str(v) => v.clear(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            OutCol::I1(v) => v.len(),
            OutCol::I64(v) => v.len(),
            OutCol::F64(v) => v.len(),
            OutCol::Str(v) => v.len(),
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
}

/// Reusable per-call buffers: registers, arena, output builders. Create once
/// with [`interp::InterpFn::new_state`], reuse across calls — `run` clears
/// (capacity-preserving) and refills them; after the first warm call the
/// steady state performs zero heap allocations.
pub struct RunState {
    pub regs: Vec<RegVal>,
    pub arena: Arena,
    pub out: Vec<OutCol>,
}
