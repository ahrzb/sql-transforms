//! The closure-compiled interpreter backend — the oracle. One pre-traversal
//! of a VERIFIED program builds a vector of instruction closures per block;
//! execution is plain dispatch. Never optimized: correctness and coverage
//! over speed, and it stays that way (design doc §7).
//!
//! # Semantics pins (provisional until the DuckDB differential at M-lower)
//!
//! * `iadd`/`isub`/`imul` trap on i64 overflow (SQL overflow is an error,
//!   not a wrap); `idiv`/`irem` trap on zero and on `i64::MIN / -1`.
//! * `f*` arithmetic is raw IEEE — inf/nan flow through, no traps.
//! * `fcmp` uses DuckDB's DOUBLE order (`exec::duck_fcmp`): IEEE except that
//!   NaN equals NaN and sorts above everything, and `-0.0 = 0.0`. SQL
//!   NULL/NaN policy is the lowering's job, expressed with flags around
//!   these primitives.
//! * `icmp` is i64 order; `scmp` is byte order (Rust `str` cmp).
//! * `itof` is `as f64` (may round — same as DuckDB's BIGINT->DOUBLE).
//! * `ftoi.trunc` rounds toward zero; `ftoi.round` half-away-from-zero
//!   (matches DuckDB CAST); both trap on non-finite or out-of-i64-range.
//! * `stoi.opt`/`stof.opt` trim ASCII whitespace then `str::parse` — pinned
//!   at M-lower against DuckDB CAST (measured: `' 5'::BIGINT = 5`); the
//!   empty/whitespace-only string fails the parse.
//! * `itos`/`ftos` format into the arena; `ftos` uses Rust's shortest
//!   round-trip form (provisional; oracle-pinned at M-lower).
//! * On a false validity flag the payload is the type default; `load.opt`
//!   normalizes even if the input batch carries garbage in invalid slots.

use std::collections::HashMap;

use super::super::ir::verify::{verify, VerifyError};
use super::super::ir::{
    self, BinOp, CmpPred, Inst, NumOp1, Program, RoundMode, StaticTy, StrOp1, StrOp2, Term,
    TrimSide, Ty, Value,
};
use super::{
    Arena, Batch, ColData, KeyBits, OutCol, RegVal, RunState, ScalarVal, StaticData, StrRef, Trap,
};

#[derive(Debug)]
pub enum CompileError {
    /// The program failed verification — nothing executes unverified IR.
    Verify(Vec<VerifyError>),
    /// The static data does not match the program's static declarations.
    Static(String),
    /// A ReSpec pattern failed to compile — the frontend validates patterns
    /// at bind, so this only fires on hand-written IR.
    Regex(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Verify(errs) => {
                write!(f, "program failed verification: ")?;
                for e in errs {
                    write!(f, "[{e}] ")?;
                }
                Ok(())
            }
            CompileError::Static(msg) => write!(f, "static data mismatch: {msg}"),
            CompileError::Regex(msg) => write!(f, "regex table entry failed to compile: {msg}"),
        }
    }
}

/// Compile every [`ir::ReSpec`] in the program with the wave-B builder
/// settings (octal(true), default Unicode — see retrans.rs).
pub(super) fn compile_regexes(p: &Program) -> Result<Vec<std::rc::Rc<regex::Regex>>, String> {
    p.regexes
        .iter()
        .map(|r| {
            regex::RegexBuilder::new(&r.pattern)
                .case_insensitive(r.ci)
                .dot_matches_new_line(r.dotall)
                .octal(true)
                .build()
                .map(std::rc::Rc::new)
                .map_err(|e| e.to_string())
        })
        .collect()
}

/// One compiled instruction: reads registers/input/statics, writes registers
/// or output builders. `Ok(())` or a call-aborting trap.
type InstFn = Box<dyn for<'a> Fn(&mut Ctx<'a>) -> Result<(), Trap>>;

/// Everything a closure can touch during one row.
struct Ctx<'a> {
    regs: &'a mut [RegVal],
    arena: &'a mut Arena,
    out: &'a mut [OutCol],
    input: &'a Batch,
    statics: &'a [PreparedStatic],
    /// Stage-B self-join: the batch's rows flattened like multimap values
    /// (nullable -> validity+payload). Empty unless the program declares a
    /// batchmap static.
    batch_rows: &'a [Vec<ScalarVal>],
    row: usize,
}

pub(super) enum PreparedStatic {
    Scalar {
        valid: bool,
        val: ScalarVal,
    },
    /// Sorted by key; probed by allocation-free binary search.
    Map {
        entries: Vec<(Vec<KeyBits>, Vec<ScalarVal>)>,
    },
    /// Stage-B: sorted by key with DUPLICATES ADJACENT; probe.range finds
    /// the equal-key run, probe.read indexes into it. Keyless multimaps
    /// (cross/inequality joins) range over the whole table.
    MultiMap {
        entries: Vec<(Vec<KeyBits>, Vec<ScalarVal>)>,
    },
    /// Stage-B self-join: the rows come from the BATCH, flattened per call
    /// in `run` (see `build_batch_rows`) — nothing is prepared here.
    BatchMap,
}

struct CBlock {
    insts: Vec<InstFn>,
    term: CTerm,
}

/// Terminators are a small enum interpreted by the row loop — no closure
/// indirection needed. Branch-arg copies are (src, dst) register pairs;
/// sources and destinations are disjoint by SSA single-definition, so
/// sequential copies are safe.
enum CTerm {
    Jump {
        to: usize,
        moves: Vec<(u32, u32)>,
    },
    Brif {
        cond: u32,
        then_to: usize,
        then_moves: Vec<(u32, u32)>,
        else_to: usize,
        else_moves: Vec<(u32, u32)>,
    },
    Emit,
    /// Emit the completed row and continue at `to` (multiplicity loops).
    EmitTo {
        to: usize,
        moves: Vec<(u32, u32)>,
    },
    Skip,
    Trap(String),
}

pub struct InterpFn {
    blocks: Vec<CBlock>,
    nregs: usize,
    statics: Vec<PreparedStatic>,
    in_decl: Vec<(Ty, bool)>,
    out_decl: Vec<Ty>,
    /// True when a batchmap static exists: `run` flattens the batch's
    /// rows before the row loop (stage-B self-joins).
    has_batch_map: bool,
}

pub fn compile(p: &Program, statics: Vec<StaticData>) -> Result<InterpFn, CompileError> {
    verify(p).map_err(CompileError::Verify)?;
    let prepared = prepare_statics(p, statics)?;
    let regexes = compile_regexes(p).map_err(CompileError::Regex)?;

    // Register slots are assigned densely in definition order, decoupling
    // the frame size from raw value ids — a verified program with sparse ids
    // (they are legal) must not force a huge register vector (adversarial
    // finding: Value(u32::MAX) would have demanded a 64 GiB frame).
    let mut slots: HashMap<u32, u32> = HashMap::new();
    for b in &p.blocks {
        for (v, _) in &b.params {
            let n = slots.len() as u32;
            slots.entry(v.0).or_insert(n);
        }
        for inst in &b.insts {
            for d in inst.dsts() {
                let n = slots.len() as u32;
                slots.entry(d.0).or_insert(n);
            }
        }
    }
    let nregs = slots.len();
    let mut blocks = Vec::with_capacity(p.blocks.len());
    for b in &p.blocks {
        let mut insts: Vec<InstFn> = Vec::with_capacity(b.insts.len());
        for inst in &b.insts {
            insts.push(compile_inst(p, inst, &slots, &regexes));
        }
        blocks.push(CBlock {
            insts,
            term: compile_term(p, &b.term, &slots),
        });
    }

    Ok(InterpFn {
        blocks,
        nregs,
        statics: prepared,
        in_decl: p.in_cols.iter().map(|c| (c.ty.ty, c.ty.nullable)).collect(),
        out_decl: p.out_cols.iter().map(|c| c.ty.ty).collect(),
        has_batch_map: p.statics.iter().any(|s| matches!(s, StaticTy::BatchMap { .. })),
    })
}

impl InterpFn {
    /// Fresh reusable buffers for `run`. Allocate once, reuse per call.
    pub fn new_state(&self) -> RunState {
        RunState {
            regs: vec![RegVal::I64(0); self.nregs],
            emitted: 0,
            arena: Arena::default(),
            out: self
                .out_decl
                .iter()
                .map(|ty| match ty {
                    Ty::I1 => OutCol::I1(Vec::new()),
                    Ty::I64 => OutCol::I64(Vec::new()),
                    Ty::F64 => OutCol::F64(Vec::new()),
                    Ty::Str => OutCol::Str(Vec::new()),
                })
                .collect(),
        }
    }

    /// Execute over `input`, filling `st.out` (cleared first, capacity kept).
    /// On `Err` the output is meaningless and the whole call is void.
    pub fn run(&self, input: &Batch, st: &mut RunState) -> Result<(), Trap> {
        self.check_input(input)?;
        self.check_state(st)?;
        st.arena.clear();
        st.emitted = 0;
        for col in st.out.iter_mut() {
            col.clear();
        }
        reserve_out(&mut st.out, input.rows);

        // Stage-B self-joins: flatten the batch ONCE per call into the
        // multimap-value layout (nullable -> validity+payload).
        let batch_rows: Vec<Vec<ScalarVal>> = if self.has_batch_map {
            build_batch_rows(input, &self.in_decl)
        } else {
            Vec::new()
        };
        let mut emitted = 0usize;
        for row in 0..input.rows {
            let mut ctx = Ctx {
                regs: &mut st.regs,
                arena: &mut st.arena,
                out: &mut st.out,
                input,
                statics: &self.statics,
                batch_rows: &batch_rows,
                row,
            };
            let mut bi = 0usize;
            loop {
                for f in &self.blocks[bi].insts {
                    f(&mut ctx)?;
                }
                match &self.blocks[bi].term {
                    CTerm::Jump { to, moves } => {
                        do_moves(ctx.regs, moves);
                        bi = *to;
                    }
                    CTerm::Brif {
                        cond,
                        then_to,
                        then_moves,
                        else_to,
                        else_moves,
                    } => {
                        if as_i1(ctx.regs[*cond as usize]) {
                            do_moves(ctx.regs, then_moves);
                            bi = *then_to;
                        } else {
                            do_moves(ctx.regs, else_moves);
                            bi = *else_to;
                        }
                    }
                    CTerm::Emit => {
                        emitted += 1;
                        break;
                    }
                    CTerm::EmitTo { to, moves } => {
                        emitted += 1;
                        do_moves(ctx.regs, moves);
                        bi = *to;
                    }
                    CTerm::Skip => break,
                    CTerm::Trap(msg) => return Err(Trap(msg.clone())),
                }
            }
        }
        st.emitted = emitted;
        Ok(())
    }

    /// The prepared statics, shared with the Cranelift backend's helpers.
    pub(super) fn statics(&self) -> &[PreparedStatic] {
        &self.statics
    }

    /// A `RunState` is only valid for the `InterpFn` that created it —
    /// reject a foreign one with a trap instead of an index panic.
    pub(super) fn check_state(&self, st: &RunState) -> Result<(), Trap> {
        if st.regs.len() < self.nregs {
            return Err(Trap(format!(
                "RunState has {} register(s), this function needs {} — states are not \
                 shareable across compiled functions",
                st.regs.len(),
                self.nregs
            )));
        }
        if st.out.len() != self.out_decl.len() {
            return Err(Trap(format!(
                "RunState has {} out column(s), this function declares {}",
                st.out.len(),
                self.out_decl.len()
            )));
        }
        for (ci, (col, ty)) in st.out.iter().zip(self.out_decl.iter()).enumerate() {
            let col_ty = match col {
                OutCol::I1(_) => Ty::I1,
                OutCol::I64(_) => Ty::I64,
                OutCol::F64(_) => Ty::F64,
                OutCol::Str(_) => Ty::Str,
            };
            if col_ty != *ty {
                return Err(Trap(format!(
                    "RunState out column {ci} is {}, this function declares {}",
                    col_ty.name(),
                    ty.name()
                )));
            }
        }
        Ok(())
    }

    pub(super) fn check_input(&self, input: &Batch) -> Result<(), Trap> {
        if input.cols.len() != self.in_decl.len() {
            return Err(Trap(format!(
                "input has {} column(s), the program declares {}",
                input.cols.len(),
                self.in_decl.len()
            )));
        }
        for (ci, (col, (ty, _))) in input.cols.iter().zip(self.in_decl.iter()).enumerate() {
            if col.ty() != *ty {
                return Err(Trap(format!(
                    "input column {ci} is {}, the program declares {}",
                    col.ty().name(),
                    ty.name()
                )));
            }
            if col.len() != input.rows {
                return Err(Trap(format!(
                    "input column {ci} has {} row(s), the batch declares {}",
                    col.len(),
                    input.rows
                )));
            }
            // For a nullable column the validity lane is part of the input
            // shape — a short vector must not silently mean "valid".
            let (_, nullable) = self.in_decl[ci];
            if nullable && valid_len(col) != input.rows {
                return Err(Trap(format!(
                    "input column {ci} is nullable but its validity vector has {} \
                     entries for {} row(s)",
                    valid_len(col),
                    input.rows
                )));
            }
        }
        Ok(())
    }
}

fn do_moves(regs: &mut [RegVal], moves: &[(u32, u32)]) {
    for (src, dst) in moves {
        regs[*dst as usize] = regs[*src as usize];
    }
}

pub(super) fn reserve_out(out: &mut [OutCol], rows: usize) {
    for col in out {
        match col {
            OutCol::I1(v) => v.reserve(rows),
            OutCol::I64(v) => v.reserve(rows),
            OutCol::F64(v) => v.reserve(rows),
            OutCol::Str(v) => v.reserve(rows),
        }
    }
}

/// Flatten the batch into multimap-value rows for a batchmap static:
/// per input row, each column contributes payload (non-nullable) or
/// validity + payload (nullable), in declaration order — the same layout
/// the lowering declared for the batchmap's values.
fn build_batch_rows(input: &Batch, in_decl: &[(Ty, bool)]) -> Vec<Vec<ScalarVal>> {
    let mut rows = Vec::with_capacity(input.rows);
    for r in 0..input.rows {
        let mut vals = Vec::new();
        for (ci, &(ty, nullable)) in in_decl.iter().enumerate() {
            let c = &input.cols[ci];
            let valid = col_valid(c, r);
            if nullable {
                vals.push(ScalarVal::I1(valid));
            }
            let v = if valid {
                match c {
                    ColData::I1 { data, .. } => ScalarVal::I1(data[r]),
                    ColData::I64 { data, .. } => ScalarVal::I64(data[r]),
                    ColData::F64 { data, .. } => ScalarVal::F64(data[r]),
                    ColData::Str { buf, spans, .. } => {
                        let sp = spans[r];
                        ScalarVal::Str(buf[sp.off as usize..(sp.off + sp.len) as usize].to_string())
                    }
                }
            } else {
                match ty {
                    Ty::I1 => ScalarVal::I1(false),
                    Ty::I64 => ScalarVal::I64(0),
                    Ty::F64 => ScalarVal::F64(0.0),
                    Ty::Str => ScalarVal::Str(String::new()),
                }
            };
            vals.push(v);
        }
        rows.push(vals);
    }
    rows
}

pub(super) fn prepare_statics(
    p: &Program,
    statics: Vec<StaticData>,
) -> Result<Vec<PreparedStatic>, CompileError> {
    if statics.len() != p.statics.len() {
        return Err(CompileError::Static(format!(
            "program declares {} static(s), {} provided",
            p.statics.len(),
            statics.len()
        )));
    }
    let mut prepared = Vec::with_capacity(statics.len());
    for (i, (decl, data)) in p.statics.iter().zip(statics.into_iter()).enumerate() {
        match (decl, data) {
            (StaticTy::Scalar(ct), StaticData::Scalar { valid, val }) => {
                if val.ty() != ct.ty {
                    return Err(CompileError::Static(format!(
                        "@{i}: scalar is {}, declared {}",
                        val.ty().name(),
                        ct.ty.name()
                    )));
                }
                if !valid && !ct.nullable {
                    return Err(CompileError::Static(format!(
                        "@{i}: NULL scalar for a non-nullable declaration"
                    )));
                }
                prepared.push(PreparedStatic::Scalar { valid, val });
            }
            (StaticTy::Map { keys, values }, StaticData::Map(mut entries)) => {
                for (ei, (k, v)) in entries.iter().enumerate() {
                    let kt: Vec<Ty> = k.iter().map(|kb| kb.ty()).collect();
                    let vt: Vec<Ty> = v.iter().map(|sv| sv.ty()).collect();
                    if kt != *keys || vt != *values {
                        return Err(CompileError::Static(format!(
                            "@{i}: entry {ei} has shape ({kt:?}) -> ({vt:?}), declared \
                             ({keys:?}) -> ({values:?})"
                        )));
                    }
                }
                // Canonicalize f64 key bits BEFORE sorting, so the stored
                // order agrees with the canonical bits cmp_key searches by.
                for (k, _) in entries.iter_mut() {
                    for kb in k.iter_mut() {
                        if let KeyBits::F64(bits) = kb {
                            *bits = super::canon_f64_bits(f64::from_bits(*bits));
                        }
                    }
                }
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                if entries.windows(2).any(|w| w[0].0 == w[1].0) {
                    return Err(CompileError::Static(format!("@{i}: duplicate map key")));
                }
                prepared.push(PreparedStatic::Map { entries });
            }
            (StaticTy::MultiMap { keys, values }, StaticData::Map(mut entries)) => {
                for (ei, (k, v)) in entries.iter().enumerate() {
                    let kt: Vec<Ty> = k.iter().map(|kb| kb.ty()).collect();
                    let vt: Vec<Ty> = v.iter().map(|sv| sv.ty()).collect();
                    if kt != *keys || vt != *values {
                        return Err(CompileError::Static(format!(
                            "@{i}: entry {ei} has shape ({kt:?}) -> ({vt:?}), declared                              ({keys:?}) -> ({values:?})"
                        )));
                    }
                }
                for (k, _) in entries.iter_mut() {
                    for kb in k.iter_mut() {
                        if let KeyBits::F64(bits) = kb {
                            *bits = super::canon_f64_bits(f64::from_bits(*bits));
                        }
                    }
                }
                // Stable sort: equal keys keep INSERTION order — the
                // engine's documented emission order for 1:N matches.
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                prepared.push(PreparedStatic::MultiMap { entries });
            }
            (StaticTy::BatchMap { .. }, StaticData::Map(entries)) => {
                if !entries.is_empty() {
                    return Err(CompileError::Static(format!(
                        "@{i}: batchmap takes no prepared entries"
                    )));
                }
                prepared.push(PreparedStatic::BatchMap);
            }
            (StaticTy::BatchMap { .. }, StaticData::Scalar { .. }) => {
                return Err(CompileError::Static(format!(
                    "@{i}: declared batchmap, got scalar data"
                )))
            }
            (StaticTy::MultiMap { .. }, StaticData::Scalar { .. }) => {
                return Err(CompileError::Static(format!(
                    "@{i}: declared multimap, got scalar data"
                )))
            }
            (StaticTy::Scalar(_), StaticData::Map(_)) => {
                return Err(CompileError::Static(format!(
                    "@{i}: declared scalar, got map data"
                )))
            }
            (StaticTy::Map { .. }, StaticData::Scalar { .. }) => {
                return Err(CompileError::Static(format!(
                    "@{i}: declared map, got scalar data"
                )))
            }
        }
    }
    Ok(prepared)
}

// ------------------------------------------------------------ registers --
// Verified programs make these matches infallible; a miss is a bug in the
// verifier or the compiler, not in the program, hence unreachable!.

fn as_i1(r: RegVal) -> bool {
    match r {
        RegVal::I1(b) => b,
        _ => unreachable!("type hole past the verifier: expected i1"),
    }
}

fn as_i64(r: RegVal) -> i64 {
    match r {
        RegVal::I64(v) => v,
        _ => unreachable!("type hole past the verifier: expected i64"),
    }
}

fn as_f64(r: RegVal) -> f64 {
    match r {
        RegVal::F64(v) => v,
        _ => unreachable!("type hole past the verifier: expected f64"),
    }
}

fn as_str(r: RegVal) -> StrRef {
    match r {
        RegVal::Str(s) => s,
        _ => unreachable!("type hole past the verifier: expected str"),
    }
}

fn scalar_to_reg(v: &ScalarVal, arena: &mut Arena) -> RegVal {
    match v {
        ScalarVal::I1(b) => RegVal::I1(*b),
        ScalarVal::I64(i) => RegVal::I64(*i),
        ScalarVal::F64(f) => RegVal::F64(*f),
        ScalarVal::Str(s) => RegVal::Str(arena.push_str(s)),
    }
}

fn default_reg(ty: Ty) -> RegVal {
    match ty {
        Ty::I1 => RegVal::I1(false),
        Ty::I64 => RegVal::I64(0),
        Ty::F64 => RegVal::F64(0.0),
        Ty::Str => RegVal::Str(StrRef { off: 0, len: 0 }),
    }
}

// ---------------------------------------------------------- compilation --

fn compile_term(p: &Program, t: &Term, slots: &HashMap<u32, u32>) -> CTerm {
    let mk_moves = |to: ir::BlockId, args: &Vec<Value>| -> (usize, Vec<(u32, u32)>) {
        let params = &p.blocks[to.0 as usize].params;
        let moves = args
            .iter()
            .zip(params.iter())
            .map(|(src, (dst, _))| (sl(slots, *src) as u32, sl(slots, *dst) as u32))
            .collect();
        (to.0 as usize, moves)
    };
    match t {
        Term::Jump { to, args } => {
            let (to, moves) = mk_moves(*to, args);
            CTerm::Jump { to, moves }
        }
        Term::Brif {
            cond,
            then_to,
            then_args,
            else_to,
            else_args,
        } => {
            let (then_to, then_moves) = mk_moves(*then_to, then_args);
            let (else_to, else_moves) = mk_moves(*else_to, else_args);
            CTerm::Brif {
                cond: sl(slots, *cond) as u32,
                then_to,
                then_moves,
                else_to,
                else_moves,
            }
        }
        Term::EmitTo { to, args } => {
            let (to, moves) = mk_moves(*to, args);
            CTerm::EmitTo { to, moves }
        }
        Term::Emit => CTerm::Emit,
        Term::Skip => CTerm::Skip,
        Term::Trap { msg } => CTerm::Trap(msg.clone()),
    }
}

/// DuckDB's substr window arithmetic (measured 1.5.5), on codepoints — NOT
/// grapheme clusters (substr slices inside ZWJ emoji). 1-based virtual
/// positions: negative start counts from the end (`start = n + start + 1`),
/// start <= 0 consumes length before character 1, negative len is "". A
/// missing SQL length arrives as i64::MAX; the saturating add makes that
/// "rest of the string".
/// DuckDB's substr window arithmetic — the VECTORIZED path, which columns
/// (and therefore every real query and the mined corpus) take; DuckDB's own
/// constant-fold path disagrees with it on negative starts (measured
/// 2026-07-26, see the builtin-pins spec). Codepoints, NOT grapheme
/// clusters. 1-based positions: a negative start counts from the end and
/// clamps to 1 (`rs = max(n + start + 1, 1)`) while start 0 stays virtual;
/// a non-negative length runs forward `[rs, rs+len)`, a NEGATIVE length
/// slices BACKWARDS `[rs+len, rs)`; `len: None` is the 2-arg rest-of-string
/// form.
pub(super) fn substr_window(s: &str, start: i64, len: Option<i64>) -> std::ops::Range<usize> {
    let n = s.chars().count() as i64;
    let rs = if start < 0 {
        (n + start + 1).max(1)
    } else {
        start
    };
    let (lo, hi) = match len {
        Some(l) if l >= 0 => (rs, rs.saturating_add(l)),
        Some(l) => (rs.saturating_add(l), rs),
        None => (rs, n + 1),
    };
    let (lo, hi) = (lo.max(1), hi.min(n + 1));
    if hi <= lo {
        return 0..0;
    }
    // Byte range of the char window — the output is a subview of `s`, so
    // callers slice instead of copying.
    let skip = (lo - 1) as usize;
    let take = (hi - lo) as usize;
    let b0 = s
        .char_indices()
        .nth(skip)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    let b1 = s[b0..]
        .char_indices()
        .nth(take)
        .map(|(i, _)| b0 + i)
        .unwrap_or(s.len());
    b0..b1
}

/// The byte range of `s` that survives trimming `set` chars from the chosen
/// ends — pure arithmetic, the output aliases the input. Membership scans
/// `set` per char (a trim set is a handful of chars; no Vec).
pub(super) fn trim_bounds(s: &str, set: &str, side: TrimSide) -> std::ops::Range<usize> {
    let hit = |c: char| set.chars().any(|k| k == c);
    let t = match side {
        TrimSide::Both => s.trim_matches(hit),
        TrimSide::Lead => s.trim_start_matches(hit),
        TrimSide::Trail => s.trim_end_matches(hit),
    };
    let start = t.as_ptr() as usize - s.as_ptr() as usize;
    start..start + t.len()
}

/// The offset/length guard DuckDB applies before the window: values outside
/// [-2^32, 2^32-1] raise an Out of Range error (measured boundary-exactly).
pub(super) fn substr_range_ok(v: i64) -> bool {
    (-(1i64 << 32)..(1i64 << 32)).contains(&v)
}

/// DuckDB's DOUBLE -> VARCHAR text (measured 1.5.5): Rust's shortest
/// round-trip form, except the exponent carries an explicit sign and at
/// least two digits (`1e+300`, `1e-05`) and NaN is lowercase `nan`.
pub(super) struct DuckF64(pub(super) f64);

impl std::fmt::Display for DuckF64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_nan() {
            return f.write_str("nan");
        }
        // Stack-render the shortest round-trip form (≤ 24 bytes for any
        // f64) so the hot path never builds a temp String.
        let mut buf = StackStr::<32>::default();
        {
            use std::fmt::Write;
            write!(buf, "{:?}", self.0).expect("f64 debug fits 32 bytes");
        }
        let s = buf.as_str();
        match s.find('e') {
            None => f.write_str(s),
            Some(pos) => {
                let exp: i64 = s[pos + 1..].parse().expect("float exponent");
                write!(
                    f,
                    "{}e{}{:02}",
                    &s[..pos],
                    if exp < 0 { '-' } else { '+' },
                    exp.abs()
                )
            }
        }
    }
}

/// Fixed-capacity ASCII scratch for `write!` — errors instead of growing.
struct StackStr<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> Default for StackStr<N> {
    fn default() -> Self {
        StackStr {
            buf: [0; N],
            len: 0,
        }
    }
}

impl<const N: usize> StackStr<N> {
    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.buf[..self.len]).expect("writes were valid UTF-8")
    }
}

impl<const N: usize> std::fmt::Write for StackStr<N> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let b = s.as_bytes();
        if self.len + b.len() > N {
            return Err(std::fmt::Error);
        }
        self.buf[self.len..self.len + b.len()].copy_from_slice(b);
        self.len += b.len();
        Ok(())
    }
}

// ------------------------------------------------- wave-1 math semantics --
// One shared fn per op, used verbatim by BOTH backends (pins:
// docs/superpowers/specs/2026-07-26-wave1-builtin-pins.md). Trap messages
// are DuckDB 1.5.5's own, measured — including its typo.

fn log_guard(x: f64) -> Result<(), Trap> {
    // Comparison-based, not is_finite-based: NaN fails both checks and
    // flows through to libm; +inf passes (measured).
    if x == 0.0 {
        return Err(Trap("cannot take logarithm of zero".into()));
    }
    if x < 0.0 {
        return Err(Trap("cannot take logarithm of a negative number".into()));
    }
    Ok(())
}

pub(super) fn duck_ln(x: f64) -> Result<f64, Trap> {
    log_guard(x)?;
    Ok(if x.is_nan() { x } else { x.ln() })
}
pub(super) fn duck_log2(x: f64) -> Result<f64, Trap> {
    log_guard(x)?;
    Ok(if x.is_nan() { x } else { x.log2() })
}
pub(super) fn duck_log10(x: f64) -> Result<f64, Trap> {
    log_guard(x)?;
    Ok(if x.is_nan() { x } else { x.log10() })
}
/// log(base, x): bit-exactly log10(x)/log10(base) — NOT an ln ratio
/// (refuted on 20k fuzz samples). Base is domain-checked FIRST; base==1
/// carries DuckDB's message verbatim, typo included.
pub(super) fn duck_logb(base: f64, x: f64) -> Result<f64, Trap> {
    log_guard(base)?;
    if base == 1.0 {
        return Err(Trap("divison by zero in based logarithm".into()));
    }
    log_guard(x)?;
    Ok(x.log10() / base.log10())
}
pub(super) fn duck_exp(x: f64) -> Result<f64, Trap> {
    // TOTAL: overflow -> inf, underflow -> denormals -> +0.0 (measured).
    Ok(x.exp())
}
pub(super) fn duck_sqrt(x: f64) -> Result<f64, Trap> {
    if x < 0.0 {
        return Err(Trap("cannot take square root of a negative number".into()));
    }
    Ok(x.sqrt()) // sqrt(-0.0) = -0.0 (not negative, not a trap); NaN passes
}
pub(super) fn duck_cbrt(x: f64) -> Result<f64, Trap> {
    Ok(x.cbrt()) // TOTAL; cbrt(-8) = -2.0 exactly (NOT pow(x, 1/3))
}
fn trig_guard(x: f64) -> Result<(), Trap> {
    if x.is_infinite() {
        return Err(Trap(format!(
            "input value {} is out of range for numeric function",
            if x > 0.0 { "inf" } else { "-inf" }
        )));
    }
    Ok(())
}
pub(super) fn duck_sin(x: f64) -> Result<f64, Trap> {
    trig_guard(x)?;
    // NaN passes through BIT-EXACTLY (payload + sign) — never hand it to
    // libm (measured: DuckDB preserves even signaling patterns).
    Ok(if x.is_nan() { x } else { x.sin() })
}
pub(super) fn duck_cos(x: f64) -> Result<f64, Trap> {
    trig_guard(x)?;
    Ok(if x.is_nan() { x } else { x.cos() })
}
pub(super) fn duck_tan(x: f64) -> Result<f64, Trap> {
    trig_guard(x)?;
    Ok(if x.is_nan() { x } else { x.tan() })
}
pub(super) fn duck_pow(x: f64, y: f64) -> Result<f64, Trap> {
    // TOTAL, pure IEEE: pow(NaN,0)=1, pow(1,NaN)=1, pow(0,-1)=inf,
    // negative^fractional=NaN, overflow=inf (all measured).
    Ok(x.powf(y))
}
pub(super) fn duck_floor(x: f64) -> Result<f64, Trap> {
    Ok(x.floor())
}
pub(super) fn duck_ceil(x: f64) -> Result<f64, Trap> {
    Ok(x.ceil())
}
pub(super) fn duck_trunc(x: f64) -> Result<f64, Trap> {
    Ok(x.trunc())
}

/// The wave-1 f64 unaries as shared fn pointers (Iabs/Fabs/Fround keep
/// their original arms).
pub(super) fn math1_fn(op: NumOp1) -> fn(f64) -> Result<f64, Trap> {
    match op {
        NumOp1::Ln => duck_ln,
        NumOp1::Log2 => duck_log2,
        NumOp1::Log10 => duck_log10,
        NumOp1::Fexp => duck_exp,
        NumOp1::Fsqrt => duck_sqrt,
        NumOp1::Fcbrt => duck_cbrt,
        NumOp1::Fsin => duck_sin,
        NumOp1::Fcos => duck_cos,
        NumOp1::Ftan => duck_tan,
        NumOp1::Ffloor => duck_floor,
        NumOp1::Fceil => duck_ceil,
        NumOp1::Ftrunc => duck_trunc,
        NumOp1::Iabs | NumOp1::Fabs | NumOp1::Fround => {
            unreachable!("legacy unaries keep dedicated arms")
        }
    }
}

/// DuckDB's pow(10, k) — the oracle-extracted table; inf beyond 308.
fn pow10(k: i64) -> f64 {
    if (0..=308).contains(&k) {
        super::pow10::POW10[k as usize]
    } else {
        f64::INFINITY
    }
}

/// round(x, n) on f64 — scale-then-round with the oracle pow table.
/// Non-finite results fall back to the INPUT for n >= 0 and to +0.0 for
/// n < 0 (measured asymmetry: round(NaN, -2) = 0.0).
pub(super) fn round_prec_f64(x: f64, n: i64) -> f64 {
    if n >= 0 {
        let m = pow10(n);
        let r = (x * m).round() / m;
        if r.is_infinite() || r.is_nan() {
            x
        } else {
            r
        }
    } else {
        let m = pow10(n.unsigned_abs() as i64);
        let r = (x / m).round() * m;
        if r.is_infinite() || r.is_nan() {
            0.0
        } else {
            r
        }
    }
}

/// trunc(x, n) on f64: same shape as round, but BOTH branches fall back
/// to the input — the round/trunc asymmetry is measured, not a bug.
pub(super) fn trunc_prec_f64(x: f64, n: i64) -> f64 {
    if n >= 0 {
        let m = pow10(n);
        let r = (x * m).trunc() / m;
        if r.is_infinite() || r.is_nan() {
            x
        } else {
            r
        }
    } else {
        let m = pow10(n.unsigned_abs() as i64);
        let r = (x / m).trunc() * m;
        if r.is_infinite() || r.is_nan() {
            x
        } else {
            r
        }
    }
}

/// Integer round with digits: identity for n >= 0; n < 0 WRAPS at i64
/// (measured: round(i64::MAX, -2) = -9223372036854775700) — never traps.
pub(super) fn round_prec_i64(x: i64, n: i64) -> i64 {
    if n >= 0 {
        return x;
    }
    let p = n.unsigned_abs();
    if p >= 19 {
        return 0;
    }
    let power = 10i64.pow(p as u32);
    let half = power / 2;
    let y = if x >= 0 {
        x.wrapping_add(half)
    } else {
        x.wrapping_sub(half)
    };
    (y / power) * power
}

/// Integer trunc with digits: identity for n >= 0, truncating scale for
/// n < 0 — no half-add, never wraps.
pub(super) fn trunc_prec_i64(x: i64, n: i64) -> i64 {
    if n >= 0 {
        return x;
    }
    let p = n.unsigned_abs();
    if p >= 19 {
        return 0;
    }
    let power = 10i64.pow(p as u32);
    (x / power) * power
}

// ------------------------------------------------------ LIKE (wave 2) --
// Byte-based matcher with codepoint `_`, reproducing every DuckDB 1.5.5
// pin including the DATA-DEPENDENT dangling-escape error (raised only
// when the matcher examines a trailing escape while string bytes remain;
// plain false when the string is exhausted). Iterative two-pointer
// restart: identical booleans and identical error rows to the leftmost-
// first recursive semantics, never DuckDB's own O(n^k) blowup (measured
// 23s/row there on pathological patterns; ours is O(n*m)).
// Pins: docs/superpowers/specs/pins-wave1/pins_like.json.

fn utf8_width(b: u8) -> usize {
    match b {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        // Continuation/invalid lead: advance one byte (spans are valid
        // UTF-8; this arm is reachable only from a % restart landing on a
        // continuation byte, where the subsequent literal compare fails
        // anyway — pinned by the multibyte %_ duck_checks).
        _ => 1,
    }
}

pub(super) fn like_match(s: &[u8], p: &[u8], esc: Option<u8>) -> Result<bool, Trap> {
    let (mut si, mut pi) = (0usize, 0usize);
    // Backtrack state: pattern index just past the last %, and the string
    // index its current attempt started at.
    let (mut star_p, mut star_s): (Option<usize>, usize) = (None, 0);
    loop {
        if pi < p.len() {
            let pc = p[pi];
            if Some(pc) == esc {
                // Escape intro beats % and _ (ESCAPE '%' de-wildcards it).
                if si == s.len() {
                    return Ok(false); // exhausted string: plain false
                }
                if pi + 1 == p.len() {
                    return Err(Trap(
                        "Like pattern must not end with escape character!".into(),
                    ));
                }
                if s[si] == p[pi + 1] {
                    si += 1;
                    pi += 2;
                    continue;
                }
            } else if pc == b'%' {
                while pi < p.len() && p[pi] == b'%' && Some(b'%') != esc {
                    pi += 1;
                }
                if pi == p.len() {
                    return Ok(true);
                }
                star_p = Some(pi);
                star_s = si;
                continue;
            } else if pc == b'_' {
                if si < s.len() {
                    si += utf8_width(s[si]);
                    pi += 1;
                    continue;
                }
            } else if si < s.len() && s[si] == pc {
                si += 1;
                pi += 1;
                continue;
            }
        } else if si == s.len() {
            return Ok(true);
        }
        // Mismatch: restart at the last % with one more byte consumed.
        match star_p {
            Some(sp) if star_s < s.len() => {
                star_s += 1;
                si = star_s;
                pi = sp;
            }
            _ => return Ok(false),
        }
    }
}

/// Validate an ESCAPE operand per row (AFTER NULL handling): empty means
/// no escape; the limit is one BYTE (a single 2-byte codepoint errors).
pub(super) fn like_escape_of(esc: &str) -> Result<Option<u8>, Trap> {
    match esc.len() {
        0 => Ok(None),
        1 => Ok(Some(esc.as_bytes()[0])),
        _ => Err(Trap(
            "Invalid escape string. Escape string must be empty or one character.".into(),
        )),
    }
}

// ------------------------------------------ wave-3 builtins (TASK-49) --
// One shared fn per op, used verbatim by BOTH backends. Pins:
// docs/superpowers/specs/2026-07-26-wave3-builtin-pins.md — all
// similarity ops are raw UTF-8 BYTE-based; error texts are DuckDB's own.

/// levenshtein == editdist3: classic two-row DP over bytes.
pub(super) fn duck_levenshtein(a: &[u8], b: &[u8]) -> i64 {
    if a.is_empty() || b.is_empty() {
        return (a.len() + b.len()) as i64;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ac) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &bc) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ac != bc);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()] as i64
}

/// damerau_levenshtein: the UNRESTRICTED DL variant (NOT OSA — witness
/// ('ca','abc') = 2), transposition cost 1, over bytes. Full matrix plus
/// a last-occurrence table, as the unrestricted recurrence requires.
// ponytail: O(n*m) memory; corpus strings are short — stream it if huge
// inputs ever matter.
pub(super) fn duck_damerau(a: &[u8], b: &[u8]) -> i64 {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return (n + m) as i64;
    }
    let w = m + 2;
    let inf = n + m;
    let mut d = vec![0usize; (n + 2) * w];
    d[0] = inf;
    for i in 0..=n {
        d[(i + 1) * w] = inf;
        d[(i + 1) * w + 1] = i;
    }
    for j in 0..=m {
        d[j + 1] = inf;
        d[w + j + 1] = j;
    }
    let mut last_a = [0usize; 256]; // last row where each byte occurred in a
    for i in 1..=n {
        let mut last_b = 0usize; // last column where a[i-1] matched in b
        for j in 1..=m {
            let (i1, j1) = (last_a[b[j - 1] as usize], last_b);
            let cost = usize::from(a[i - 1] != b[j - 1]);
            if cost == 0 {
                last_b = j;
            }
            let sub = d[i * w + j] + cost;
            let ins = d[(i + 1) * w + j] + 1;
            let del = d[i * w + j + 1] + 1;
            let trans = d[i1 * w + j1] + (i - i1 - 1) + 1 + (j - j1 - 1);
            d[(i + 1) * w + j + 1] = sub.min(ins).min(del).min(trans);
        }
        last_a[a[i - 1] as usize] = i;
    }
    d[(n + 1) * w + m + 1] as i64
}

/// jaccard: |A∩B|/|A∪B| over single-BYTE sets; empty either side traps.
pub(super) fn duck_jaccard(a: &[u8], b: &[u8]) -> Result<f64, Trap> {
    if a.is_empty() || b.is_empty() {
        return Err(Trap("Jaccard Function: An argument too short!".into()));
    }
    let (mut in_a, mut in_b) = ([false; 256], [false; 256]);
    for &c in a {
        in_a[c as usize] = true;
    }
    for &c in b {
        in_b[c as usize] = true;
    }
    let (mut inter, mut union) = (0i64, 0i64);
    for i in 0..256 {
        inter += i64::from(in_a[i] && in_b[i]);
        union += i64::from(in_a[i] || in_b[i]);
    }
    Ok(inter as f64 / union as f64)
}

/// hamming == mismatches: byte-wise; empty inputs and length mismatch
/// trap — the messages say "Mismatch Function" even for hamming
/// (measured; the error text leaks the shared implementation).
pub(super) fn duck_hamming(a: &[u8], b: &[u8]) -> Result<i64, Trap> {
    if a.is_empty() || b.is_empty() {
        return Err(Trap(
            "Mismatch Function: Strings must be of length > 0!".into(),
        ));
    }
    if a.len() != b.len() {
        return Err(Trap(
            "Mismatch Function: Strings must be of equal length!".into(),
        ));
    }
    Ok(a.iter().zip(b).filter(|(x, y)| x != y).count() as i64)
}

/// Engine guard for string-building ops: DuckDB errors on absurd result
/// sizes too (exact threshold/message unpinned — huge-n was deliberately
/// not probed); 1 GiB keeps the engine alive without shadowing any pin.
const STR_BUILD_CAP: u64 = 1 << 30;

/// repeat(s, n): n <= 0 -> '' silently.
pub(super) fn duck_repeat(s: &str, n: i64) -> Result<String, Trap> {
    if n <= 0 {
        return Ok(String::new());
    }
    match (s.len() as u64).checked_mul(n as u64) {
        Some(sz) if sz <= STR_BUILD_CAP => Ok(s.repeat(n as usize)),
        _ => Err(Trap("string builder result exceeds 1 GiB".into())),
    }
}

/// lpad/rpad(s, l, pad): l counts CODEPOINTS; truncation keeps the FIRST
/// l codepoints for BOTH sides; the pad cycles cut at codepoint
/// boundaries; empty pad traps ONLY when growth is needed.
pub(super) fn duck_pad(left: bool, s: &str, l: i64, pad: &str) -> Result<String, Trap> {
    if l <= 0 {
        return Ok(String::new());
    }
    let l = l as usize;
    let n = s.chars().count();
    if l <= n {
        return Ok(s.chars().take(l).collect());
    }
    if pad.is_empty() {
        return Err(Trap(format!(
            "Insufficient padding in {}.",
            if left { "LPAD" } else { "RPAD" }
        )));
    }
    if l as u64 > STR_BUILD_CAP / 4 {
        return Err(Trap("string builder result exceeds 1 GiB".into()));
    }
    let fill: String = pad.chars().cycle().take(l - n).collect();
    Ok(if left {
        fill + s
    } else {
        let mut out = s.to_string();
        out.push_str(&fill);
        out
    })
}

/// replace(s, from, to): empty needle is a strict NO-OP; leftmost
/// non-overlapping single pass (Rust `str::replace` is exactly that).
pub(super) fn duck_replace(s: &str, from: &str, to: &str) -> String {
    if from.is_empty() {
        return s.to_string();
    }
    s.replace(from, to)
}

/// translate(s, from, to): per-codepoint map, FIRST occurrence in `from`
/// wins, from-chars beyond |to| are deleted.
pub(super) fn duck_translate(s: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match from.chars().position(|f| f == c) {
            None => out.push(c),
            Some(i) => {
                if let Some(t) = to.chars().nth(i) {
                    out.push(t);
                }
            }
        }
    }
    out
}

/// array_extract/list_extract/s[i] on VARCHAR: 1-based codepoint,
/// negative resolves len+1+i, out-of-range/0 -> '' (NOT NULL). Returns
/// the byte range so callers can subview instead of copying.
pub(super) fn extract_window(s: &str, i: i64) -> std::ops::Range<usize> {
    let n = s.chars().count() as i64;
    let pos = if i < 0 { (n + 1).saturating_add(i) } else { i };
    if pos < 1 || pos > n {
        return 0..0;
    }
    let skip = (pos - 1) as usize;
    let b0 = s
        .char_indices()
        .nth(skip)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    let b1 = s[b0..]
        .chars()
        .next()
        .map(|c| b0 + c.len_utf8())
        .unwrap_or(s.len());
    b0..b1
}

/// array_slice/list_slice/s[a:b] on VARCHAR: 1-based, both-ends-INCLUSIVE
/// codepoints, negative from-end (-1 = last), lo <= 0 clamps to start,
/// hi > len clamps to end, reversed/out-of-range -> ''. Byte range out.
pub(super) fn slice_window(s: &str, lo: i64, hi: i64) -> std::ops::Range<usize> {
    let n = s.chars().count() as i64;
    let rlo = (if lo < 0 {
        (n + 1).saturating_add(lo)
    } else {
        lo
    })
    .max(1);
    let rhi = (if hi < 0 {
        (n + 1).saturating_add(hi)
    } else {
        hi
    })
    .min(n);
    if rlo > rhi {
        return 0..0;
    }
    let skip = (rlo - 1) as usize;
    let take = (rhi - rlo + 1) as usize;
    let b0 = s
        .char_indices()
        .nth(skip)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    let b1 = s[b0..]
        .char_indices()
        .nth(take)
        .map(|(i, _)| b0 + i)
        .unwrap_or(s.len());
    b0..b1
}

/// unicode/ord ('' -> -1) and ascii ('' -> 0 — the sole divergence):
/// first codepoint as i64.
pub(super) fn duck_ord(s: &str, empty_zero: bool) -> i64 {
    match s.chars().next() {
        Some(c) => c as i64,
        None => {
            if empty_zero {
                0
            } else {
                -1
            }
        }
    }
}

/// strip_accents: all-ASCII passes VERBATIM (NULs preserved); otherwise
/// truncate at the first NUL (the measured context-dependent quirk),
/// per-codepoint oracle map, then Hangul jamo composition. TOTAL.
/// DuckDB reverse() (pins-waveA/reverse-graphemes.json): an all-ASCII
/// string BYTE-reverses (this splits CRLF — measured, not a bug to "fix");
/// anything else reverses UAX-29 EXTENDED grapheme clusters, each cluster
/// byte-preserved, no normalization.
pub(super) fn duck_reverse(s: &str) -> String {
    if s.is_ascii() {
        s.bytes().rev().map(|b| b as char).collect()
    } else {
        use unicode_segmentation::UnicodeSegmentation;
        s.graphemes(true).rev().collect()
    }
}

pub(super) fn duck_strip_accents(s: &str) -> Option<String> {
    if s.is_ascii() {
        return None; // caller keeps the input span — no copy
    }
    let cut = s.find('\0').map(|i| &s[..i]).unwrap_or(s);
    let mut out = String::with_capacity(cut.len());
    for c in cut.chars() {
        let mapped = match super::strip_accents::strip_map(c) {
            None => Some(c),
            Some(m) => m,
        };
        let Some(mc) = mapped else { continue };
        // Hangul compose against the last emitted codepoint: L+V -> LV,
        // LV+T -> LVT (formulaic; wrong-order jamo never compose).
        if let Some(p) = out.chars().next_back() {
            let (pu, cu) = (p as u32, mc as u32);
            if (0x1100..=0x1112).contains(&pu) && (0x1161..=0x1175).contains(&cu) {
                out.pop();
                let lv = 0xAC00 + (pu - 0x1100) * 588 + (cu - 0x1161) * 28;
                out.push(char::from_u32(lv).expect("Hangul LV in range"));
                continue;
            }
            if (0xAC00..0xAC00 + 11172).contains(&pu)
                && (pu - 0xAC00) % 28 == 0
                && (0x11A8..=0x11C2).contains(&cu)
            {
                out.pop();
                let lvt = pu + (cu - 0x11A7);
                out.push(char::from_u32(lvt).expect("Hangul LVT in range"));
                continue;
            }
        }
        out.push(mc);
    }
    Some(out)
}

/// SQL fdiv(x, y) = floor(x / y) — TOTAL, ±inf on zero divisor.
pub(super) fn duck_fdiv(x: f64, y: f64) -> f64 {
    (x / y).floor()
}

/// SQL fmod(x, y) = x − floor(x/y)·y — FLOORED (divisor's sign), NaN on
/// zero or infinite divisor. NOT C fmod (that is SQL `%`).
pub(super) fn duck_fmod(x: f64, y: f64) -> f64 {
    x - (x / y).floor() * y
}

/// C nextafter, bit-exact; x == y returns y (incl. signed zeros).
pub(super) fn duck_nextafter(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        // toward y from zero: the smallest denormal with y's direction.
        return f64::from_bits(1).copysign(y - x);
    }
    let bits = x.to_bits();
    let up = (y > x) == (x > 0.0); // move away from zero?
    f64::from_bits(if up { bits + 1 } else { bits - 1 })
}

/// DuckDB's int-overflow trap texts, verbatim (wave-3 pins: measured via
/// both the operators and their function aliases). Division covers `//`
/// AND `%` on i64::MIN op -1 — DuckDB's own % message says "division",
/// and only add/sub/mul carry the trailing '!'.
/// One parsed GLOB pattern element (bytes, not codepoints — measured).
enum GTok {
    Star,
    Any,
    Byte(u8),
    Class { neg: bool, set: Vec<(u8, u8)> },
}

/// Parse a GLOB pattern per the wave-5 pins. `None` = dead pattern
/// (dangling `\`, unclosed class — incl. `[a-]` whose `]` is eaten as the
/// range endpoint): matches nothing, never errors.
fn parse_glob(p: &[u8]) -> Option<Vec<GTok>> {
    let mut toks = Vec::new();
    let mut i = 0;
    while i < p.len() {
        match p[i] {
            b'*' => {
                toks.push(GTok::Star);
                i += 1;
            }
            b'?' => {
                toks.push(GTok::Any);
                i += 1;
            }
            b'\\' => {
                // Escape OUTSIDE classes only; dangling -> dead.
                let &b = p.get(i + 1)?;
                toks.push(GTok::Byte(b));
                i += 2;
            }
            b'[' => {
                i += 1;
                let mut neg = false;
                if p.get(i) == Some(&b'!') {
                    // Only '!' negates; '^' is a literal member.
                    neg = true;
                    i += 1;
                }
                let mut set: Vec<(u8, u8)> = Vec::new();
                let mut first = true;
                loop {
                    let &b = p.get(i)?; // unclosed -> dead
                    if b == b']' && !first {
                        i += 1;
                        break;
                    }
                    first = false;
                    // ']' is a literal if first; '-' is literal when not
                    // followed by a range endpoint (which may be ']').
                    if p.get(i + 1) == Some(&b'-') && p.get(i + 2).is_some() && b != b'-' {
                        set.push((b, p[i + 2])); // inverted range = empty
                        i += 3;
                    } else {
                        set.push((b, b));
                        i += 1;
                    }
                }
                toks.push(GTok::Class { neg, set });
            }
            b => {
                toks.push(GTok::Byte(b));
                i += 1;
            }
        }
    }
    Some(toks)
}

/// GLOB (wave-5 pins): raw-byte matcher, `*` = any run, `?` = ONE byte,
/// case-sensitive, malformed patterns match nothing. NOT expressible via
/// LIKE (classes; `?` is byte-based while `_` is codepoint-based).
pub(in crate::specializer) fn duck_glob(s: &str, p: &str) -> bool {
    let Some(toks) = parse_glob(p.as_bytes()) else {
        return false;
    };
    let s = s.as_bytes();
    let (mut ti, mut si) = (0usize, 0usize);
    let (mut bt_t, mut bt_s) = (usize::MAX, 0usize);
    while si < s.len() {
        let stepped = match toks.get(ti) {
            Some(GTok::Star) => {
                bt_t = ti;
                ti += 1;
                bt_s = si;
                continue;
            }
            Some(GTok::Any) => true,
            Some(GTok::Byte(b)) => *b == s[si],
            Some(GTok::Class { neg, set }) => {
                set.iter().any(|(lo, hi)| (*lo..=*hi).contains(&s[si])) != *neg
            }
            None => false,
        };
        if stepped {
            ti += 1;
            si += 1;
        } else if bt_t != usize::MAX {
            bt_s += 1;
            si = bt_s;
            ti = bt_t + 1;
        } else {
            return false;
        }
    }
    while matches!(toks.get(ti), Some(GTok::Star)) {
        ti += 1;
    }
    ti == toks.len()
}

/// i64 `<<` per the wave-5 pins ladder: negative value first (even << 0),
/// then negative count, then the zero-value shortcut, then count range,
/// then overflow (value >= 2^(63-count), computed in i128 because
/// 1 << 63 doesn't fit i64). Texts DuckDB-verbatim sans the class prefix.
pub(in crate::specializer) fn duck_shl(x: i64, y: i64) -> Result<i64, Trap> {
    if x < 0 {
        return Err(Trap(format!("Cannot left-shift negative number {x}")));
    }
    if y < 0 {
        return Err(Trap(format!("Cannot left-shift by negative number {y}")));
    }
    if x == 0 {
        return Ok(0);
    }
    if y >= 64 {
        return Err(Trap(format!("Left-shift value {y} is out of range")));
    }
    if (x as i128) >= (1i128 << (63 - y)) {
        return Err(Trap(format!("Overflow in left shift ({x} << {y})")));
    }
    Ok(x << y)
}

/// i64 `>>` — total: out-of-range counts (either direction) give 0.
pub(in crate::specializer) fn duck_shr(x: i64, y: i64) -> i64 {
    if (0..64).contains(&y) {
        x >> y
    } else {
        0
    }
}

pub(super) fn overflow_msg(op: BinOp, x: i64, y: i64) -> String {
    match op {
        BinOp::Iadd => format!("Overflow in addition of INT64 ({x} + {y})!"),
        BinOp::Isub => format!("Overflow in subtraction of INT64 ({x} - {y})!"),
        BinOp::Imul => format!("Overflow in multiplication of INT64 ({x} * {y})!"),
        _ => format!("Overflow in division of {x} / {y}"),
    }
}

/// DuckDB's abs(i64::MIN) trap text, verbatim (measured 2026-07-26 — no
/// trailing '!', unlike the binary-op overflow family).
pub(super) fn abs_overflow_msg(x: i64) -> String {
    format!("Overflow on abs({x})")
}

/// Wave-1 string search (pins: 1-based CODEPOINT positions, empty needle
/// matches everything, byte-wise comparison, zero unicode intelligence).
pub(super) fn str_find(s: &str, n: &str) -> i64 {
    if n.is_empty() {
        return 1;
    }
    match s.find(n) {
        None => 0,
        Some(byte) => s[..byte].chars().count() as i64 + 1,
    }
}

pub(super) fn str_pred(op: StrOp2, s: &str, n: &str) -> bool {
    match op {
        StrOp2::Contains => s.contains(n),
        StrOp2::Starts => s.starts_with(n),
        StrOp2::Ends => s.ends_with(n),
        StrOp2::Glob => duck_glob(s, n),
        _ => unreachable!("non-predicate str2 ops have dedicated arms"),
    }
}

/// Value id -> dense register slot.
fn sl(slots: &HashMap<u32, u32>, v: Value) -> usize {
    slots[&v.0] as usize
}

fn compile_inst(
    p: &Program,
    inst: &Inst,
    slots: &HashMap<u32, u32>,
    regexes: &[std::rc::Rc<regex::Regex>],
) -> InstFn {
    match inst.clone() {
        Inst::ReMatch { re, dst, a } => {
            let rx = regexes[re as usize].clone();
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                let s = ctx.arena.get(as_str(ctx.regs[a]));
                let m = rx.is_match(s);
                ctx.regs[dst] = RegVal::I1(m);
                Ok(())
            })
        }
        Inst::ReExtract { re, group, dst, a } => {
            let rx = regexes[re as usize].clone();
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                // No match / non-participating group -> '' (wave-B pins).
                let out = {
                    let s = ctx.arena.get(as_str(ctx.regs[a]));
                    rx.captures(s)
                        .and_then(|c| c.get(group as usize))
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_default()
                };
                ctx.regs[dst] = RegVal::Str(ctx.arena.push_str(&out));
                Ok(())
            })
        }
        Inst::ReReplace { re, global, dst, a } => {
            let rx = regexes[re as usize].clone();
            let template = p.regexes[re as usize]
                .rewrite
                .clone()
                .expect("verified: rereplace has a template");
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                let out = {
                    let s = ctx.arena.get(as_str(ctx.regs[a]));
                    if global {
                        rx.replace_all(s, template.as_str()).into_owned()
                    } else {
                        rx.replace(s, template.as_str()).into_owned()
                    }
                };
                ctx.regs[dst] = RegVal::Str(ctx.arena.push_str(&out));
                Ok(())
            })
        }
        Inst::Const { dst, lit } => {
            let dst = sl(slots, dst);
            match lit {
                ir::Lit::I1(b) => Box::new(move |ctx| {
                    ctx.regs[dst] = RegVal::I1(b);
                    Ok(())
                }),
                ir::Lit::I64(i) => Box::new(move |ctx| {
                    ctx.regs[dst] = RegVal::I64(i);
                    Ok(())
                }),
                ir::Lit::F64(f) => Box::new(move |ctx| {
                    ctx.regs[dst] = RegVal::F64(f);
                    Ok(())
                }),
                ir::Lit::Str(s) => Box::new(move |ctx| {
                    ctx.regs[dst] = RegVal::Str(ctx.arena.push_str(&s));
                    Ok(())
                }),
            }
        }
        Inst::Bin { op, dst, a, b } => {
            let (dst, a, b) = (sl(slots, dst), sl(slots, a), sl(slots, b));
            match op {
                BinOp::Iadd | BinOp::Isub | BinOp::Imul => Box::new(move |ctx| {
                    let (x, y) = (as_i64(ctx.regs[a]), as_i64(ctx.regs[b]));
                    let r = match op {
                        BinOp::Iadd => x.checked_add(y),
                        BinOp::Isub => x.checked_sub(y),
                        _ => x.checked_mul(y),
                    };
                    match r {
                        Some(v) => {
                            ctx.regs[dst] = RegVal::I64(v);
                            Ok(())
                        }
                        None => Err(Trap(overflow_msg(op, x, y))),
                    }
                }),
                BinOp::Idiv | BinOp::Irem => Box::new(move |ctx| {
                    let (x, y) = (as_i64(ctx.regs[a]), as_i64(ctx.regs[b]));
                    if y == 0 {
                        return Err(Trap(format!("division by zero in {}", op.name())));
                    }
                    let r = match op {
                        BinOp::Idiv => x.checked_div(y),
                        _ => x.checked_rem(y),
                    };
                    match r {
                        Some(v) => {
                            ctx.regs[dst] = RegVal::I64(v);
                            Ok(())
                        }
                        None => Err(Trap(overflow_msg(op, x, y))),
                    }
                }),
                BinOp::Fadd
                | BinOp::Fsub
                | BinOp::Fmul
                | BinOp::Fdiv
                | BinOp::Frem
                | BinOp::Fpow
                | BinOp::Ffloordiv
                | BinOp::Ffloormod
                | BinOp::Fnextafter => Box::new(move |ctx| {
                    let (x, y) = (as_f64(ctx.regs[a]), as_f64(ctx.regs[b]));
                    let v = match op {
                        BinOp::Fadd => x + y,
                        BinOp::Fsub => x - y,
                        BinOp::Fmul => x * y,
                        BinOp::Fdiv => x / y,
                        BinOp::Fpow => duck_pow(x, y)?,
                        BinOp::Ffloordiv => duck_fdiv(x, y),
                        BinOp::Ffloormod => duck_fmod(x, y),
                        BinOp::Fnextafter => duck_nextafter(x, y),
                        _ => x % y,
                    };
                    ctx.regs[dst] = RegVal::F64(v);
                    Ok(())
                }),
                // log(base, x): a is the base (SQL argument order).
                BinOp::Flogb => Box::new(move |ctx| {
                    let (base, x) = (as_f64(ctx.regs[a]), as_f64(ctx.regs[b]));
                    ctx.regs[dst] = RegVal::F64(duck_logb(base, x)?);
                    Ok(())
                }),
                BinOp::Ishl => Box::new(move |ctx| {
                    let (x, y) = (as_i64(ctx.regs[a]), as_i64(ctx.regs[b]));
                    ctx.regs[dst] = RegVal::I64(duck_shl(x, y)?);
                    Ok(())
                }),
                BinOp::Ishr | BinOp::Iand | BinOp::Ior | BinOp::Ixor => Box::new(move |ctx| {
                    let (x, y) = (as_i64(ctx.regs[a]), as_i64(ctx.regs[b]));
                    let v = match op {
                        BinOp::Ishr => duck_shr(x, y),
                        BinOp::Iand => x & y,
                        BinOp::Ior => x | y,
                        _ => x ^ y,
                    };
                    ctx.regs[dst] = RegVal::I64(v);
                    Ok(())
                }),
                BinOp::And | BinOp::Or | BinOp::Xor => Box::new(move |ctx| {
                    let (x, y) = (as_i1(ctx.regs[a]), as_i1(ctx.regs[b]));
                    let v = match op {
                        BinOp::And => x && y,
                        BinOp::Or => x || y,
                        _ => x ^ y,
                    };
                    ctx.regs[dst] = RegVal::I1(v);
                    Ok(())
                }),
            }
        }
        Inst::Cmp {
            pred,
            ty,
            dst,
            a,
            b,
        } => {
            let (dst, a, b) = (sl(slots, dst), sl(slots, a), sl(slots, b));
            Box::new(move |ctx| {
                let v = match ty {
                    Ty::I64 => apply_ord(pred, as_i64(ctx.regs[a]).cmp(&as_i64(ctx.regs[b]))),
                    Ty::F64 => {
                        // DuckDB DOUBLE order, not IEEE: NaN = NaN, NaN above
                        // everything, zeros equal (see exec::duck_fcmp).
                        let (x, y) = (as_f64(ctx.regs[a]), as_f64(ctx.regs[b]));
                        apply_ord(pred, super::duck_fcmp(x, y))
                    }
                    Ty::Str => {
                        let (x, y) = (as_str(ctx.regs[a]), as_str(ctx.regs[b]));
                        apply_ord(pred, ctx.arena.get(x).cmp(ctx.arena.get(y)))
                    }
                    Ty::I1 => unreachable!("cmp on i1 is rejected by the verifier"),
                };
                ctx.regs[dst] = RegVal::I1(v);
                Ok(())
            })
        }
        Inst::Not { dst, a } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                ctx.regs[dst] = RegVal::I1(!as_i1(ctx.regs[a]));
                Ok(())
            })
        }
        Inst::Select { dst, cond, a, b } => {
            let (dst, cond, a, b) = (sl(slots, dst), sl(slots, cond), sl(slots, a), sl(slots, b));
            Box::new(move |ctx| {
                ctx.regs[dst] = if as_i1(ctx.regs[cond]) {
                    ctx.regs[a]
                } else {
                    ctx.regs[b]
                };
                Ok(())
            })
        }
        Inst::Itof { dst, a } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                ctx.regs[dst] = RegVal::F64(as_i64(ctx.regs[a]) as f64);
                Ok(())
            })
        }
        Inst::Ftoi { mode, dst, a } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                let x = as_f64(ctx.regs[a]);
                let r = match mode {
                    RoundMode::Trunc => x.trunc(),
                    RoundMode::Round => x.round(), // half away from zero
                };
                // 2^63 is exactly representable; anything in [-2^63, 2^63)
                // fits i64 after rounding.
                if r.is_finite() && r >= -(2f64.powi(63)) && r < 2f64.powi(63) {
                    ctx.regs[dst] = RegVal::I64(r as i64);
                    Ok(())
                } else {
                    Err(Trap(format!("f64 value {x:?} out of i64 range in ftoi")))
                }
            })
        }
        Inst::Itos { dst, a } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                let v = as_i64(ctx.regs[a]);
                ctx.regs[dst] = RegVal::Str(ctx.arena.push_fmt(format_args!("{v}")));
                Ok(())
            })
        }
        Inst::Ftos { dst, a } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                let v = as_f64(ctx.regs[a]);
                ctx.regs[dst] = RegVal::Str(ctx.arena.push_fmt(format_args!("{}", DuckF64(v))));
                Ok(())
            })
        }
        Inst::StoiOpt { flag, dst, a } => {
            let (flag, dst, a) = (sl(slots, flag), sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                let s = ctx.arena.get(as_str(ctx.regs[a]));
                match s.trim_ascii().parse::<i64>() {
                    Ok(v) => {
                        ctx.regs[flag] = RegVal::I1(true);
                        ctx.regs[dst] = RegVal::I64(v);
                    }
                    Err(_) => {
                        ctx.regs[flag] = RegVal::I1(false);
                        ctx.regs[dst] = RegVal::I64(0);
                    }
                }
                Ok(())
            })
        }
        Inst::StofOpt { flag, dst, a } => {
            let (flag, dst, a) = (sl(slots, flag), sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                let s = ctx.arena.get(as_str(ctx.regs[a]));
                match s.trim_ascii().parse::<f64>() {
                    Ok(v) => {
                        ctx.regs[flag] = RegVal::I1(true);
                        ctx.regs[dst] = RegVal::F64(v);
                    }
                    Err(_) => {
                        ctx.regs[flag] = RegVal::I1(false);
                        ctx.regs[dst] = RegVal::F64(0.0);
                    }
                }
                Ok(())
            })
        }
        Inst::Round2f { trunc, dst, a, n } => {
            let (dst, a, n) = (sl(slots, dst), sl(slots, a), sl(slots, n));
            let f = if trunc {
                trunc_prec_f64
            } else {
                round_prec_f64
            };
            Box::new(move |ctx| {
                ctx.regs[dst] = RegVal::F64(f(as_f64(ctx.regs[a]), as_i64(ctx.regs[n])));
                Ok(())
            })
        }
        Inst::Round2i { trunc, dst, a, n } => {
            let (dst, a, n) = (sl(slots, dst), sl(slots, a), sl(slots, n));
            let f = if trunc {
                trunc_prec_i64
            } else {
                round_prec_i64
            };
            Box::new(move |ctx| {
                ctx.regs[dst] = RegVal::I64(f(as_i64(ctx.regs[a]), as_i64(ctx.regs[n])));
                Ok(())
            })
        }
        Inst::Slike { ci, dst, a, p, esc } => {
            let (dst, a, p) = (sl(slots, dst), sl(slots, a), sl(slots, p));
            let esc = esc.map(|e| sl(slots, e));
            Box::new(move |ctx| {
                let e = match esc {
                    None => None,
                    Some(e) => like_escape_of(ctx.arena.get(as_str(ctx.regs[e])))?,
                };
                let (sr, pr) = (as_str(ctx.regs[a]), as_str(ctx.regs[p]));
                let (sr, pr) = if ci {
                    // ILIKE: fold BOTH sides with the measured simple
                    // casemap (== DuckDB lower(); generic vectorized path.
                    // Known divergence, documented in the pins: DuckDB
                    // swaps an ASCII-only fold when column STATS are pure
                    // ASCII — K/U+0130 in the pattern are the entire
                    // observable surface).
                    (
                        ctx.arena.case_map(sr, super::casemap::simple_lower),
                        ctx.arena.case_map(pr, super::casemap::simple_lower),
                    )
                } else {
                    (sr, pr)
                };
                // Two immutable reads after any folding appends.
                let ok = {
                    let sv = ctx.arena.get(sr).as_bytes();
                    let pv = ctx.arena.get(pr).as_bytes();
                    like_match(sv, pv, e)?
                };
                ctx.regs[dst] = RegVal::I1(ok);
                Ok(())
            })
        }
        Inst::Str2 { op, dst, a, b } => {
            let (dst, a, b) = (sl(slots, dst), sl(slots, a), sl(slots, b));
            match op {
                StrOp2::Find => Box::new(move |ctx| {
                    let s = ctx.arena.get(as_str(ctx.regs[a]));
                    let n = ctx.arena.get(as_str(ctx.regs[b]));
                    ctx.regs[dst] = RegVal::I64(str_find(s, n));
                    Ok(())
                }),
                StrOp2::Levenshtein => Box::new(move |ctx| {
                    let s = ctx.arena.get(as_str(ctx.regs[a])).as_bytes();
                    let n = ctx.arena.get(as_str(ctx.regs[b])).as_bytes();
                    ctx.regs[dst] = RegVal::I64(duck_levenshtein(s, n));
                    Ok(())
                }),
                StrOp2::Damerau => Box::new(move |ctx| {
                    let s = ctx.arena.get(as_str(ctx.regs[a])).as_bytes();
                    let n = ctx.arena.get(as_str(ctx.regs[b])).as_bytes();
                    ctx.regs[dst] = RegVal::I64(duck_damerau(s, n));
                    Ok(())
                }),
                StrOp2::Jaccard => Box::new(move |ctx| {
                    let s = ctx.arena.get(as_str(ctx.regs[a])).as_bytes();
                    let n = ctx.arena.get(as_str(ctx.regs[b])).as_bytes();
                    ctx.regs[dst] = RegVal::F64(duck_jaccard(s, n)?);
                    Ok(())
                }),
                StrOp2::Hamming => Box::new(move |ctx| {
                    let s = ctx.arena.get(as_str(ctx.regs[a])).as_bytes();
                    let n = ctx.arena.get(as_str(ctx.regs[b])).as_bytes();
                    ctx.regs[dst] = RegVal::I64(duck_hamming(s, n)?);
                    Ok(())
                }),
                op => Box::new(move |ctx| {
                    let s = ctx.arena.get(as_str(ctx.regs[a]));
                    let n = ctx.arena.get(as_str(ctx.regs[b]));
                    ctx.regs[dst] = RegVal::I1(str_pred(op, s, n));
                    Ok(())
                }),
            }
        }
        Inst::Str3 { op, dst, a, b, c } => {
            let (dst, a, b, c) = (sl(slots, dst), sl(slots, a), sl(slots, b), sl(slots, c));
            Box::new(move |ctx| {
                let out = {
                    let s = ctx.arena.get(as_str(ctx.regs[a]));
                    let x = ctx.arena.get(as_str(ctx.regs[b]));
                    let y = ctx.arena.get(as_str(ctx.regs[c]));
                    match op {
                        ir::StrOp3::Replace => duck_replace(s, x, y),
                        ir::StrOp3::Translate => duck_translate(s, x, y),
                    }
                };
                ctx.regs[dst] = RegVal::Str(ctx.arena.push_str(&out));
                Ok(())
            })
        }
        Inst::Str2i { op, dst, a, n } => {
            let (dst, a, n) = (sl(slots, dst), sl(slots, a), sl(slots, n));
            match op {
                ir::StrOp2i::Repeat => Box::new(move |ctx| {
                    let out = duck_repeat(
                        ctx.arena.get(as_str(ctx.regs[a])),
                        as_i64(ctx.regs[n]),
                    )?;
                    ctx.regs[dst] = RegVal::Str(ctx.arena.push_str(&out));
                    Ok(())
                }),
                ir::StrOp2i::Extract => Box::new(move |ctx| {
                    let i = as_i64(ctx.regs[n]);
                    // Same +-2^32 window and trap as substr (measured:
                    // pins-wave5/subscripts-extended.json); NULL rows never
                    // reach here, matching DuckDB's NULL-skips-the-check.
                    if !substr_range_ok(i) {
                        return Err(Trap(
                            "substring offset outside of supported range".to_string(),
                        ));
                    }
                    let sref = as_str(ctx.regs[a]);
                    let rng = extract_window(ctx.arena.get(sref), i);
                    // The extracted char is a subview of the input span.
                    ctx.regs[dst] = RegVal::Str(StrRef {
                        off: sref.off + rng.start,
                        len: rng.end - rng.start,
                    });
                    Ok(())
                }),
            }
        }
        Inst::Spad {
            left,
            dst,
            a,
            len,
            pad,
        } => {
            let (dst, a, len, pad) = (sl(slots, dst), sl(slots, a), sl(slots, len), sl(slots, pad));
            Box::new(move |ctx| {
                let out = duck_pad(
                    left,
                    ctx.arena.get(as_str(ctx.regs[a])),
                    as_i64(ctx.regs[len]),
                    ctx.arena.get(as_str(ctx.regs[pad])),
                )?;
                ctx.regs[dst] = RegVal::Str(ctx.arena.push_str(&out));
                Ok(())
            })
        }
        Inst::Sslice { dst, a, lo, hi } => {
            let (dst, a, lo, hi) = (sl(slots, dst), sl(slots, a), sl(slots, lo), sl(slots, hi));
            Box::new(move |ctx| {
                let sref = as_str(ctx.regs[a]);
                let rng = slice_window(
                    ctx.arena.get(sref),
                    as_i64(ctx.regs[lo]),
                    as_i64(ctx.regs[hi]),
                );
                // The slice is a subview of the input span — no copy.
                ctx.regs[dst] = RegVal::Str(StrRef {
                    off: sref.off + rng.start,
                    len: rng.end - rng.start,
                });
                Ok(())
            })
        }
        Inst::Sord {
            empty_zero,
            dst,
            a,
        } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                let s = ctx.arena.get(as_str(ctx.regs[a]));
                ctx.regs[dst] = RegVal::I64(duck_ord(s, empty_zero));
                Ok(())
            })
        }
        Inst::SLen { bytes, dst, a } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                let s = ctx.arena.get(as_str(ctx.regs[a]));
                let v = if bytes {
                    s.len() as i64
                } else {
                    s.chars().count() as i64
                };
                ctx.regs[dst] = RegVal::I64(v);
                Ok(())
            })
        }
        Inst::Sconcat { dst, a, b } => {
            let (dst, a, b) = (sl(slots, dst), sl(slots, a), sl(slots, b));
            Box::new(move |ctx| {
                let (x, y) = (as_str(ctx.regs[a]), as_str(ctx.regs[b]));
                ctx.regs[dst] = RegVal::Str(ctx.arena.concat(x, y));
                Ok(())
            })
        }
        Inst::Str1 { op, dst, a } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            match op {
                // DuckDB uses SIMPLE (1:1) case maps; casemap.rs carries
                // the measured exception table over Rust's full maps.
                StrOp1::Upper | StrOp1::Lower => {
                    let map: fn(char) -> char = match op {
                        StrOp1::Upper => super::casemap::simple_upper,
                        _ => super::casemap::simple_lower,
                    };
                    Box::new(move |ctx| {
                        ctx.regs[dst] = RegVal::Str(ctx.arena.case_map(as_str(ctx.regs[a]), map));
                        Ok(())
                    })
                }
                StrOp1::StripAccents => Box::new(move |ctx| {
                    let sref = as_str(ctx.regs[a]);
                    let out = duck_strip_accents(ctx.arena.get(sref));
                    ctx.regs[dst] = match out {
                        None => RegVal::Str(sref), // ASCII fast path: verbatim
                        Some(s) => RegVal::Str(ctx.arena.push_str(&s)),
                    };
                    Ok(())
                }),
                StrOp1::Reverse => Box::new(move |ctx| {
                    let sref = as_str(ctx.regs[a]);
                    let out = duck_reverse(ctx.arena.get(sref));
                    ctx.regs[dst] = RegVal::Str(ctx.arena.push_str(&out));
                    Ok(())
                }),
            }
        }
        Inst::Strim {
            side,
            dst,
            a,
            chars,
        } => {
            let (dst, a, chars) = (sl(slots, dst), sl(slots, a), sl(slots, chars));
            Box::new(move |ctx| {
                let sref = as_str(ctx.regs[a]);
                let rng = trim_bounds(
                    ctx.arena.get(sref),
                    ctx.arena.get(as_str(ctx.regs[chars])),
                    side,
                );
                // The trimmed value is a subview of the input span — no copy.
                ctx.regs[dst] = RegVal::Str(StrRef {
                    off: sref.off + rng.start,
                    len: rng.end - rng.start,
                });
                Ok(())
            })
        }
        Inst::Ssubstr { dst, a, start, len } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            let start = sl(slots, start);
            let len = len.map(|l| sl(slots, l));
            Box::new(move |ctx| {
                let st = as_i64(ctx.regs[start]);
                if !substr_range_ok(st) {
                    return Err(Trap(
                        "substring offset outside of supported range".to_string(),
                    ));
                }
                let ln = match len {
                    Some(l) => {
                        let v = as_i64(ctx.regs[l]);
                        if !substr_range_ok(v) {
                            return Err(Trap(
                                "substring length outside of supported range".to_string(),
                            ));
                        }
                        Some(v)
                    }
                    None => None,
                };
                let sref = as_str(ctx.regs[a]);
                let rng = substr_window(ctx.arena.get(sref), st, ln);
                // The window is a subview of the input span — no copy.
                ctx.regs[dst] = RegVal::Str(StrRef {
                    off: sref.off + rng.start,
                    len: rng.end - rng.start,
                });
                Ok(())
            })
        }
        Inst::Num1 { op, dst, a } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            match op {
                NumOp1::Iabs => Box::new(move |ctx| {
                    let x = as_i64(ctx.regs[a]);
                    match x.checked_abs() {
                        Some(v) => {
                            ctx.regs[dst] = RegVal::I64(v);
                            Ok(())
                        }
                        None => Err(Trap(abs_overflow_msg(x))),
                    }
                }),
                NumOp1::Fabs => Box::new(move |ctx| {
                    ctx.regs[dst] = RegVal::F64(as_f64(ctx.regs[a]).abs());
                    Ok(())
                }),
                NumOp1::Fround => Box::new(move |ctx| {
                    ctx.regs[dst] = RegVal::F64(as_f64(ctx.regs[a]).round());
                    Ok(())
                }),
                op => {
                    let f = math1_fn(op);
                    Box::new(move |ctx| {
                        ctx.regs[dst] = RegVal::F64(f(as_f64(ctx.regs[a]))?);
                        Ok(())
                    })
                }
            }
        }
        Inst::Load { dst, col } => {
            let (dst, col) = (sl(slots, dst), col as usize);
            Box::new(move |ctx| {
                ctx.regs[dst] = load_payload(&ctx.input.cols[col], ctx.row, ctx.arena);
                Ok(())
            })
        }
        Inst::LoadOpt { flag, dst, col } => {
            let (flag, dst, col) = (sl(slots, flag), sl(slots, dst), col as usize);
            let ty = p.in_cols[col].ty.ty;
            Box::new(move |ctx| {
                let c = &ctx.input.cols[col];
                let valid = col_valid(c, ctx.row);
                ctx.regs[flag] = RegVal::I1(valid);
                ctx.regs[dst] = if valid {
                    load_payload(c, ctx.row, ctx.arena)
                } else {
                    default_reg(ty)
                };
                Ok(())
            })
        }
        Inst::Store { col, val } => {
            let (col, val) = (col as usize, sl(slots, val));
            Box::new(move |ctx| {
                push_out(&mut ctx.out[col], true, ctx.regs[val]);
                Ok(())
            })
        }
        Inst::StoreOpt { col, flag, val } => {
            let (col, flag, val) = (col as usize, sl(slots, flag), sl(slots, val));
            // Spec: on a false flag the stored payload is the type default —
            // never the live register (adversarial finding).
            let default = default_reg(p.out_cols[col].ty.ty);
            Box::new(move |ctx| {
                let valid = as_i1(ctx.regs[flag]);
                let v = if valid { ctx.regs[val] } else { default };
                push_out(&mut ctx.out[col], valid, v);
                Ok(())
            })
        }
        Inst::Probe {
            static_id,
            hit,
            dsts,
            keys,
        } => {
            let static_id = static_id as usize;
            let hit = sl(slots, hit);
            let dsts: Vec<usize> = dsts.iter().map(|d| sl(slots, *d)).collect();
            let keys: Vec<usize> = keys.iter().map(|k| sl(slots, *k)).collect();
            let value_tys: Vec<Ty> = match &p.statics[static_id] {
                StaticTy::Map { values, .. } => values.clone(),
                _ => unreachable!("probe on non-map is rejected by the verifier"),
            };
            Box::new(move |ctx| {
                let PreparedStatic::Map { entries } = &ctx.statics[static_id] else {
                    unreachable!("static kind checked at compile");
                };
                let found = entries
                    .binary_search_by(|(k, _)| cmp_key(k, &keys, ctx))
                    .ok();
                match found {
                    Some(idx) => {
                        ctx.regs[hit] = RegVal::I1(true);
                        // scalar_to_reg may append to the arena for str
                        // values — fine, spans stay valid, it only grows.
                        for (di, v) in dsts.iter().zip(entries[idx].1.iter()) {
                            ctx.regs[*di] = scalar_to_reg(v, ctx.arena);
                        }
                    }
                    None => {
                        ctx.regs[hit] = RegVal::I1(false);
                        for (di, ty) in dsts.iter().zip(value_tys.iter()) {
                            ctx.regs[*di] = default_reg(*ty);
                        }
                    }
                }
                Ok(())
            })
        }
        Inst::ProbeRange {
            static_id,
            start,
            end,
            keys,
        } => {
            let static_id = static_id as usize;
            let (start, end) = (sl(slots, start), sl(slots, end));
            let keys: Vec<usize> = keys.iter().map(|k| sl(slots, *k)).collect();
            let is_batch = matches!(&p.statics[static_id], StaticTy::BatchMap { .. });
            Box::new(move |ctx| {
                let (lo, hi) = if is_batch {
                    (0, ctx.batch_rows.len())
                } else {
                    let PreparedStatic::MultiMap { entries } = &ctx.statics[static_id] else {
                        unreachable!("static kind checked at compile");
                    };
                    if keys.is_empty() {
                        (0, entries.len())
                    } else {
                        let lo = entries.partition_point(|(k, _)| {
                            cmp_key(k, &keys, ctx) == std::cmp::Ordering::Less
                        });
                        let hi = entries.partition_point(|(k, _)| {
                            cmp_key(k, &keys, ctx) != std::cmp::Ordering::Greater
                        });
                        (lo, hi)
                    }
                };
                ctx.regs[start] = RegVal::I64(lo as i64);
                ctx.regs[end] = RegVal::I64(hi as i64);
                Ok(())
            })
        }
        Inst::ProbeRead {
            static_id,
            idx,
            dsts,
        } => {
            let static_id = static_id as usize;
            let idx = sl(slots, idx);
            let dsts: Vec<usize> = dsts.iter().map(|d| sl(slots, *d)).collect();
            let is_batch = matches!(&p.statics[static_id], StaticTy::BatchMap { .. });
            Box::new(move |ctx| {
                let i = as_i64(ctx.regs[idx]) as usize;
                if is_batch {
                    let Some(row) = ctx.batch_rows.get(i) else {
                        return Err(Trap("probe.read index out of range (lowering bug)".into()));
                    };
                    for (di, v) in dsts.iter().zip(row.iter()) {
                        ctx.regs[*di] = scalar_to_reg(v, ctx.arena);
                    }
                    return Ok(());
                }
                let PreparedStatic::MultiMap { entries } = &ctx.statics[static_id] else {
                    unreachable!("static kind checked at compile");
                };
                let Some(row) = entries.get(i) else {
                    return Err(Trap("probe.read index out of range (lowering bug)".into()));
                };
                for (di, v) in dsts.iter().zip(row.1.iter()) {
                    ctx.regs[*di] = scalar_to_reg(v, ctx.arena);
                }
                Ok(())
            })
        }
        Inst::Sload { static_id, dst } => {
            let (static_id, dst) = (static_id as usize, sl(slots, dst));
            Box::new(move |ctx| {
                let PreparedStatic::Scalar { val, .. } = &ctx.statics[static_id] else {
                    unreachable!("static kind checked at compile");
                };
                ctx.regs[dst] = scalar_to_reg(val, ctx.arena);
                Ok(())
            })
        }
        Inst::SloadOpt {
            static_id,
            flag,
            dst,
        } => {
            let (static_id, flag, dst) = (static_id as usize, sl(slots, flag), sl(slots, dst));
            let ty = match &p.statics[static_id] {
                StaticTy::Scalar(ct) => ct.ty,
                _ => unreachable!("sload on non-scalar is rejected by the verifier"),
            };
            Box::new(move |ctx| {
                let PreparedStatic::Scalar { valid, val } = &ctx.statics[static_id] else {
                    unreachable!("static kind checked at compile");
                };
                ctx.regs[flag] = RegVal::I1(*valid);
                ctx.regs[dst] = if *valid {
                    scalar_to_reg(val, ctx.arena)
                } else {
                    default_reg(ty)
                };
                Ok(())
            })
        }
    }
}

/// Compare a stored key tuple against the probe registers, position-wise —
/// the search-time mirror of `KeyBits: Ord` (which the entries were sorted
/// by), so the binary search stays allocation-free.
fn cmp_key(stored: &[KeyBits], key_regs: &[usize], ctx: &Ctx<'_>) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    for (kb, reg) in stored.iter().zip(key_regs.iter()) {
        let ord = match (kb, ctx.regs[*reg]) {
            (KeyBits::I1(s), RegVal::I1(v)) => s.cmp(&v),
            (KeyBits::I64(s), RegVal::I64(v)) => s.cmp(&v),
            (KeyBits::F64(s), RegVal::F64(v)) => s.cmp(&super::canon_f64_bits(v)),
            (KeyBits::Str(s), RegVal::Str(v)) => s.as_str().cmp(ctx.arena.get(v)),
            _ => unreachable!("probe key types checked at compile"),
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

pub(super) fn apply_ord(pred: CmpPred, ord: std::cmp::Ordering) -> bool {
    use std::cmp::Ordering::*;
    match pred {
        CmpPred::Eq => ord == Equal,
        CmpPred::Ne => ord != Equal,
        CmpPred::Lt => ord == Less,
        CmpPred::Le => ord != Greater,
        CmpPred::Gt => ord == Greater,
        CmpPred::Ge => ord != Less,
    }
}

pub(super) fn valid_len(c: &ColData) -> usize {
    match c {
        ColData::I1 { valid, .. }
        | ColData::I64 { valid, .. }
        | ColData::F64 { valid, .. }
        | ColData::Str { valid, .. } => valid.len(),
    }
}

/// Only called for nullable columns, whose validity length check_input has
/// already enforced — the unwrap_or is unreachable there and don't-care for
/// non-nullable columns (which never reach this).
pub(super) fn col_valid(c: &ColData, row: usize) -> bool {
    match c {
        ColData::I1 { valid, .. }
        | ColData::I64 { valid, .. }
        | ColData::F64 { valid, .. }
        | ColData::Str { valid, .. } => valid.get(row).copied().unwrap_or(true),
    }
}

fn load_payload(c: &ColData, row: usize, arena: &mut Arena) -> RegVal {
    match c {
        ColData::I1 { data, .. } => RegVal::I1(data[row]),
        ColData::I64 { data, .. } => RegVal::I64(data[row]),
        ColData::F64 { data, .. } => RegVal::F64(data[row]),
        c @ ColData::Str { .. } => RegVal::Str(arena.push_str(c.str_at(row))),
    }
}

fn push_out(col: &mut OutCol, valid: bool, v: RegVal) {
    match (col, v) {
        (OutCol::I1(vec), RegVal::I1(b)) => vec.push((valid, b)),
        (OutCol::I64(vec), RegVal::I64(i)) => vec.push((valid, i)),
        (OutCol::F64(vec), RegVal::F64(f)) => vec.push((valid, f)),
        (OutCol::Str(vec), RegVal::Str(s)) => vec.push((valid, s)),
        _ => unreachable!("store type checked by the verifier"),
    }
}
