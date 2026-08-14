//! The closure-compiled interpreter backend — the oracle. One pre-traversal
//! of a VERIFIED program builds a vector of instruction closures per block;
//! execution is plain dispatch. Never optimized: correctness and coverage
//! over speed, and it stays that way (design doc §7).
//!
//! # What this module is, and is not
//!
//! Two things live here: the COMPILE front half (verify, prepare statics,
//! compile regexes) which both backends run, and the EVAL LOOP, which only
//! this backend runs. The SQL semantics themselves are not here — they are
//! in [`super::kernels`], shared: the interpreter calls those kernels and
//! the Cranelift backend emits calls to the same ones.
//!
//! So `cranelift::compile_ext` calling `interp::compile_ext` is not the JIT
//! falling back on an interpreter — it is both backends running the same
//! front half, and the JIT then reading the prepared statics and extern
//! impls this compile produced.
//!
//! The eval loop is genuinely a second implementation of the same 40
//! instructions, and it is load-bearing for one reason: `shape="many"`.
//! Cranelift refuses multiplicity programs (EmitTo loops, multimap probes)
//! up front, so join multiplicity runs HERE and nowhere else. It is also
//! the differential oracle for codegen bugs and the bench control
//! (`SPECIALIZER_FORCE_INTERP=1`).
//!
//! What it is NOT, despite what the old comments claimed, is a coverage
//! fallback for instructions Cranelift cannot lower: that dispatch is an
//! exhaustive `match` over `Inst`, so a new instruction cannot compile
//! without a Cranelift arm. Measured 2026-08-14: of 137 fuzz-generated
//! programs that built, 137 chose Cranelift.
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
    Ty, Value,
};
use super::kernels::*;
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
    /// Extern (UDF) implementations, one per program `extern @N`.
    externs: &'a [super::ExternImpl],
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
    /// A fitted ensemble; the packed layout is the kernel's own.
    Model(Box<super::tree_ensemble::TreeEnsemble>),
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
    externs: Vec<super::ExternImpl>,
    in_decl: Vec<(Ty, bool)>,
    out_decl: Vec<Ty>,
    /// True when a batchmap static exists: `run` flattens the batch's
    /// rows before the row loop (stage-B self-joins).
    has_batch_map: bool,
}

pub fn compile(p: &Program, statics: Vec<StaticData>) -> Result<InterpFn, CompileError> {
    compile_ext(p, statics, Vec::new())
}

/// [`compile`] plus the extern (UDF) implementations — one per program
/// `extern @N`, name-checked against the declaration.
pub fn compile_ext(
    p: &Program,
    statics: Vec<StaticData>,
    externs: Vec<super::ExternImpl>,
) -> Result<InterpFn, CompileError> {
    verify(p).map_err(CompileError::Verify)?;
    if externs.len() != p.externs.len() {
        return Err(CompileError::Static(format!(
            "program declares {} extern(s), {} implementation(s) provided",
            p.externs.len(),
            externs.len()
        )));
    }
    for (i, (spec, imp)) in p.externs.iter().zip(&externs).enumerate() {
        if spec.name != imp.name {
            return Err(CompileError::Static(format!(
                "extern @{i} is declared '{}', implementation is named '{}'",
                spec.name, imp.name
            )));
        }
    }
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
        externs,
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
                    // Narrow out columns compute and land in the i64 lane;
                    // the width is applied at the arrow emit boundary.
                    Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => OutCol::I64(Vec::new()),
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
                externs: &self.externs,
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

    /// The extern implementations, shared with the Cranelift backend's
    /// `h_extern` helper.
    pub(super) fn externs(&self) -> &[super::ExternImpl] {
        &self.externs
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
            // Narrow declarations run in their lane; width applies at emit.
            if col_ty != ty.lane() {
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
            if col.ty() != ty.lane() {
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
                    Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => ScalarVal::I64(0),
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
            (StaticTy::Model { n_features }, StaticData::Model(m)) => {
                if m.n_features() != *n_features {
                    return Err(CompileError::Static(format!(
                        "@{i}: model takes {} feature(s), declared {n_features}",
                        m.n_features()
                    )));
                }
                prepared.push(PreparedStatic::Model(m));
            }
            (StaticTy::Model { .. }, _) => {
                return Err(CompileError::Static(format!(
                    "@{i}: declared model, got non-model data"
                )))
            }
            (_, StaticData::Model(_)) => {
                return Err(CompileError::Static(format!(
                    "@{i}: got model data for a non-model declaration"
                )))
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
        Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => RegVal::I64(0),
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










// ------------------------------------------------- wave-1 math semantics --
// One shared fn per op, used verbatim by BOTH backends (pins:
// docs/superpowers/specs/2026-07-26-wave1-builtin-pins.md). Trap messages
// are DuckDB 1.5.5's own, measured — including its typo.









// ------------------------------------------------------ LIKE (wave 2) --
// Byte-based matcher with codepoint `_`, reproducing every DuckDB 1.5.5
// pin including the DATA-DEPENDENT dangling-escape error (raised only
// when the matcher examines a trailing escape while string bytes remain;
// plain false when the string is exhausted). Iterative two-pointer
// restart: identical booleans and identical error rows to the leftmost-
// first recursive semantics, never DuckDB's own O(n^k) blowup (measured
// 23s/row there on pathological patterns; ours is O(n*m)).
// Pins: docs/superpowers/specs/pins-wave1/pins_like.json.




// ------------------------------------------ wave-3 builtins (TASK-49) --
// One shared fn per op, used verbatim by BOTH backends. Pins:
// docs/superpowers/specs/2026-07-26-wave3-builtin-pins.md — all
// similarity ops are raw UTF-8 BYTE-based; error texts are DuckDB's own.



























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
                    Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => {
                        apply_ord(pred, as_i64(ctx.regs[a]).cmp(&as_i64(ctx.regs[b])))
                    }
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
        Inst::Itof { narrow, dst, a } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                let n = as_i64(ctx.regs[a]);
                // `n as f32 as f64` is NOT `n as f64` above 2**53 — one
                // rounding versus two. That is the point (TASK-77).
                let v = if narrow { n as f32 as f64 } else { n as f64 };
                ctx.regs[dst] = RegVal::F64(v);
                Ok(())
            })
        }
        Inst::Ftoi { mode, dst, a } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                let x = as_f64(ctx.regs[a]);
                let r = match mode {
                    RoundMode::Trunc => x.trunc(),
                    // half to EVEN — DuckDB's cast, not the round() builtin
                    RoundMode::Nearest => x.round_ties_even(),
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
        Inst::Predict {
            static_id,
            dst,
            id,
            feats,
        } => {
            let static_id = static_id as usize;
            let dst = sl(slots, dst);
            let id = sl(slots, id);
            let feats: Vec<usize> = feats.iter().map(|f| sl(slots, *f)).collect();
            // Per-site scratch so the kernel gets a contiguous &[f64] without
            // a per-row allocation. `InstFn` is `Fn`, not `FnMut`, so the
            // buffer lives behind a RefCell; it reaches its final capacity on
            // the first row and never grows again, which is the zero-alloc
            // contract RunState documents. (Cranelift uses a stack slot
            // instead — same routine, no cell.)
            let scratch = std::cell::RefCell::new(vec![0.0f64; feats.len()]);
            Box::new(move |ctx| {
                let PreparedStatic::Model(m) = &ctx.statics[static_id] else {
                    unreachable!("static kind checked at compile");
                };
                let mut buf = scratch.borrow_mut();
                for (j, f) in feats.iter().enumerate() {
                    buf[j] = as_f64(ctx.regs[*f]);
                }
                let v = m.predict(as_i64(ctx.regs[id]), &buf)?;
                ctx.regs[dst] = RegVal::F64(v);
                Ok(())
            })
        }
        Inst::ExternCall { ext, dsts, args } => {
            let ei = ext as usize;
            let dsts: Vec<usize> = dsts.iter().map(|d| sl(slots, *d)).collect();
            let args: Vec<usize> = args.iter().map(|a| sl(slots, *a)).collect();
            let param_tys = p.externs[ei].params.clone();
            let ret_tys = p.externs[ei].rets.clone();
            Box::new(move |ctx| {
                let mut a = Vec::with_capacity(param_tys.len());
                for (j, &ty) in param_tys.iter().enumerate() {
                    let valid = as_i1(ctx.regs[args[2 * j]]);
                    a.push(if valid {
                        Some(match ty {
                            Ty::I1 => ScalarVal::I1(as_i1(ctx.regs[args[2 * j + 1]])),
                            Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => {
                                ScalarVal::I64(as_i64(ctx.regs[args[2 * j + 1]]))
                            }
                            Ty::F64 => ScalarVal::F64(as_f64(ctx.regs[args[2 * j + 1]])),
                            Ty::Str => ScalarVal::Str(
                                ctx.arena.get(as_str(ctx.regs[args[2 * j + 1]])).to_string(),
                            ),
                        })
                    } else {
                        None
                    });
                }
                let (whole, outs) = call_extern(&ctx.externs[ei], &ret_tys, &a)?;
                ctx.regs[dsts[0]] = RegVal::I1(whole);
                for (j, (f, v)) in outs.iter().enumerate() {
                    ctx.regs[dsts[1 + 2 * j]] = RegVal::I1(*f);
                    ctx.regs[dsts[2 + 2 * j]] = scalar_to_reg(v, ctx.arena);
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
