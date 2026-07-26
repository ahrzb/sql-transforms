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
//! * `fcmp` is IEEE-ordered: every predicate involving NaN is false, except
//!   `ne`, which is `!(a == b)` and therefore true. SQL NULL/NaN policy is
//!   the lowering's job, expressed with flags around these primitives.
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
use std::fmt::Write as _;

use super::super::ir::verify::{verify, VerifyError};
use super::super::ir::{
    self, BinOp, CmpPred, Inst, NumOp1, Program, RoundMode, StaticTy, StrOp1, Term, TrimSide, Ty,
    Value,
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
        }
    }
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
    row: usize,
}

enum PreparedStatic {
    Scalar {
        valid: bool,
        val: ScalarVal,
    },
    /// Sorted by key; probed by allocation-free binary search.
    Map {
        entries: Vec<(Vec<KeyBits>, Vec<ScalarVal>)>,
    },
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
    Skip,
    Trap(String),
}

pub struct InterpFn {
    blocks: Vec<CBlock>,
    nregs: usize,
    statics: Vec<PreparedStatic>,
    in_decl: Vec<(Ty, bool)>,
    out_decl: Vec<Ty>,
}

pub fn compile(p: &Program, statics: Vec<StaticData>) -> Result<InterpFn, CompileError> {
    verify(p).map_err(CompileError::Verify)?;
    let prepared = prepare_statics(p, statics)?;

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
            insts.push(compile_inst(p, inst, &slots));
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

        let mut emitted = 0usize;
        for row in 0..input.rows {
            let mut ctx = Ctx {
                regs: &mut st.regs,
                arena: &mut st.arena,
                out: &mut st.out,
                input,
                statics: &self.statics,
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
                    CTerm::Skip => break,
                    CTerm::Trap(msg) => return Err(Trap(msg.clone())),
                }
            }
        }
        st.emitted = emitted;
        Ok(())
    }

    /// A `RunState` is only valid for the `InterpFn` that created it —
    /// reject a foreign one with a trap instead of an index panic.
    fn check_state(&self, st: &RunState) -> Result<(), Trap> {
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

    fn check_input(&self, input: &Batch) -> Result<(), Trap> {
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

fn reserve_out(out: &mut [OutCol], rows: usize) {
    for col in out {
        match col {
            OutCol::I1(v) => v.reserve(rows),
            OutCol::I64(v) => v.reserve(rows),
            OutCol::F64(v) => v.reserve(rows),
            OutCol::Str(v) => v.reserve(rows),
        }
    }
}

fn prepare_statics(
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
        Term::Emit => CTerm::Emit,
        Term::Skip => CTerm::Skip,
        Term::Trap { msg } => CTerm::Trap(msg.clone()),
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

/// DuckDB's substr window arithmetic (measured 1.5.5), on codepoints — NOT
/// grapheme clusters (substr slices inside ZWJ emoji). 1-based virtual
/// positions: negative start counts from the end (`start = n + start + 1`),
/// start <= 0 consumes length before character 1, negative len is "". A
/// missing SQL length arrives as i64::MAX; the saturating add makes that
/// "rest of the string".
fn substr_window(s: &str, start: i64, len: i64) -> String {
    if len < 0 {
        return String::new();
    }
    let n = s.chars().count() as i64;
    let start = if start < 0 { n + start + 1 } else { start };
    let end = start.saturating_add(len);
    let (lo, hi) = (start.max(1), end.min(n + 1));
    if hi <= lo {
        return String::new();
    }
    s.chars()
        .skip((lo - 1) as usize)
        .take((hi - lo) as usize)
        .collect()
}

fn fmt_into_arena(arena: &mut Arena, args: std::fmt::Arguments<'_>) -> StrRef {
    let off = arena.0.len();
    let _ = ArenaWriter(&mut arena.0).write_fmt(args);
    StrRef {
        off,
        len: arena.0.len() - off,
    }
}

/// Value id -> dense register slot.
fn sl(slots: &HashMap<u32, u32>, v: Value) -> usize {
    slots[&v.0] as usize
}

fn compile_inst(p: &Program, inst: &Inst, slots: &HashMap<u32, u32>) -> InstFn {
    match inst.clone() {
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
                        None => Err(Trap(format!("i64 overflow in {}", op.name()))),
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
                        None => Err(Trap(format!("i64 overflow in {}", op.name()))),
                    }
                }),
                BinOp::Fadd | BinOp::Fsub | BinOp::Fmul | BinOp::Fdiv | BinOp::Frem => {
                    Box::new(move |ctx| {
                        let (x, y) = (as_f64(ctx.regs[a]), as_f64(ctx.regs[b]));
                        let v = match op {
                            BinOp::Fadd => x + y,
                            BinOp::Fsub => x - y,
                            BinOp::Fmul => x * y,
                            BinOp::Fdiv => x / y,
                            _ => x % y,
                        };
                        ctx.regs[dst] = RegVal::F64(v);
                        Ok(())
                    })
                }
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
                ctx.regs[dst] = RegVal::Str(fmt_into_arena(ctx.arena, format_args!("{v}")));
                Ok(())
            })
        }
        Inst::Ftos { dst, a } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            Box::new(move |ctx| {
                let v = as_f64(ctx.regs[a]);
                ctx.regs[dst] = RegVal::Str(fmt_into_arena(ctx.arena, format_args!("{v:?}")));
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
            Box::new(move |ctx| {
                let s = ctx.arena.get(as_str(ctx.regs[a]));
                let mut out = String::with_capacity(s.len());
                for c in s.chars() {
                    // DuckDB uses SIMPLE (1:1) case maps; Rust std only has
                    // full maps. Take the full map iff it is 1:1, else keep
                    // the char — exact on ASCII, diverges on ß/İ (see the
                    // 2026-07-26 builtin-pins spec; xfail'd differentially).
                    let mapped = match op {
                        StrOp1::Upper => {
                            let mut it = c.to_uppercase();
                            (it.next().unwrap(), it.next().is_none())
                        }
                        StrOp1::Lower => {
                            let mut it = c.to_lowercase();
                            (it.next().unwrap(), it.next().is_none())
                        }
                    };
                    out.push(if mapped.1 { mapped.0 } else { c });
                }
                ctx.regs[dst] = RegVal::Str(ctx.arena.push_str(&out));
                Ok(())
            })
        }
        Inst::Strim {
            side,
            dst,
            a,
            chars,
        } => {
            let (dst, a, chars) = (sl(slots, dst), sl(slots, a), sl(slots, chars));
            Box::new(move |ctx| {
                let set: Vec<char> = ctx.arena.get(as_str(ctx.regs[chars])).chars().collect();
                let s = ctx.arena.get(as_str(ctx.regs[a]));
                let hit = |c: char| set.contains(&c);
                let t = match side {
                    TrimSide::Both => s.trim_matches(hit),
                    TrimSide::Lead => s.trim_start_matches(hit),
                    TrimSide::Trail => s.trim_end_matches(hit),
                }
                .to_owned();
                ctx.regs[dst] = RegVal::Str(ctx.arena.push_str(&t));
                Ok(())
            })
        }
        Inst::Ssubstr { dst, a, start, len } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            let (start, len) = (sl(slots, start), sl(slots, len));
            Box::new(move |ctx| {
                let (st, ln) = (as_i64(ctx.regs[start]), as_i64(ctx.regs[len]));
                let s = ctx.arena.get(as_str(ctx.regs[a]));
                let out = substr_window(s, st, ln);
                ctx.regs[dst] = RegVal::Str(ctx.arena.push_str(&out));
                Ok(())
            })
        }
        Inst::Num1 { op, dst, a } => {
            let (dst, a) = (sl(slots, dst), sl(slots, a));
            match op {
                NumOp1::Iabs => Box::new(move |ctx| match as_i64(ctx.regs[a]).checked_abs() {
                    Some(v) => {
                        ctx.regs[dst] = RegVal::I64(v);
                        Ok(())
                    }
                    None => Err(Trap("i64 overflow in iabs".to_string())),
                }),
                NumOp1::Fabs => Box::new(move |ctx| {
                    ctx.regs[dst] = RegVal::F64(as_f64(ctx.regs[a]).abs());
                    Ok(())
                }),
                NumOp1::Fround => Box::new(move |ctx| {
                    ctx.regs[dst] = RegVal::F64(as_f64(ctx.regs[a]).round());
                    Ok(())
                }),
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

fn apply_ord(pred: CmpPred, ord: std::cmp::Ordering) -> bool {
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

fn valid_len(c: &ColData) -> usize {
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
fn col_valid(c: &ColData, row: usize) -> bool {
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
        ColData::Str { data, .. } => RegVal::Str(arena.push_str(&data[row])),
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
