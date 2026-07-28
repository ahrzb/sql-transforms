//! The columnar (batch-at-a-time) executor — the third backend. Mask-based
//! vectorization of the acyclic CFG: SSA values become full-batch lanes,
//! every block carries an active-row mask, and instructions run as masked
//! kernels over the whole batch. No new IR; semantics are the interpreter's,
//! bit for bit — every kernel calls the same `duck_*`/window/casemap helpers
//! `interp.rs` uses, and the differential test holds the two backends equal
//! on outputs, emitted counts, AND traps.
//!
//! # Scope (TASK-61)
//!
//! Stage-B multiplicity is row-path-only: programs containing `emit.to`,
//! multimap/batchmap statics, `probe.range`/`probe.read`, or any CFG cycle
//! are rejected at compile with [`CompileError::Static`]. Everything else is
//! covered. Without multiplicity each row emits at most once, so gathering
//! staged outputs in row order reproduces the row loop's output exactly.
//!
//! # Traps
//!
//! The row loop surfaces the trap of the SMALLEST row index that traps
//! anywhere (and, within that row, its earliest instruction). Columnar
//! kernels therefore never abort mid-batch: a trapping row records its
//! first message, is deactivated from the mask (it computes nothing
//! further), and execution continues; at the end the smallest recorded row
//! wins. Deactivation-on-first-trap makes the recorded message that row's
//! earliest, and topo order visits each row's path in row-execution order.

use std::collections::HashMap;

use super::super::ir::{
    BinOp, BlockId, ColTy, Inst, Lit, NumOp1, Program, RoundMode, StaticTy, StrOp1, StrOp2,
    StrOp2i, StrOp3, Term, Ty, Value,
};
use super::interp::{
    self, abs_overflow_msg, apply_ord, compile_regexes, duck_damerau, duck_fdiv, duck_fmod,
    duck_hamming, duck_jaccard, duck_levenshtein, duck_logb, duck_nextafter, duck_pad,
    duck_pow, duck_repeat, duck_replace, duck_reverse, duck_shl, duck_shr, duck_strip_accents,
    duck_translate, extract_window, like_escape_of, like_match, math1_fn, overflow_msg,
    round_prec_f64, round_prec_i64, slice_window, str_find, str_pred, substr_range_ok,
    substr_window, trim_bounds, trunc_prec_f64, trunc_prec_i64, CompileError, DuckF64, InterpFn,
    PreparedStatic,
};
use super::{
    canon_f64_bits, duck_fcmp, Arena, Batch, ColData, KeyBits, OutCol, RunState, ScalarVal,
    StaticData, StrRef, Trap,
};

/// One SSA value as a full-batch lane. Slots of deactivated/never-active
/// rows hold garbage (type defaults) — nothing downstream reads them.
enum VLane {
    I1(Vec<bool>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    Str(Vec<StrRef>),
}

pub struct ColumnarFn {
    /// Embedded interpreter compile: owns verification, prepared statics,
    /// input/state checks, and `new_state` — the columnar layer adds only
    /// the batch execution strategy on top.
    interp: InterpFn,
    p: Program,
    regexes: Vec<std::rc::Rc<regex::Regex>>,
    /// Value id -> dense lane slot (same idiom as the interpreter's
    /// register slots: sparse ids must not inflate the lane vector).
    slots: HashMap<u32, u32>,
    nslots: usize,
    /// Topological block order; edges are merged into a block before it
    /// runs, which is exactly what acyclicity buys.
    topo: Vec<usize>,
}

pub fn compile(p: &Program, statics: Vec<StaticData>) -> Result<ColumnarFn, CompileError> {
    // Verify + prepare statics + regexes exactly as the interpreter does.
    let interp = interp::compile(p, statics)?;

    // Stage-B multiplicity is row-path-only — reject it with a clear
    // message rather than approximating loop semantics in lanes.
    for (i, s) in p.statics.iter().enumerate() {
        if matches!(s, StaticTy::MultiMap { .. } | StaticTy::BatchMap { .. }) {
            return Err(CompileError::Static(format!(
                "columnar: static @{i} is a multimap/batchmap — stage-B multiplicity is \
                 row-path-only (use the interpreter or cranelift backend)"
            )));
        }
    }
    for (bi, b) in p.blocks.iter().enumerate() {
        if b.insts
            .iter()
            .any(|i| matches!(i, Inst::ProbeRange { .. } | Inst::ProbeRead { .. }))
        {
            return Err(CompileError::Static(format!(
                "columnar: block b{bi} uses probe.range/probe.read — stage-B multiplicity \
                 is row-path-only (use the interpreter or cranelift backend)"
            )));
        }
        if matches!(b.term, Term::EmitTo { .. }) {
            return Err(CompileError::Static(format!(
                "columnar: block b{bi} ends in emit.to — stage-B multiplicity is \
                 row-path-only (use the interpreter or cranelift backend)"
            )));
        }
    }

    // Topological order via Kahn; any cycle (legal in stage-B IR for
    // multiplicity loops) is out of scope for the mask model.
    let n = p.blocks.len();
    let mut indeg = vec![0usize; n];
    for b in &p.blocks {
        for (s, _) in b.term.successors() {
            indeg[s.0 as usize] += 1;
        }
    }
    let mut stack: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut topo = Vec::with_capacity(n);
    while let Some(b) = stack.pop() {
        topo.push(b);
        for (s, _) in p.blocks[b].term.successors() {
            let s = s.0 as usize;
            indeg[s] -= 1;
            if indeg[s] == 0 {
                stack.push(s);
            }
        }
    }
    if topo.len() != n {
        return Err(CompileError::Static(
            "columnar: CFG contains a cycle — stage-B multiplicity is row-path-only \
             (use the interpreter or cranelift backend)"
            .to_string(),
        ));
    }

    let regexes = compile_regexes(p).map_err(CompileError::Regex)?;

    // Dense lane slots in definition order (params, then inst defs).
    let mut slots: HashMap<u32, u32> = HashMap::new();
    for b in &p.blocks {
        for (v, _) in &b.params {
            let s = slots.len() as u32;
            slots.entry(v.0).or_insert(s);
        }
        for inst in &b.insts {
            for d in inst.dsts() {
                let s = slots.len() as u32;
                slots.entry(d.0).or_insert(s);
            }
        }
    }
    let nslots = slots.len();

    Ok(ColumnarFn {
        interp,
        p: p.clone(),
        regexes,
        slots,
        nslots,
        topo,
    })
}

impl ColumnarFn {
    /// Fresh reusable buffers, interchangeable with the interpreter's.
    pub fn new_state(&self) -> RunState {
        self.interp.new_state()
    }

    /// Execute over `input`, filling `st.out` (cleared first) and
    /// `st.emitted` — the same contract as [`InterpFn::run`]. On `Err` the
    /// output is meaningless and the whole call is void.
    pub fn run(&self, input: &Batch, st: &mut RunState) -> Result<(), Trap> {
        self.interp.check_input(input)?;
        self.interp.check_state(st)?;
        st.arena.clear();
        st.emitted = 0;
        for col in st.out.iter_mut() {
            match col {
                OutCol::I1(v) => v.clear(),
                OutCol::I64(v) => v.clear(),
                OutCol::F64(v) => v.clear(),
                OutCol::Str(v) => v.clear(),
            }
        }
        interp::reserve_out(&mut st.out, input.rows);

        let rows = input.rows;
        let mut ex = Exec {
            rows,
            lanes: (0..self.nslots).map(|_| None).collect(),
            traps: vec![None; rows],
            emitted: vec![false; rows],
            stage: self
                .p
                .out_cols
                .iter()
                .map(|c| match c.ty.ty {
                    Ty::I1 => OutCol::I1(vec![(false, false); rows]),
                    Ty::I64 => OutCol::I64(vec![(false, 0); rows]),
                    Ty::F64 => OutCol::F64(vec![(false, 0.0); rows]),
                    Ty::Str => OutCol::Str(vec![(false, StrRef { off: 0, len: 0 }); rows]),
                })
                .collect(),
            input,
            arena: &mut st.arena,
        };
        // Pre-create param lanes so edge merges have a target to write into.
        for b in &self.p.blocks {
            for (v, ty) in &b.params {
                ex.lanes[self.sl(*v)] = Some(default_lane(*ty, rows));
            }
        }
        // Entry starts all-active; every other mask accumulates from edges.
        let mut masks: Vec<Vec<bool>> = (0..self.p.blocks.len())
            .map(|bi| vec![bi == 0; rows])
            .collect();

        for &bi in &self.topo {
            let mut mask = std::mem::take(&mut masks[bi]);
            let block = &self.p.blocks[bi];
            for inst in &block.insts {
                self.exec_inst(inst, &mut mask, &mut ex);
            }
            match &block.term {
                Term::Jump { to, args } => {
                    self.apply_edge(&mut ex, &mut masks, *to, &mask, args)
                }
                Term::Brif {
                    cond,
                    then_to,
                    then_args,
                    else_to,
                    else_args,
                } => {
                    let (then_mask, else_mask) = {
                        let c = i1s(&ex.lanes[self.sl(*cond)]);
                        let t: Vec<bool> =
                            mask.iter().zip(c).map(|(&m, &cv)| m && cv).collect();
                        let e: Vec<bool> =
                            mask.iter().zip(c).map(|(&m, &cv)| m && !cv).collect();
                        (t, e)
                    };
                    self.apply_edge(&mut ex, &mut masks, *then_to, &then_mask, then_args);
                    self.apply_edge(&mut ex, &mut masks, *else_to, &else_mask, else_args);
                }
                Term::Emit => {
                    for (em, &m) in ex.emitted.iter_mut().zip(mask.iter()) {
                        if m {
                            *em = true;
                        }
                    }
                }
                Term::Skip => {}
                Term::Trap { msg } => {
                    for (t, &m) in ex.traps.iter_mut().zip(mask.iter()) {
                        if m && t.is_none() {
                            *t = Some(msg.clone());
                        }
                    }
                }
                Term::EmitTo { .. } => unreachable!("emit.to rejected at compile"),
            }
        }

        // The smallest trapping row's (earliest) trap is the one the row
        // loop would have surfaced.
        let Exec {
            traps,
            emitted: emit_rows,
            stage,
            ..
        } = ex;
        if let Some(msg) = traps.into_iter().flatten().next() {
            return Err(Trap(msg));
        }

        // Gather staged values in row order; without multiplicity each row
        // emits at most once, reproducing the row loop's output exactly.
        let mut emitted = 0usize;
        for r in 0..rows {
            if !emit_rows[r] {
                continue;
            }
            emitted += 1;
            for (oc, sc) in st.out.iter_mut().zip(stage.iter()) {
                match (oc, sc) {
                    (OutCol::I1(o), OutCol::I1(s)) => o.push(s[r]),
                    (OutCol::I64(o), OutCol::I64(s)) => o.push(s[r]),
                    (OutCol::F64(o), OutCol::F64(s)) => o.push(s[r]),
                    (OutCol::Str(o), OutCol::Str(s)) => o.push(s[r]),
                    _ => unreachable!("stage lanes built from the out declarations"),
                }
            }
        }
        st.emitted = emitted;
        Ok(())
    }

    fn sl(&self, v: Value) -> usize {
        self.slots[&v.0] as usize
    }

    /// Merge one CFG edge into its target: OR the edge mask into the
    /// target's accumulated mask and copy branch args into param lanes at
    /// the edge's active rows. Rows are active on at most one edge (paths
    /// are disjoint), so masked writes never conflict.
    fn apply_edge(
        &self,
        ex: &mut Exec,
        masks: &mut [Vec<bool>],
        to: BlockId,
        edge_mask: &[bool],
        args: &[Value],
    ) {
        let to = to.0 as usize;
        for (arg, (pv, _)) in args.iter().zip(self.p.blocks[to].params.iter()) {
            let ps = self.sl(*pv);
            let mut plane = ex.lanes[ps].take().expect("param lane pre-created");
            {
                let alane = ex.lanes[self.sl(*arg)]
                    .as_ref()
                    .expect("branch arg lane defined before the edge");
                copy_masked(&mut plane, alane, edge_mask);
            }
            ex.lanes[ps] = Some(plane);
        }
        let tm = &mut masks[to];
        for (r, &m) in edge_mask.iter().enumerate() {
            if m {
                tm[r] = true;
            }
        }
    }

    /// One instruction as a masked kernel: read operand lanes, compute the
    /// destination lane(s) for active rows (defaults elsewhere — garbage
    /// where inactive is fine), record traps per row and deactivate. Every
    /// arm mirrors the corresponding `interp.rs` closure exactly, calling
    /// the same shared semantic helpers.
    #[allow(clippy::too_many_lines)]
    fn exec_inst(&self, inst: &Inst, mask: &mut [bool], ex: &mut Exec) {
        let rows = ex.rows;
        match inst {
            Inst::Const { dst, lit } => {
                let sd = self.sl(*dst);
                ex.lanes[sd] = Some(match lit {
                    Lit::I1(b) => VLane::I1(vec![*b; rows]),
                    Lit::I64(i) => VLane::I64(vec![*i; rows]),
                    Lit::F64(f) => VLane::F64(vec![*f; rows]),
                    Lit::Str(s) => {
                        let r = ex.arena.push_str(s);
                        VLane::Str(vec![r; rows])
                    }
                });
            }
            Inst::Bin { op, dst, a, b } => {
                let (sd, sa, sb) = (self.sl(*dst), self.sl(*a), self.sl(*b));
                match op {
                    BinOp::Iadd | BinOp::Isub | BinOp::Imul => {
                        let mut out = vec![0i64; rows];
                        let x = i64s(&ex.lanes[sa]);
                        let y = i64s(&ex.lanes[sb]);
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            let v = match op {
                                BinOp::Iadd => x[r].checked_add(y[r]),
                                BinOp::Isub => x[r].checked_sub(y[r]),
                                _ => x[r].checked_mul(y[r]),
                            };
                            match v {
                                Some(v) => out[r] = v,
                                None => fail(
                                    mask,
                                    &mut ex.traps,
                                    r,
                                    Trap(overflow_msg(*op, x[r], y[r])),
                                ),
                            }
                        }
                        ex.lanes[sd] = Some(VLane::I64(out));
                    }
                    BinOp::Idiv | BinOp::Irem => {
                        let mut out = vec![0i64; rows];
                        let x = i64s(&ex.lanes[sa]);
                        let y = i64s(&ex.lanes[sb]);
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            if y[r] == 0 {
                                fail(
                                    mask,
                                    &mut ex.traps,
                                    r,
                                    Trap(format!("division by zero in {}", op.name())),
                                );
                                continue;
                            }
                            let v = match op {
                                BinOp::Idiv => x[r].checked_div(y[r]),
                                _ => x[r].checked_rem(y[r]),
                            };
                            match v {
                                Some(v) => out[r] = v,
                                None => fail(
                                    mask,
                                    &mut ex.traps,
                                    r,
                                    Trap(overflow_msg(*op, x[r], y[r])),
                                ),
                            }
                        }
                        ex.lanes[sd] = Some(VLane::I64(out));
                    }
                    BinOp::Fadd
                    | BinOp::Fsub
                    | BinOp::Fmul
                    | BinOp::Fdiv
                    | BinOp::Frem
                    | BinOp::Fpow
                    | BinOp::Ffloordiv
                    | BinOp::Ffloormod
                    | BinOp::Fnextafter => {
                        let mut out = vec![0.0f64; rows];
                        let x = f64s(&ex.lanes[sa]);
                        let y = f64s(&ex.lanes[sb]);
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            let v = match op {
                                BinOp::Fadd => Ok(x[r] + y[r]),
                                BinOp::Fsub => Ok(x[r] - y[r]),
                                BinOp::Fmul => Ok(x[r] * y[r]),
                                BinOp::Fdiv => Ok(x[r] / y[r]),
                                BinOp::Fpow => duck_pow(x[r], y[r]),
                                BinOp::Ffloordiv => Ok(duck_fdiv(x[r], y[r])),
                                BinOp::Ffloormod => Ok(duck_fmod(x[r], y[r])),
                                BinOp::Fnextafter => Ok(duck_nextafter(x[r], y[r])),
                                _ => Ok(x[r] % y[r]),
                            };
                            match v {
                                Ok(v) => out[r] = v,
                                Err(t) => fail(mask, &mut ex.traps, r, t),
                            }
                        }
                        ex.lanes[sd] = Some(VLane::F64(out));
                    }
                    // log(base, x): a is the base (SQL argument order).
                    BinOp::Flogb => {
                        let mut out = vec![0.0f64; rows];
                        let x = f64s(&ex.lanes[sa]);
                        let y = f64s(&ex.lanes[sb]);
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            match duck_logb(x[r], y[r]) {
                                Ok(v) => out[r] = v,
                                Err(t) => fail(mask, &mut ex.traps, r, t),
                            }
                        }
                        ex.lanes[sd] = Some(VLane::F64(out));
                    }
                    BinOp::Ishl => {
                        let mut out = vec![0i64; rows];
                        let x = i64s(&ex.lanes[sa]);
                        let y = i64s(&ex.lanes[sb]);
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            match duck_shl(x[r], y[r]) {
                                Ok(v) => out[r] = v,
                                Err(t) => fail(mask, &mut ex.traps, r, t),
                            }
                        }
                        ex.lanes[sd] = Some(VLane::I64(out));
                    }
                    BinOp::Ishr | BinOp::Iand | BinOp::Ior | BinOp::Ixor => {
                        let mut out = vec![0i64; rows];
                        let x = i64s(&ex.lanes[sa]);
                        let y = i64s(&ex.lanes[sb]);
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            out[r] = match op {
                                BinOp::Ishr => duck_shr(x[r], y[r]),
                                BinOp::Iand => x[r] & y[r],
                                BinOp::Ior => x[r] | y[r],
                                _ => x[r] ^ y[r],
                            };
                        }
                        ex.lanes[sd] = Some(VLane::I64(out));
                    }
                    BinOp::And | BinOp::Or | BinOp::Xor => {
                        let mut out = vec![false; rows];
                        let x = i1s(&ex.lanes[sa]);
                        let y = i1s(&ex.lanes[sb]);
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            out[r] = match op {
                                BinOp::And => x[r] && y[r],
                                BinOp::Or => x[r] || y[r],
                                _ => x[r] ^ y[r],
                            };
                        }
                        ex.lanes[sd] = Some(VLane::I1(out));
                    }
                }
            }
            Inst::Cmp {
                pred,
                ty,
                dst,
                a,
                b,
            } => {
                let (sd, sa, sb) = (self.sl(*dst), self.sl(*a), self.sl(*b));
                let mut out = vec![false; rows];
                match ty {
                    Ty::I64 => {
                        let x = i64s(&ex.lanes[sa]);
                        let y = i64s(&ex.lanes[sb]);
                        for r in 0..rows {
                            if mask[r] {
                                out[r] = apply_ord(*pred, x[r].cmp(&y[r]));
                            }
                        }
                    }
                    Ty::F64 => {
                        // DuckDB DOUBLE order, not IEEE (exec::duck_fcmp).
                        let x = f64s(&ex.lanes[sa]);
                        let y = f64s(&ex.lanes[sb]);
                        for r in 0..rows {
                            if mask[r] {
                                out[r] = apply_ord(*pred, duck_fcmp(x[r], y[r]));
                            }
                        }
                    }
                    Ty::Str => {
                        let x = strs(&ex.lanes[sa]);
                        let y = strs(&ex.lanes[sb]);
                        for r in 0..rows {
                            if mask[r] {
                                out[r] = apply_ord(
                                    *pred,
                                    ex.arena.get(x[r]).cmp(ex.arena.get(y[r])),
                                );
                            }
                        }
                    }
                    Ty::I1 => unreachable!("cmp on i1 is rejected by the verifier"),
                }
                ex.lanes[sd] = Some(VLane::I1(out));
            }
            Inst::Not { dst, a } => {
                let (sd, sa) = (self.sl(*dst), self.sl(*a));
                let mut out = vec![false; rows];
                let x = i1s(&ex.lanes[sa]);
                for r in 0..rows {
                    if mask[r] {
                        out[r] = !x[r];
                    }
                }
                ex.lanes[sd] = Some(VLane::I1(out));
            }
            Inst::Select { dst, cond, a, b } => {
                let (sd, sc, sa, sb) = (
                    self.sl(*dst),
                    self.sl(*cond),
                    self.sl(*a),
                    self.sl(*b),
                );
                let out = {
                    let c = i1s(&ex.lanes[sc]);
                    let pick = |r: usize| mask[r] && c[r];
                    match (
                        ex.lanes[sa].as_ref().expect("lane defined"),
                        ex.lanes[sb].as_ref().expect("lane defined"),
                    ) {
                        (VLane::I1(x), VLane::I1(y)) => VLane::I1(
                            (0..rows).map(|r| if pick(r) { x[r] } else { y[r] }).collect(),
                        ),
                        (VLane::I64(x), VLane::I64(y)) => VLane::I64(
                            (0..rows).map(|r| if pick(r) { x[r] } else { y[r] }).collect(),
                        ),
                        (VLane::F64(x), VLane::F64(y)) => VLane::F64(
                            (0..rows).map(|r| if pick(r) { x[r] } else { y[r] }).collect(),
                        ),
                        (VLane::Str(x), VLane::Str(y)) => VLane::Str(
                            (0..rows).map(|r| if pick(r) { x[r] } else { y[r] }).collect(),
                        ),
                        _ => unreachable!("select operand types checked by the verifier"),
                    }
                };
                ex.lanes[sd] = Some(out);
            }
            Inst::Itof { dst, a } => {
                let (sd, sa) = (self.sl(*dst), self.sl(*a));
                let mut out = vec![0.0f64; rows];
                let x = i64s(&ex.lanes[sa]);
                for r in 0..rows {
                    if mask[r] {
                        out[r] = x[r] as f64;
                    }
                }
                ex.lanes[sd] = Some(VLane::F64(out));
            }
            Inst::Ftoi { mode, dst, a } => {
                let (sd, sa) = (self.sl(*dst), self.sl(*a));
                let mut out = vec![0i64; rows];
                let xs = f64s(&ex.lanes[sa]);
                for r in 0..rows {
                    if !mask[r] {
                        continue;
                    }
                    let x = xs[r];
                    let v = match mode {
                        RoundMode::Trunc => x.trunc(),
                        RoundMode::Round => x.round(), // half away from zero
                    };
                    // 2^63 is exactly representable; anything in
                    // [-2^63, 2^63) fits i64 after rounding.
                    if v.is_finite() && v >= -(2f64.powi(63)) && v < 2f64.powi(63) {
                        out[r] = v as i64;
                    } else {
                        fail(
                            mask,
                            &mut ex.traps,
                            r,
                            Trap(format!("f64 value {x:?} out of i64 range in ftoi")),
                        );
                    }
                }
                ex.lanes[sd] = Some(VLane::I64(out));
            }
            Inst::Itos { dst, a } => {
                let (sd, sa) = (self.sl(*dst), self.sl(*a));
                let mut out = vec![StrRef { off: 0, len: 0 }; rows];
                let x = i64s(&ex.lanes[sa]);
                for r in 0..rows {
                    if mask[r] {
                        let v = x[r];
                        out[r] = ex.arena.push_fmt(format_args!("{v}"));
                    }
                }
                ex.lanes[sd] = Some(VLane::Str(out));
            }
            Inst::Ftos { dst, a } => {
                let (sd, sa) = (self.sl(*dst), self.sl(*a));
                let mut out = vec![StrRef { off: 0, len: 0 }; rows];
                let x = f64s(&ex.lanes[sa]);
                for r in 0..rows {
                    if mask[r] {
                        out[r] = ex
                            .arena
                            .push_fmt(format_args!("{}", DuckF64(x[r])));
                    }
                }
                ex.lanes[sd] = Some(VLane::Str(out));
            }
            Inst::StoiOpt { flag, dst, a } => {
                let (sf, sd, sa) = (self.sl(*flag), self.sl(*dst), self.sl(*a));
                let mut flags = vec![false; rows];
                let mut out = vec![0i64; rows];
                let x = strs(&ex.lanes[sa]);
                for r in 0..rows {
                    if !mask[r] {
                        continue;
                    }
                    if let Ok(v) = ex.arena.get(x[r]).trim_ascii().parse::<i64>() {
                        flags[r] = true;
                        out[r] = v;
                    }
                }
                ex.lanes[sf] = Some(VLane::I1(flags));
                ex.lanes[sd] = Some(VLane::I64(out));
            }
            Inst::StofOpt { flag, dst, a } => {
                let (sf, sd, sa) = (self.sl(*flag), self.sl(*dst), self.sl(*a));
                let mut flags = vec![false; rows];
                let mut out = vec![0.0f64; rows];
                let x = strs(&ex.lanes[sa]);
                for r in 0..rows {
                    if !mask[r] {
                        continue;
                    }
                    if let Ok(v) = ex.arena.get(x[r]).trim_ascii().parse::<f64>() {
                        flags[r] = true;
                        out[r] = v;
                    }
                }
                ex.lanes[sf] = Some(VLane::I1(flags));
                ex.lanes[sd] = Some(VLane::F64(out));
            }
            Inst::Sconcat { dst, a, b } => {
                let (sd, sa, sb) = (self.sl(*dst), self.sl(*a), self.sl(*b));
                let mut out = vec![StrRef { off: 0, len: 0 }; rows];
                let x = strs(&ex.lanes[sa]);
                let y = strs(&ex.lanes[sb]);
                for r in 0..rows {
                    if mask[r] {
                        out[r] = ex.arena.concat(x[r], y[r]);
                    }
                }
                ex.lanes[sd] = Some(VLane::Str(out));
            }
            Inst::Str2 { op, dst, a, b } => {
                let (sd, sa, sb) = (self.sl(*dst), self.sl(*a), self.sl(*b));
                let x = strs(&ex.lanes[sa]);
                let y = strs(&ex.lanes[sb]);
                match op {
                    StrOp2::Find => {
                        let mut out = vec![0i64; rows];
                        for r in 0..rows {
                            if mask[r] {
                                out[r] =
                                    str_find(ex.arena.get(x[r]), ex.arena.get(y[r]));
                            }
                        }
                        ex.lanes[sd] = Some(VLane::I64(out));
                    }
                    StrOp2::Levenshtein => {
                        let mut out = vec![0i64; rows];
                        for r in 0..rows {
                            if mask[r] {
                                out[r] = duck_levenshtein(
                                    ex.arena.get(x[r]).as_bytes(),
                                    ex.arena.get(y[r]).as_bytes(),
                                );
                            }
                        }
                        ex.lanes[sd] = Some(VLane::I64(out));
                    }
                    StrOp2::Damerau => {
                        let mut out = vec![0i64; rows];
                        for r in 0..rows {
                            if mask[r] {
                                out[r] = duck_damerau(
                                    ex.arena.get(x[r]).as_bytes(),
                                    ex.arena.get(y[r]).as_bytes(),
                                );
                            }
                        }
                        ex.lanes[sd] = Some(VLane::I64(out));
                    }
                    StrOp2::Jaccard => {
                        let mut out = vec![0.0f64; rows];
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            match duck_jaccard(
                                ex.arena.get(x[r]).as_bytes(),
                                ex.arena.get(y[r]).as_bytes(),
                            ) {
                                Ok(v) => out[r] = v,
                                Err(t) => fail(mask, &mut ex.traps, r, t),
                            }
                        }
                        ex.lanes[sd] = Some(VLane::F64(out));
                    }
                    StrOp2::Hamming => {
                        let mut out = vec![0i64; rows];
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            match duck_hamming(
                                ex.arena.get(x[r]).as_bytes(),
                                ex.arena.get(y[r]).as_bytes(),
                            ) {
                                Ok(v) => out[r] = v,
                                Err(t) => fail(mask, &mut ex.traps, r, t),
                            }
                        }
                        ex.lanes[sd] = Some(VLane::I64(out));
                    }
                    op => {
                        let mut out = vec![false; rows];
                        for r in 0..rows {
                            if mask[r] {
                                out[r] = str_pred(
                                    *op,
                                    ex.arena.get(x[r]),
                                    ex.arena.get(y[r]),
                                );
                            }
                        }
                        ex.lanes[sd] = Some(VLane::I1(out));
                    }
                }
            }
            Inst::Str3 { op, dst, a, b, c } => {
                let (sd, sa, sb, sc) =
                    (self.sl(*dst), self.sl(*a), self.sl(*b), self.sl(*c));
                let mut out = vec![StrRef { off: 0, len: 0 }; rows];
                let av = strs(&ex.lanes[sa]);
                let bv = strs(&ex.lanes[sb]);
                let cv = strs(&ex.lanes[sc]);
                for r in 0..rows {
                    if !mask[r] {
                        continue;
                    }
                    let o = {
                        let s = ex.arena.get(av[r]);
                        let x = ex.arena.get(bv[r]);
                        let y = ex.arena.get(cv[r]);
                        match op {
                            StrOp3::Replace => duck_replace(s, x, y),
                            StrOp3::Translate => duck_translate(s, x, y),
                        }
                    };
                    out[r] = ex.arena.push_str(&o);
                }
                ex.lanes[sd] = Some(VLane::Str(out));
            }
            Inst::Str2i { op, dst, a, n } => {
                let (sd, sa, sn) = (self.sl(*dst), self.sl(*a), self.sl(*n));
                let mut out = vec![StrRef { off: 0, len: 0 }; rows];
                let av = strs(&ex.lanes[sa]);
                let nv = i64s(&ex.lanes[sn]);
                match op {
                    StrOp2i::Repeat => {
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            match duck_repeat(ex.arena.get(av[r]), nv[r]) {
                                Ok(s) => out[r] = ex.arena.push_str(&s),
                                Err(t) => fail(mask, &mut ex.traps, r, t),
                            }
                        }
                    }
                    StrOp2i::Extract => {
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            // Same ±2^32 window and trap as substr.
                            if !substr_range_ok(nv[r]) {
                                fail(
                                    mask,
                                    &mut ex.traps,
                                    r,
                                    Trap(
                                        "substring offset outside of supported range"
                                            .to_string(),
                                    ),
                                );
                                continue;
                            }
                            let sref = av[r];
                            let rng = extract_window(ex.arena.get(sref), nv[r]);
                            // The extracted char is a subview of the input.
                            out[r] = StrRef {
                                off: sref.off + rng.start,
                                len: rng.end - rng.start,
                            };
                        }
                    }
                }
                ex.lanes[sd] = Some(VLane::Str(out));
            }
            Inst::Spad {
                left,
                dst,
                a,
                len,
                pad,
            } => {
                let (sd, sa, sl_, sp) = (
                    self.sl(*dst),
                    self.sl(*a),
                    self.sl(*len),
                    self.sl(*pad),
                );
                let mut out = vec![StrRef { off: 0, len: 0 }; rows];
                let av = strs(&ex.lanes[sa]);
                let lv = i64s(&ex.lanes[sl_]);
                let pv = strs(&ex.lanes[sp]);
                for r in 0..rows {
                    if !mask[r] {
                        continue;
                    }
                    match duck_pad(*left, ex.arena.get(av[r]), lv[r], ex.arena.get(pv[r]))
                    {
                        Ok(s) => out[r] = ex.arena.push_str(&s),
                        Err(t) => fail(mask, &mut ex.traps, r, t),
                    }
                }
                ex.lanes[sd] = Some(VLane::Str(out));
            }
            Inst::Sslice { dst, a, lo, hi } => {
                let (sd, sa, slo, shi) =
                    (self.sl(*dst), self.sl(*a), self.sl(*lo), self.sl(*hi));
                let mut out = vec![StrRef { off: 0, len: 0 }; rows];
                let av = strs(&ex.lanes[sa]);
                let lov = i64s(&ex.lanes[slo]);
                let hiv = i64s(&ex.lanes[shi]);
                for r in 0..rows {
                    if !mask[r] {
                        continue;
                    }
                    let sref = av[r];
                    let rng = slice_window(ex.arena.get(sref), lov[r], hiv[r]);
                    out[r] = StrRef {
                        off: sref.off + rng.start,
                        len: rng.end - rng.start,
                    };
                }
                ex.lanes[sd] = Some(VLane::Str(out));
            }
            Inst::Sord {
                empty_zero,
                dst,
                a,
            } => {
                let (sd, sa) = (self.sl(*dst), self.sl(*a));
                let mut out = vec![0i64; rows];
                let av = strs(&ex.lanes[sa]);
                for r in 0..rows {
                    if mask[r] {
                        out[r] = interp::duck_ord(ex.arena.get(av[r]), *empty_zero);
                    }
                }
                ex.lanes[sd] = Some(VLane::I64(out));
            }
            Inst::SLen { bytes, dst, a } => {
                let (sd, sa) = (self.sl(*dst), self.sl(*a));
                let mut out = vec![0i64; rows];
                let av = strs(&ex.lanes[sa]);
                for r in 0..rows {
                    if mask[r] {
                        let s = ex.arena.get(av[r]);
                        out[r] = if *bytes {
                            s.len() as i64
                        } else {
                            s.chars().count() as i64
                        };
                    }
                }
                ex.lanes[sd] = Some(VLane::I64(out));
            }
            Inst::Slike { ci, dst, a, p, esc } => {
                let (sd, sa, sp) = (self.sl(*dst), self.sl(*a), self.sl(*p));
                let mut out = vec![false; rows];
                let av = strs(&ex.lanes[sa]);
                let pv = strs(&ex.lanes[sp]);
                let ev = esc.map(|e| strs(&ex.lanes[self.sl(e)]));
                for r in 0..rows {
                    if !mask[r] {
                        continue;
                    }
                    let e = match ev {
                        None => None,
                        Some(ev) => match like_escape_of(ex.arena.get(ev[r])) {
                            Ok(e) => e,
                            Err(t) => {
                                fail(mask, &mut ex.traps, r, t);
                                continue;
                            }
                        },
                    };
                    let (sr, pr) = (av[r], pv[r]);
                    let (sr, pr) = if *ci {
                        // ILIKE: fold BOTH sides with the measured simple
                        // casemap (see interp.rs for the pinned divergence).
                        (
                            ex.arena.case_map(sr, super::casemap::simple_lower),
                            ex.arena.case_map(pr, super::casemap::simple_lower),
                        )
                    } else {
                        (sr, pr)
                    };
                    let ok = {
                        let sv = ex.arena.get(sr).as_bytes();
                        let pb = ex.arena.get(pr).as_bytes();
                        like_match(sv, pb, e)
                    };
                    match ok {
                        Ok(v) => out[r] = v,
                        Err(t) => fail(mask, &mut ex.traps, r, t),
                    }
                }
                ex.lanes[sd] = Some(VLane::I1(out));
            }
            Inst::Round2f { trunc, dst, a, n } => {
                let (sd, sa, sn) = (self.sl(*dst), self.sl(*a), self.sl(*n));
                let f = if *trunc { trunc_prec_f64 } else { round_prec_f64 };
                let mut out = vec![0.0f64; rows];
                let av = f64s(&ex.lanes[sa]);
                let nv = i64s(&ex.lanes[sn]);
                for r in 0..rows {
                    if mask[r] {
                        out[r] = f(av[r], nv[r]);
                    }
                }
                ex.lanes[sd] = Some(VLane::F64(out));
            }
            Inst::Round2i { trunc, dst, a, n } => {
                let (sd, sa, sn) = (self.sl(*dst), self.sl(*a), self.sl(*n));
                let f = if *trunc { trunc_prec_i64 } else { round_prec_i64 };
                let mut out = vec![0i64; rows];
                let av = i64s(&ex.lanes[sa]);
                let nv = i64s(&ex.lanes[sn]);
                for r in 0..rows {
                    if mask[r] {
                        out[r] = f(av[r], nv[r]);
                    }
                }
                ex.lanes[sd] = Some(VLane::I64(out));
            }
            Inst::Str1 { op, dst, a } => {
                let (sd, sa) = (self.sl(*dst), self.sl(*a));
                let mut out = vec![StrRef { off: 0, len: 0 }; rows];
                let av = strs(&ex.lanes[sa]);
                match op {
                    StrOp1::Upper | StrOp1::Lower => {
                        let map: fn(char) -> char = match op {
                            StrOp1::Upper => super::casemap::simple_upper,
                            _ => super::casemap::simple_lower,
                        };
                        for r in 0..rows {
                            if mask[r] {
                                out[r] = ex.arena.case_map(av[r], map);
                            }
                        }
                    }
                    StrOp1::StripAccents => {
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            let sref = av[r];
                            let o = duck_strip_accents(ex.arena.get(sref));
                            out[r] = match o {
                                None => sref, // ASCII fast path: verbatim
                                Some(s) => ex.arena.push_str(&s),
                            };
                        }
                    }
                    StrOp1::Reverse => {
                        for r in 0..rows {
                            if mask[r] {
                                let o = duck_reverse(ex.arena.get(av[r]));
                                out[r] = ex.arena.push_str(&o);
                            }
                        }
                    }
                }
                ex.lanes[sd] = Some(VLane::Str(out));
            }
            Inst::Strim {
                side,
                dst,
                a,
                chars,
            } => {
                let (sd, sa, sc) = (self.sl(*dst), self.sl(*a), self.sl(*chars));
                let mut out = vec![StrRef { off: 0, len: 0 }; rows];
                let av = strs(&ex.lanes[sa]);
                let cv = strs(&ex.lanes[sc]);
                for r in 0..rows {
                    if !mask[r] {
                        continue;
                    }
                    let sref = av[r];
                    let rng =
                        trim_bounds(ex.arena.get(sref), ex.arena.get(cv[r]), *side);
                    out[r] = StrRef {
                        off: sref.off + rng.start,
                        len: rng.end - rng.start,
                    };
                }
                ex.lanes[sd] = Some(VLane::Str(out));
            }
            Inst::Ssubstr { dst, a, start, len } => {
                let (sd, sa, ss) = (self.sl(*dst), self.sl(*a), self.sl(*start));
                let len_slot = len.map(|l| self.sl(l));
                let mut out = vec![StrRef { off: 0, len: 0 }; rows];
                let av = strs(&ex.lanes[sa]);
                let sv = i64s(&ex.lanes[ss]);
                let lv = len_slot.map(|l| i64s(&ex.lanes[l]));
                for r in 0..rows {
                    if !mask[r] {
                        continue;
                    }
                    let st_ = sv[r];
                    if !substr_range_ok(st_) {
                        fail(
                            mask,
                            &mut ex.traps,
                            r,
                            Trap("substring offset outside of supported range".to_string()),
                        );
                        continue;
                    }
                    let ln = match lv {
                        Some(lv) => {
                            let v = lv[r];
                            if !substr_range_ok(v) {
                                fail(
                                    mask,
                                    &mut ex.traps,
                                    r,
                                    Trap(
                                        "substring length outside of supported range"
                                            .to_string(),
                                    ),
                                );
                                continue;
                            }
                            Some(v)
                        }
                        None => None,
                    };
                    let sref = av[r];
                    let rng = substr_window(ex.arena.get(sref), st_, ln);
                    out[r] = StrRef {
                        off: sref.off + rng.start,
                        len: rng.end - rng.start,
                    };
                }
                ex.lanes[sd] = Some(VLane::Str(out));
            }
            Inst::Num1 { op, dst, a } => {
                let (sd, sa) = (self.sl(*dst), self.sl(*a));
                match op {
                    NumOp1::Iabs => {
                        let mut out = vec![0i64; rows];
                        let x = i64s(&ex.lanes[sa]);
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            match x[r].checked_abs() {
                                Some(v) => out[r] = v,
                                None => fail(
                                    mask,
                                    &mut ex.traps,
                                    r,
                                    Trap(abs_overflow_msg(x[r])),
                                ),
                            }
                        }
                        ex.lanes[sd] = Some(VLane::I64(out));
                    }
                    NumOp1::Fabs | NumOp1::Fround => {
                        let mut out = vec![0.0f64; rows];
                        let x = f64s(&ex.lanes[sa]);
                        for r in 0..rows {
                            if mask[r] {
                                out[r] = match op {
                                    NumOp1::Fabs => x[r].abs(),
                                    _ => x[r].round(),
                                };
                            }
                        }
                        ex.lanes[sd] = Some(VLane::F64(out));
                    }
                    op => {
                        let f = math1_fn(*op);
                        let mut out = vec![0.0f64; rows];
                        let x = f64s(&ex.lanes[sa]);
                        for r in 0..rows {
                            if !mask[r] {
                                continue;
                            }
                            match f(x[r]) {
                                Ok(v) => out[r] = v,
                                Err(t) => fail(mask, &mut ex.traps, r, t),
                            }
                        }
                        ex.lanes[sd] = Some(VLane::F64(out));
                    }
                }
            }
            Inst::Load { dst, col } => {
                let sd = self.sl(*dst);
                let c = &ex.input.cols[*col as usize];
                let lane = match c {
                    ColData::I1 { data, .. } => VLane::I1(data.clone()),
                    ColData::I64 { data, .. } => VLane::I64(data.clone()),
                    ColData::F64 { data, .. } => VLane::F64(data.clone()),
                    c @ ColData::Str { .. } => {
                        let mut v = vec![StrRef { off: 0, len: 0 }; rows];
                        for (r, slot) in v.iter_mut().enumerate() {
                            if mask[r] {
                                *slot = ex.arena.push_str(c.str_at(r));
                            }
                        }
                        VLane::Str(v)
                    }
                };
                ex.lanes[sd] = Some(lane);
            }
            Inst::LoadOpt { flag, dst, col } => {
                let (sf, sd) = (self.sl(*flag), self.sl(*dst));
                let c = &ex.input.cols[*col as usize];
                let flags: Vec<bool> = (0..rows).map(|r| interp::col_valid(c, r)).collect();
                // Invalid slots normalize to the type default, never the
                // batch's garbage payload (spec pin).
                let lane = match c {
                    ColData::I1 { data, .. } => VLane::I1(
                        (0..rows).map(|r| if flags[r] { data[r] } else { false }).collect(),
                    ),
                    ColData::I64 { data, .. } => VLane::I64(
                        (0..rows).map(|r| if flags[r] { data[r] } else { 0 }).collect(),
                    ),
                    ColData::F64 { data, .. } => VLane::F64(
                        (0..rows).map(|r| if flags[r] { data[r] } else { 0.0 }).collect(),
                    ),
                    c @ ColData::Str { .. } => {
                        let mut v = vec![StrRef { off: 0, len: 0 }; rows];
                        for (r, slot) in v.iter_mut().enumerate() {
                            if mask[r] && flags[r] {
                                *slot = ex.arena.push_str(c.str_at(r));
                            }
                        }
                        VLane::Str(v)
                    }
                };
                ex.lanes[sd] = Some(lane);
                ex.lanes[sf] = Some(VLane::I1(flags));
            }
            Inst::Store { col, val } => {
                let lane = ex.lanes[self.sl(*val)].as_ref().expect("lane defined");
                match (&mut ex.stage[*col as usize], lane) {
                    (OutCol::I1(st_), VLane::I1(v)) => {
                        for r in 0..rows {
                            if mask[r] {
                                st_[r] = (true, v[r]);
                            }
                        }
                    }
                    (OutCol::I64(st_), VLane::I64(v)) => {
                        for r in 0..rows {
                            if mask[r] {
                                st_[r] = (true, v[r]);
                            }
                        }
                    }
                    (OutCol::F64(st_), VLane::F64(v)) => {
                        for r in 0..rows {
                            if mask[r] {
                                st_[r] = (true, v[r]);
                            }
                        }
                    }
                    (OutCol::Str(st_), VLane::Str(v)) => {
                        for r in 0..rows {
                            if mask[r] {
                                st_[r] = (true, v[r]);
                            }
                        }
                    }
                    _ => unreachable!("store type checked by the verifier"),
                }
            }
            Inst::StoreOpt { col, flag, val } => {
                let flags = i1s(&ex.lanes[self.sl(*flag)]);
                let lane = ex.lanes[self.sl(*val)].as_ref().expect("lane defined");
                // Spec: on a false flag the stored payload is the type
                // default — never the live lane value.
                match (&mut ex.stage[*col as usize], lane) {
                    (OutCol::I1(st_), VLane::I1(v)) => {
                        for r in 0..rows {
                            if mask[r] {
                                let ok = flags[r];
                                st_[r] = (ok, if ok { v[r] } else { false });
                            }
                        }
                    }
                    (OutCol::I64(st_), VLane::I64(v)) => {
                        for r in 0..rows {
                            if mask[r] {
                                let ok = flags[r];
                                st_[r] = (ok, if ok { v[r] } else { 0 });
                            }
                        }
                    }
                    (OutCol::F64(st_), VLane::F64(v)) => {
                        for r in 0..rows {
                            if mask[r] {
                                let ok = flags[r];
                                st_[r] = (ok, if ok { v[r] } else { 0.0 });
                            }
                        }
                    }
                    (OutCol::Str(st_), VLane::Str(v)) => {
                        for r in 0..rows {
                            if mask[r] {
                                let ok = flags[r];
                                st_[r] =
                                    (ok, if ok { v[r] } else { StrRef { off: 0, len: 0 } });
                            }
                        }
                    }
                    _ => unreachable!("store type checked by the verifier"),
                }
            }
            Inst::Probe {
                static_id,
                hit,
                dsts,
                keys,
            } => {
                let sid = *static_id as usize;
                let PreparedStatic::Map { entries } = &self.interp.statics()[sid] else {
                    unreachable!("static kind checked at compile");
                };
                let value_tys = match &self.p.statics[sid] {
                    StaticTy::Map { values, .. } => values,
                    _ => unreachable!("probe on non-map is rejected by the verifier"),
                };
                let key_slots: Vec<usize> = keys.iter().map(|k| self.sl(*k)).collect();
                let mut hit_lane = vec![false; rows];
                let mut out_lanes: Vec<VLane> =
                    value_tys.iter().map(|t| default_lane(*t, rows)).collect();
                for r in 0..rows {
                    if !mask[r] {
                        continue;
                    }
                    let found = entries
                        .binary_search_by(|(k, _)| {
                            cmp_key_row(k, &key_slots, &ex.lanes, ex.arena, r)
                        })
                        .ok();
                    if let Some(idx) = found {
                        hit_lane[r] = true;
                        for (ol, v) in out_lanes.iter_mut().zip(entries[idx].1.iter()) {
                            write_scalar(ol, r, v, ex.arena);
                        }
                    }
                }
                ex.lanes[self.sl(*hit)] = Some(VLane::I1(hit_lane));
                for (d, ol) in dsts.iter().zip(out_lanes) {
                    ex.lanes[self.sl(*d)] = Some(ol);
                }
            }
            Inst::ProbeRange { .. } | Inst::ProbeRead { .. } => {
                unreachable!("probe.range/probe.read rejected at compile")
            }
            Inst::Sload { static_id, dst } => {
                let sd = self.sl(*dst);
                let PreparedStatic::Scalar { val, .. } =
                    &self.interp.statics()[*static_id as usize]
                else {
                    unreachable!("static kind checked at compile");
                };
                ex.lanes[sd] = Some(broadcast_scalar(val, rows, ex.arena));
            }
            Inst::SloadOpt {
                static_id,
                flag,
                dst,
            } => {
                let (sf, sd) = (self.sl(*flag), self.sl(*dst));
                let ty = match &self.p.statics[*static_id as usize] {
                    StaticTy::Scalar(ColTy { ty, .. }) => *ty,
                    _ => unreachable!("sload on non-scalar is rejected by the verifier"),
                };
                let PreparedStatic::Scalar { valid, val } =
                    &self.interp.statics()[*static_id as usize]
                else {
                    unreachable!("static kind checked at compile");
                };
                ex.lanes[sd] = Some(if *valid {
                    broadcast_scalar(val, rows, ex.arena)
                } else {
                    default_lane(ty, rows)
                });
                ex.lanes[sf] = Some(VLane::I1(vec![*valid; rows]));
            }
            Inst::ReMatch { re, dst, a } => {
                let (sd, sa) = (self.sl(*dst), self.sl(*a));
                let rx = &self.regexes[*re as usize];
                let mut out = vec![false; rows];
                let av = strs(&ex.lanes[sa]);
                for r in 0..rows {
                    if mask[r] {
                        out[r] = rx.is_match(ex.arena.get(av[r]));
                    }
                }
                ex.lanes[sd] = Some(VLane::I1(out));
            }
            Inst::ReExtract { re, group, dst, a } => {
                let (sd, sa) = (self.sl(*dst), self.sl(*a));
                let rx = &self.regexes[*re as usize];
                let mut out = vec![StrRef { off: 0, len: 0 }; rows];
                let av = strs(&ex.lanes[sa]);
                for r in 0..rows {
                    if !mask[r] {
                        continue;
                    }
                    // No match / non-participating group -> '' (wave-B pins).
                    let o = {
                        let s = ex.arena.get(av[r]);
                        rx.captures(s)
                            .and_then(|c| c.get(*group as usize))
                            .map(|m| m.as_str().to_string())
                            .unwrap_or_default()
                    };
                    out[r] = ex.arena.push_str(&o);
                }
                ex.lanes[sd] = Some(VLane::Str(out));
            }
            Inst::ReReplace { re, global, dst, a } => {
                let (sd, sa) = (self.sl(*dst), self.sl(*a));
                let rx = &self.regexes[*re as usize];
                let template = self.p.regexes[*re as usize]
                    .rewrite
                    .as_deref()
                    .expect("verified: rereplace has a template");
                let mut out = vec![StrRef { off: 0, len: 0 }; rows];
                let av = strs(&ex.lanes[sa]);
                for r in 0..rows {
                    if !mask[r] {
                        continue;
                    }
                    let o = {
                        let s = ex.arena.get(av[r]);
                        if *global {
                            rx.replace_all(s, template).into_owned()
                        } else {
                            rx.replace(s, template).into_owned()
                        }
                    };
                    out[r] = ex.arena.push_str(&o);
                }
                ex.lanes[sd] = Some(VLane::Str(out));
            }
        }
    }
}

/// Mutable per-run state, split from `ColumnarFn` so field borrows stay
/// disjoint inside kernels (lanes read + arena write + trap record).
struct Exec<'a> {
    rows: usize,
    lanes: Vec<Option<VLane>>,
    /// First trap message per row; a recorded row is deactivated.
    traps: Vec<Option<String>>,
    /// Rows whose path reached an `emit`.
    emitted: Vec<bool>,
    /// Full-batch staging per out column; gathered in row order at the end.
    stage: Vec<OutCol>,
    input: &'a Batch,
    arena: &'a mut Arena,
}

/// Record row `r`'s first trap and deactivate it — it computes nothing
/// further, exactly like the row loop aborting that row's execution.
fn fail(mask: &mut [bool], traps: &mut [Option<String>], r: usize, t: Trap) {
    mask[r] = false;
    if traps[r].is_none() {
        traps[r] = Some(t.0);
    }
}

fn default_lane(ty: Ty, rows: usize) -> VLane {
    match ty {
        Ty::I1 => VLane::I1(vec![false; rows]),
        Ty::I64 => VLane::I64(vec![0; rows]),
        Ty::F64 => VLane::F64(vec![0.0; rows]),
        Ty::Str => VLane::Str(vec![StrRef { off: 0, len: 0 }; rows]),
    }
}

fn broadcast_scalar(v: &ScalarVal, rows: usize, arena: &mut Arena) -> VLane {
    match v {
        ScalarVal::I1(b) => VLane::I1(vec![*b; rows]),
        ScalarVal::I64(i) => VLane::I64(vec![*i; rows]),
        ScalarVal::F64(f) => VLane::F64(vec![*f; rows]),
        ScalarVal::Str(s) => {
            let r = arena.push_str(s);
            VLane::Str(vec![r; rows])
        }
    }
}

/// Write one probed static value into row `r` of its destination lane.
fn write_scalar(lane: &mut VLane, r: usize, v: &ScalarVal, arena: &mut Arena) {
    match (lane, v) {
        (VLane::I1(l), ScalarVal::I1(b)) => l[r] = *b,
        (VLane::I64(l), ScalarVal::I64(i)) => l[r] = *i,
        (VLane::F64(l), ScalarVal::F64(f)) => l[r] = *f,
        (VLane::Str(l), ScalarVal::Str(s)) => l[r] = arena.push_str(s),
        _ => unreachable!("static value types checked at compile"),
    }
}

/// The lane-side mirror of the interpreter's `cmp_key`: compare a stored
/// key tuple against the probe lanes at row `r`, position-wise.
fn cmp_key_row(
    stored: &[KeyBits],
    key_slots: &[usize],
    lanes: &[Option<VLane>],
    arena: &Arena,
    r: usize,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    for (kb, slot) in stored.iter().zip(key_slots.iter()) {
        let ord = match (kb, lanes[*slot].as_ref().expect("key lane defined")) {
            (KeyBits::I1(s), VLane::I1(v)) => s.cmp(&v[r]),
            (KeyBits::I64(s), VLane::I64(v)) => s.cmp(&v[r]),
            (KeyBits::F64(s), VLane::F64(v)) => s.cmp(&canon_f64_bits(v[r])),
            (KeyBits::Str(s), VLane::Str(v)) => s.as_str().cmp(arena.get(v[r])),
            _ => unreachable!("probe key types checked at compile"),
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn copy_masked(dst: &mut VLane, src: &VLane, m: &[bool]) {
    match (dst, src) {
        (VLane::I1(d), VLane::I1(s)) => {
            for (r, &mm) in m.iter().enumerate() {
                if mm {
                    d[r] = s[r];
                }
            }
        }
        (VLane::I64(d), VLane::I64(s)) => {
            for (r, &mm) in m.iter().enumerate() {
                if mm {
                    d[r] = s[r];
                }
            }
        }
        (VLane::F64(d), VLane::F64(s)) => {
            for (r, &mm) in m.iter().enumerate() {
                if mm {
                    d[r] = s[r];
                }
            }
        }
        (VLane::Str(d), VLane::Str(s)) => {
            for (r, &mm) in m.iter().enumerate() {
                if mm {
                    d[r] = s[r];
                }
            }
        }
        _ => unreachable!("branch arg types checked by the verifier"),
    }
}

// Verified programs make these matches infallible; a miss is a bug in the
// verifier or this compiler, not in the program — same policy as interp.rs.

fn i1s(l: &Option<VLane>) -> &[bool] {
    match l {
        Some(VLane::I1(v)) => v,
        _ => unreachable!("type hole past the verifier: expected an i1 lane"),
    }
}

fn i64s(l: &Option<VLane>) -> &[i64] {
    match l {
        Some(VLane::I64(v)) => v,
        _ => unreachable!("type hole past the verifier: expected an i64 lane"),
    }
}

fn f64s(l: &Option<VLane>) -> &[f64] {
    match l {
        Some(VLane::F64(v)) => v,
        _ => unreachable!("type hole past the verifier: expected an f64 lane"),
    }
}

fn strs(l: &Option<VLane>) -> &[StrRef] {
    match l {
        Some(VLane::Str(v)) => v,
        _ => unreachable!("type hole past the verifier: expected a str lane"),
    }
}

// ------------------------------------------------------------------ tests --

#[cfg(test)]
mod tests {
    use super::super::super::ir::{fixtures, gen, StaticTy, Ty};
    use super::super::interp::{self, CompileError};
    use super::super::testutil::{batch, built, c_f64, c_i1, c_i64, c_str, rows, snapshot};
    use super::super::{Batch, KeyBits, ScalarVal, StaticData, Trap};
    use super::{compile, ColumnarFn};

    fn run_col(f: &ColumnarFn, input: &Batch) -> Result<Vec<Vec<String>>, Trap> {
        let mut st = f.new_state();
        f.run(input, &mut st)?;
        Ok(snapshot(&st))
    }

    // -------------------------------------------------------- fixtures --
    // Expectations copied verbatim from exec/tests.rs (the interpreter's
    // hand-computed values) — the columnar backend must match them.

    #[test]
    fn projection_fixture_executes() {
        let p = built(fixtures::PROJECTION);
        let statics = vec![StaticData::Map(vec![(
            vec![KeyBits::Str("a".into())],
            vec![ScalarVal::F64(10.0)],
        )])];
        let f = compile(&p, statics).unwrap();
        let input = batch(
            3,
            vec![
                c_i64(&[Some(40), None, Some(40)]),
                c_str(&[Some("a"), Some("a"), Some("x")]),
            ],
        );
        // Row 1: 40/10 = 4. Row 2: age NULL -> NULL. Row 3: probe miss -> NULL.
        assert_eq!(
            run_col(&f, &input).unwrap(),
            rows(&[&["4.0"], &["NULL"], &["NULL"]])
        );
    }

    #[test]
    fn filter_fixture_drops_rows() {
        let p = built(fixtures::FILTER);
        let f = compile(&p, vec![]).unwrap();
        let input = batch(3, vec![c_f64(&[Some(1.5), Some(-2.0), Some(0.0)])]);
        assert_eq!(run_col(&f, &input).unwrap(), rows(&[&["1.5"]]));
    }

    #[test]
    fn case_diamond_fixture_executes() {
        let p = built(fixtures::CASE_DIAMOND);
        let f = compile(&p, vec![]).unwrap();
        let input = batch(3, vec![c_i64(&[Some(31), Some(30), Some(5)])]);
        assert_eq!(
            run_col(&f, &input).unwrap(),
            rows(&[&["old", "false"], &["old", "false"], &["young", "false"],])
        );
    }

    #[test]
    fn casts_fixture_executes_and_traps() {
        let p = built(fixtures::CASTS);
        let statics = |snd: StaticData| {
            vec![
                StaticData::Scalar {
                    valid: true,
                    val: ScalarVal::F64(2.5),
                },
                snd,
            ]
        };
        // @1 NULL: n = round(2.5) - trunc(2.5) = 3 - 2 = 1; msg = select over
        // scmp.eq("2.5:", ":") = false -> ":".
        let f = compile(
            &p,
            statics(StaticData::Scalar {
                valid: false,
                val: ScalarVal::I64(0),
            }),
        )
        .unwrap();
        let input = batch(1, vec![c_str(&[Some("12")])]);
        assert_eq!(run_col(&f, &input).unwrap(), rows(&[&["1", ":"]]));

        // @1 = 7: the select picks the static instead.
        let f7 = compile(
            &p,
            statics(StaticData::Scalar {
                valid: true,
                val: ScalarVal::I64(7),
            }),
        )
        .unwrap();
        assert_eq!(run_col(&f7, &input).unwrap(), rows(&[&["7", ":"]]));

        // Unparseable input routes to the trap block and aborts the call.
        let bad = batch(1, vec![c_str(&[Some("xx")])]);
        assert_eq!(
            run_col(&f, &bad).unwrap_err(),
            Trap("cast to i64 failed".into())
        );
    }

    #[test]
    fn kitchen_fixture_executes() {
        let p = built(fixtures::KITCHEN);
        let statics = vec![StaticData::Map(vec![(
            vec![KeyBits::I64(2), KeyBits::Str("k1".into())],
            vec![ScalarVal::F64(0.5), ScalarVal::I64(9)],
        )])];
        let f = compile(&p, statics).unwrap();
        let input = batch(
            2,
            vec![
                c_i64(&[Some(4), Some(10)]),
                c_i64(&[Some(2), None]),
                c_str(&[Some("k1"), Some("nope")]),
                c_f64(&[Some(1.5), Some(-1.0)]),
            ],
        );
        assert_eq!(
            run_col(&f, &input).unwrap(),
            rows(&[&["1.5", "9.0"], &["NULL", "0.0"]])
        );
    }

    // ---------------------------------------------------- trap ordering --

    /// Row 2 traps at a LATE instruction (irem), row 5 at an EARLY one
    /// (idiv). The row loop processes rows in order, so row 2's trap is the
    /// one that surfaces — row order beats instruction order.
    #[test]
    fn trap_surfaces_smallest_row_not_earliest_instruction() {
        let p = built(
            r#"fn f(in: batch{a: i64}, out: batch{o: i64}) {
entry:
  %a = load in.a
  %five = const.i64 5
  %two = const.i64 2
  %ten = const.i64 10
  %d1 = isub %a, %five
  %q1 = idiv %ten, %d1
  %d2 = isub %a, %two
  %q2 = irem %ten, %d2
  %s = iadd %q1, %q2
  store out.o, %s
  emit
}"#,
        );
        let vals: Vec<Option<i64>> = (0..7).map(Some).collect();
        let input = batch(7, vec![c_i64(&vals)]);
        let want = Trap("division by zero in irem".into());
        let f = compile(&p, vec![]).unwrap();
        assert_eq!(run_col(&f, &input).unwrap_err(), want);
        // The interpreter agrees — this IS the row-loop semantics.
        let fi = interp::compile(&p, vec![]).unwrap();
        let mut st = fi.new_state();
        assert_eq!(fi.run(&input, &mut st).unwrap_err(), want);
    }

    /// A brif routes row 0 AWAY from a division whose divisor is 0 only at
    /// row 0 — deactivated rows must not execute kernels, so no trap.
    #[test]
    fn masked_rows_never_execute_kernels() {
        let p = built(
            r#"fn f(in: batch{a: i64}, out: batch{o: i64}) {
entry:
  %a = load in.a
  %zero = const.i64 0
  %isz = icmp.eq %a, %zero
  brif %isz, safe, div
safe:
  %m = const.i64 -1
  store out.o, %m
  emit
div:
  %a2 = load in.a
  %ten = const.i64 10
  %q = idiv %ten, %a2
  store out.o, %q
  emit
}"#,
        );
        let f = compile(&p, vec![]).unwrap();
        let input = batch(3, vec![c_i64(&[Some(0), Some(5), Some(2)])]);
        assert_eq!(run_col(&f, &input).unwrap(), rows(&[&["-1"], &["2"], &["5"]]));
    }

    // -------------------------------------------------------- rejection --

    #[test]
    fn rejects_multiplicity_programs() {
        let p = built(fixtures::MULTI_EXPAND);
        match compile(&p, vec![StaticData::Map(vec![])]) {
            Err(CompileError::Static(msg)) => assert!(
                msg.contains("row-path-only"),
                "unclear reject message: {msg}"
            ),
            Err(e) => panic!("wrong error kind: {e}"),
            Ok(_) => panic!("columnar accepted a multiplicity program"),
        }
    }

    // ----------------------------------------------------- differential --
    // Replicates exec/tests.rs's generated-program harness (its gen_statics/
    // gen_input helpers are private there — minimal local versions).

    fn gen_scalar(rng: &mut gen::Rng, ty: Ty) -> ScalarVal {
        match ty {
            Ty::I1 => ScalarVal::I1(rng.chance(50)),
            Ty::I64 => ScalarVal::I64(rng.next() as i64 % 1000),
            Ty::F64 => ScalarVal::F64((rng.next() as i64 % 1000) as f64 / 4.0),
            Ty::Str => ScalarVal::Str(format!("s{}", rng.below(5))),
        }
    }

    fn gen_statics(rng: &mut gen::Rng, p: &super::Program) -> Vec<StaticData> {
        p.statics
            .iter()
            .map(|st| match st {
                StaticTy::Scalar(ct) => StaticData::Scalar {
                    valid: !ct.nullable || rng.chance(70),
                    val: gen_scalar(rng, ct.ty),
                },
                StaticTy::Map { keys, values } => {
                    let n = if keys[0] == Ty::I1 {
                        2
                    } else {
                        1 + rng.below(3) as usize
                    };
                    let entries = (0..n)
                        .map(|j| {
                            let key: Vec<KeyBits> = keys
                                .iter()
                                .enumerate()
                                .map(|(pos, kt)| match (pos, kt) {
                                    (0, Ty::I1) => KeyBits::I1(j % 2 == 1),
                                    (0, Ty::I64) => KeyBits::I64(j as i64),
                                    (0, Ty::F64) => KeyBits::F64((j as f64).to_bits()),
                                    (0, Ty::Str) => KeyBits::Str(format!("k{j}")),
                                    (_, Ty::I1) => KeyBits::I1(true),
                                    (_, Ty::I64) => KeyBits::I64(7),
                                    (_, Ty::F64) => KeyBits::F64(1.5f64.to_bits()),
                                    (_, Ty::Str) => KeyBits::Str("fix".into()),
                                })
                                .collect();
                            let vals = values.iter().map(|vt| gen_scalar(rng, *vt)).collect();
                            (key, vals)
                        })
                        .collect();
                    StaticData::Map(entries)
                }
                StaticTy::MultiMap { .. } | StaticTy::BatchMap { .. } => {
                    StaticData::Map(Vec::new())
                }
            })
            .collect()
    }

    fn gen_input(rng: &mut gen::Rng, p: &super::Program) -> Batch {
        let rows = rng.below(5) as usize;
        let cols = p
            .in_cols
            .iter()
            .map(|c| {
                let mk_valid = |rng: &mut gen::Rng| !c.ty.nullable || rng.chance(70);
                match c.ty.ty {
                    Ty::I1 => c_i1(
                        &(0..rows)
                            .map(|_| mk_valid(rng).then(|| rng.chance(50)))
                            .collect::<Vec<_>>(),
                    ),
                    Ty::I64 => c_i64(
                        &(0..rows)
                            .map(|_| mk_valid(rng).then(|| rng.next() as i64 % 100_000))
                            .collect::<Vec<_>>(),
                    ),
                    Ty::F64 => c_f64(
                        &(0..rows)
                            .map(|_| {
                                mk_valid(rng).then(|| match rng.below(5) {
                                    0 => f64::NAN,
                                    1 => f64::INFINITY,
                                    _ => (rng.next() as i64 % 1000) as f64 / 8.0,
                                })
                            })
                            .collect::<Vec<_>>(),
                    ),
                    Ty::Str => {
                        let opts: Vec<Option<String>> = (0..rows)
                            .map(|_| {
                                mk_valid(rng).then(|| match rng.below(4) {
                                    0 => String::new(),
                                    1 => "k1".to_string(),
                                    2 => "unicode é".to_string(),
                                    _ => format!("v{}", rng.below(9)),
                                })
                            })
                            .collect();
                        let refs: Vec<Option<&str>> = opts.iter().map(|o| o.as_deref()).collect();
                        c_str(&refs)
                    }
                }
            })
            .collect();
        Batch { rows, cols }
    }

    /// THE backend contract: columnar and the interpreter agree byte-for-
    /// byte on every generated program — outputs, emitted counts, and traps.
    /// The generator emits no multiplicity, so columnar must reject nothing.
    #[test]
    fn differential_columnar_agrees_with_interpreter() {
        let mut rejects = 0usize;
        for seed in 0..500u64 {
            let p = gen::gen_program(seed);
            let mut rng = gen::Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15);
            let statics_i = gen_statics(&mut rng, &p);
            let mut rng2 = gen::Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15);
            let statics_c = gen_statics(&mut rng2, &p);
            let input = gen_input(&mut rng, &p);

            let fi = interp::compile(&p, statics_i).expect("interp compile");
            let fc = match compile(&p, statics_c) {
                Ok(f) => f,
                Err(CompileError::Static(_)) => {
                    rejects += 1;
                    continue;
                }
                Err(e) => panic!("seed {seed}: columnar compile failed: {e}"),
            };
            let mut sti = fi.new_state();
            let a = fi.run(&input, &mut sti).map(|_| snapshot(&sti));
            let mut stc = fc.new_state();
            let b = fc.run(&input, &mut stc).map(|_| snapshot(&stc));
            match (a, b) {
                (Ok(x), Ok(y)) => {
                    assert_eq!(x, y, "seed {seed}: outputs diverge");
                    assert_eq!(
                        sti.emitted, stc.emitted,
                        "seed {seed}: emitted counts diverge"
                    );
                }
                (Err(x), Err(y)) => assert_eq!(x, y, "seed {seed}: traps diverge"),
                (x, y) => {
                    panic!("seed {seed}: outcome diverged: interp {x:?} vs columnar {y:?}")
                }
            }
        }
        assert_eq!(rejects, 0, "generator programs must never be rejected");
    }
}
