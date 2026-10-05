//! How large a program is for Cranelift, measured before anything compiles
//! it.
//!
//! Cranelift numbers a function's virtual registers in 21 bits and refuses
//! a function that needs more (`CodegenError::CodeTooLarge`). Before
//! lowering it assigns one to every block parameter and every instruction
//! result left after its optimizer (`cranelift-codegen` 0.126,
//! `machinst/lower.rs`, "Assign a vreg to each block param, each inst
//! result"), then more for temporaries as it lowers. It gets there late:
//! past bind, lower, verify, the interpreter compile, the CLIF build, its
//! own verifier and its egraph, 26 s on the native catalog's Normalizer
//! (l2, 32 features; 2026-10-05). [`vreg_floor`] is a floor on that count
//! read off the lowered IR in one pass, so a program that cannot fit
//! refuses with the same message as soon as it is lowered (5 s there).

use std::collections::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

use super::super::ir::{BinOp, BlockId, DecOp, Inst, Lit, NumOp1, Program, Term, Ty, Value};

/// Cranelift numbers fewer virtual registers than this in one function:
/// regalloc2's `VReg::MAX` (21 index bits; `VRegAllocator::alloc` refuses an
/// index that reaches it).
pub const VREG_LIMIT: usize = (1 << 21) - 1;

/// What the CLIF build makes of an instruction (`cranelift.rs`,
/// `translate_inst`).
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// A call to a helper (two for a nullable load), with a result: kept by
    /// Cranelift whether the result is read or not, since a call has
    /// effects. One result per call counts, however many values the
    /// instruction defines.
    Call,
    /// One pure CLIF instruction per def (a constant included): Cranelift
    /// keeps it only if something kept reads it, and merges two equal ones
    /// where one dominates the other.
    Pure,
    /// A call to a helper that defines nothing: a store.
    Effect,
}

fn kind(inst: &Inst) -> Kind {
    match inst {
        Inst::Store { .. } | Inst::StoreOpt { .. } => Kind::Effect,
        Inst::Const {
            lit: Lit::Str(_), ..
        } => Kind::Call,
        Inst::Bin { op, .. } => match op {
            BinOp::Fadd
            | BinOp::Fsub
            | BinOp::Fmul
            | BinOp::Fdiv
            | BinOp::Ishr
            | BinOp::Iand
            | BinOp::Ior
            | BinOp::Ixor
            | BinOp::And
            | BinOp::Or
            | BinOp::Xor => Kind::Pure,
            _ => Kind::Call,
        },
        Inst::Cmp {
            ty: Ty::F64 | Ty::Str,
            ..
        } => Kind::Call,
        Inst::Num1 { op, .. } => match op {
            NumOp1::Fabs | NumOp1::Fneg | NumOp1::Ffloor | NumOp1::Fceil | NumOp1::Ftrunc => {
                Kind::Pure
            }
            _ => Kind::Call,
        },
        Inst::Dop {
            op: DecOp::Add | DecOp::Sub | DecOp::Mul,
            check: 0,
            ..
        } => Kind::Pure,
        Inst::Dop { .. }
        | Inst::Dtof { .. }
        | Inst::Dcast { .. }
        | Inst::Dunary { .. }
        | Inst::DcastOk { .. }
        | Inst::Dtos { .. }
        | Inst::Itod { .. }
        | Inst::ExternCall { .. }
        | Inst::Ftoi { .. }
        | Inst::Itos { .. }
        | Inst::Ftos { .. }
        | Inst::StoiOpt { .. }
        | Inst::StofOpt { .. }
        | Inst::Round2f { .. }
        | Inst::Round2i { .. }
        | Inst::Slike { .. }
        | Inst::Str2 { .. }
        | Inst::Str3 { .. }
        | Inst::Str2i { .. }
        | Inst::Spad { .. }
        | Inst::Sslice { .. }
        | Inst::ReMatch { .. }
        | Inst::ReExtract { .. }
        | Inst::ReReplace { .. }
        | Inst::Sord { .. }
        | Inst::SLen { .. }
        | Inst::Sconcat { .. }
        | Inst::Str1 { .. }
        | Inst::Strim { .. }
        | Inst::Ssubstr { .. }
        | Inst::Load { .. }
        | Inst::LoadOpt { .. }
        | Inst::Probe { .. }
        | Inst::Predict { .. }
        | Inst::Sload { .. }
        | Inst::SloadOpt { .. } => Kind::Call,
        // Other constants and compares, not, select, itof, and anything not
        // named above: pure, which can only lower the floor.
        _ => Kind::Pure,
    }
}

/// Whether the floor counts a pure instruction's value: the float
/// arithmetic, which no rule of Cranelift's egraph (0.126 `opts/*.isle`)
/// removes but its constant folding. Everything else pure is a rewrite
/// away from being folded, merged or dropped: a constant, an integer
/// compare (`icmp x, x`, or a `select` over it turned into `smax`), a
/// select (a constant condition, equal arms, a select of a select), the
/// bitwise and logical ops (whose chains it reassociates; measured: a sum
/// of 100 nullable columns in 100 orders keeps 4% fewer values than value
/// numbering their flag `and` chains counts), `fneg` (`fneg (fneg x)`), and
/// a product of two of those (`fmul (fneg x) (fneg y)` is `fmul x y`).
/// Those are not counted; what they read still is.
fn counted(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Bin {
            op: BinOp::Fadd | BinOp::Fsub | BinOp::Fmul | BinOp::Fdiv,
            ..
        } | Inst::Num1 {
            op: NumOp1::Fabs | NumOp1::Ffloor | NumOp1::Fceil | NumOp1::Ftrunc,
            ..
        } | Inst::Itof { .. }
    )
}

/// Whether the build reads the trap flag after the helper call, whatever
/// the operands (`trap_check`): a load, which Cranelift keeps, since the
/// call before it may have written the flag.
fn checks_trap(inst: &Inst) -> bool {
    match inst {
        Inst::Bin { op, .. } => matches!(
            op,
            BinOp::Iadd
                | BinOp::Isub
                | BinOp::Imul
                | BinOp::Idiv
                | BinOp::Irem
                | BinOp::Ishl
                | BinOp::Flogb
        ),
        Inst::Num1 { op, .. } => matches!(
            op,
            NumOp1::Iabs
                | NumOp1::Ln
                | NumOp1::Log2
                | NumOp1::Log10
                | NumOp1::Fsqrt
                | NumOp1::Fsin
                | NumOp1::Fcos
                | NumOp1::Ftan
        ),
        // The checked arm: the pure one is `Kind::Pure`, never here.
        Inst::Dop { .. } => true,
        Inst::Dcast { .. }
        | Inst::ExternCall { .. }
        | Inst::Ftoi { .. }
        | Inst::Slike { .. }
        | Inst::Spad { .. }
        | Inst::Ssubstr { .. }
        | Inst::Predict { .. } => true,
        _ => false,
    }
}

/// Whether Cranelift's egraph may swap the operands of `op` (a pure one).
fn commutes(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Fadd
            | BinOp::Fmul
            | BinOp::Iand
            | BinOp::Ior
            | BinOp::Ixor
            | BinOp::And
            | BinOp::Or
            | BinOp::Xor
    )
}

/// A floor on the virtual registers Cranelift numbers in compiling a program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Floor {
    /// Of those it assigns before lowering, one per block parameter and
    /// instruction result of the optimized function: a floor checked
    /// against that function on every compile with debug assertions.
    pub assigned: usize,
    /// The helper calls, each of which lowering gives a register more: the
    /// helpers are imports, so not colocated, and both hosts' backends load
    /// a far callee's address into a fresh temporary (`load_ext_name`, x64
    /// and aarch64 `lower.isle`, `call` rules). Checked against the calls
    /// of the optimized function too.
    pub calls: usize,
}

impl Floor {
    pub fn total(self) -> usize {
        self.assigned + self.calls
    }
}

/// The floor of `p`, a program Cranelift compiles (one without
/// multiplicity loops, so acyclic).
///
/// `assigned` counts, in the blocks the entry reaches (Cranelift drops the
/// rest):
/// - each helper call (one each, for its result; a nullable load is two),
///   and the trap flag read after one that can trap;
/// - each block parameter, unless every predecessor passes it the same
///   instruction's value, directly or through parameters so removed, which
///   Cranelift's constant-phi removal replaces it by (on the values as
///   built, so the same value, not an equal one; one passed a parameter it
///   keeps, it keeps too);
/// - each float arithmetic value ([`counted`]) that a call, a store, a
///   branch or a kept parameter reads, directly or through other pure
///   values (Cranelift's egraph keeps no other), two pure instructions of
///   the same operation over the same operands counted once (its value
///   numbering, done here over the whole function, so more generously than
///   Cranelift, which merges a copy only where another dominates it; two
///   whose hashes collide count once too).
///
/// Each count is a value Cranelift assigns at least one register, and where
/// this cannot tell it counts less: a value the build splits in two (a
/// string, a DECIMAL) is one, and the build's own extra values (a call's
/// out-cell, the constants it passes) are none.
pub fn vreg_floor(p: &Program) -> Floor {
    let ids = p
        .blocks
        .iter()
        .flat_map(|b| {
            b.params
                .iter()
                .map(|(v, _)| *v)
                .chain(b.insts.iter().flat_map(|i| i.dsts()))
        })
        .map(|v| v.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let at = |v: Value| v.0 as usize;
    // Each successor of a terminator, with the arguments it passes.
    fn succs(t: &Term) -> [Option<(BlockId, &[Value])>; 2] {
        match t {
            Term::Jump { to, args } | Term::EmitTo { to, args } => [Some((*to, args)), None],
            Term::Brif {
                then_to,
                then_args,
                else_to,
                else_args,
                ..
            } => [Some((*then_to, then_args)), Some((*else_to, else_args))],
            Term::Emit | Term::Skip | Term::Trap { .. } => [None, None],
        }
    }
    // The reached blocks in reverse postorder: on these acyclic programs a
    // topological order, so every value is numbered before it is read.
    let mut order: Vec<usize> = Vec::new();
    let mut reached = vec![false; p.blocks.len()];
    let mut stack: Vec<(usize, bool)> = vec![(0, false)];
    while let Some((b, done)) = stack.pop() {
        if done {
            order.push(b);
            continue;
        }
        if b >= p.blocks.len() || reached[b] {
            continue;
        }
        reached[b] = true;
        stack.push((b, true));
        for (to, _) in succs(&p.blocks[b].term).into_iter().flatten() {
            stack.push((to.0 as usize, false));
        }
    }
    order.reverse();
    // Per block: the reached predecessors' edges into it.
    let mut preds: Vec<Vec<(usize, usize)>> = vec![Vec::new(); p.blocks.len()];
    for &bi in &order {
        for (k, s) in succs(&p.blocks[bi].term).into_iter().enumerate() {
            if let Some((to, _)) = s {
                if let Some(ps) = preds.get_mut(to.0 as usize) {
                    ps.push((bi, k));
                }
            }
        }
    }
    let incoming = |bi: usize, i: usize| {
        preds[bi].iter().filter_map(move |&(from, k)| {
            succs(&p.blocks[from].term)[k].and_then(|(_, args)| args.get(i).copied())
        })
    };

    const NONE: u32 = u32::MAX;
    // Per value: the value it stands for once constant phis are removed.
    let mut same: Vec<Value> = (0..ids as u32).map(Value).collect();
    // Per value: its number (two pure values with one number are one).
    let mut vn: Vec<u32> = vec![NONE; ids];
    // Per pure value: where its operands are in `operands`, and whether it
    // counts (see `counted`; a constant, or what the egraph folds from
    // constants only, does not).
    let mut reads: Vec<(u32, u32)> = vec![(0, 0); ids];
    let mut operands: Vec<Value> = Vec::new();
    let mut pure: Vec<bool> = vec![false; ids];
    // Per value: a block parameter Cranelift keeps.
    let mut param: Vec<bool> = vec![false; ids];
    let mut counts: Vec<bool> = vec![false; ids];
    let mut constant: Vec<bool> = vec![false; ids];
    // Per value: an `fneg`'s.
    let mut negated: Vec<bool> = vec![false; ids];
    let mut next: u32 = 0;
    let mut fresh = || {
        next += 1;
        next - 1
    };
    let mut count = 0usize;
    let mut calls = 0usize;
    // Values something kept reads.
    let mut work: Vec<Value> = Vec::new();
    // Pure instructions numbered so far, by a hash of their operation and
    // operand numbers. Two that collide are taken for one: that lowers the
    // floor, so it stays one.
    let mut seen: HashMap<u64, u32, Fx> = HashMap::default();
    let mut uses: Vec<Value> = Vec::new();
    for &bi in &order {
        let b = &p.blocks[bi];
        for (i, (v, _)) in b.params.iter().enumerate() {
            // What flows in along an edge, as Cranelift's solver reads it:
            // an instruction's value is itself, a removed parameter the
            // value it stands for, and a kept one many values (`None`).
            let flow = |a: Value| match same.get(at(a)) {
                Some(s) if *s != a => Some(*s),
                _ if param.get(at(a)) == Some(&true) => None,
                _ => Some(a),
            };
            let mut flows = incoming(bi, i).map(flow);
            let one = match flows.next() {
                Some(Some(f)) if flows.all(|r| r == Some(f)) => Some(f),
                _ => None,
            }
            .filter(|f| vn.get(at(*f)).is_some_and(|n| *n != NONE));
            match one {
                Some(f) => {
                    same[at(*v)] = f;
                    vn[at(*v)] = vn[at(f)];
                }
                None => {
                    param[at(*v)] = true;
                    vn[at(*v)] = fresh();
                    count += 1;
                    work.extend(incoming(bi, i));
                }
            }
        }
        for inst in &b.insts {
            let dsts = inst.dsts();
            uses.clear();
            {
                let cell = std::cell::RefCell::new(std::mem::take(&mut uses));
                inst.clone().map_values(&|v| {
                    if !dsts.contains(&v) {
                        cell.borrow_mut().push(v);
                    }
                    v
                });
                uses = cell.into_inner();
            }
            match kind(inst) {
                Kind::Effect => {
                    calls += 1;
                    work.extend(uses.iter().copied());
                }
                Kind::Call => {
                    let n = match inst {
                        Inst::LoadOpt { .. } => 2,
                        _ => 1,
                    };
                    calls += n;
                    count += usize::from(!dsts.is_empty()) * n + usize::from(checks_trap(inst));
                    work.extend(uses.iter().copied());
                    for d in &dsts {
                        vn[at(*d)] = fresh();
                    }
                }
                Kind::Pure => {
                    let n = |v: &Value| vn.get(at(*v)).copied().unwrap_or(NONE);
                    let mut ops: Vec<u32> = uses.iter().map(n).collect();
                    if matches!(inst, Inst::Bin { op, .. } if commutes(*op)) {
                        // Cranelift's egraph orders a commutative operation's
                        // operands: so does the floor.
                        ops.sort_unstable();
                    }
                    // ... and reads `x & x`, `x | x` and a select between
                    // equal arms as `x`.
                    let alias = match inst {
                        Inst::Bin {
                            op: BinOp::Iand | BinOp::Ior | BinOp::And | BinOp::Or,
                            a,
                            b,
                            ..
                        } if n(a) == n(b) => Some(*a),
                        Inst::Select { a, b, .. } if n(a) == n(b) => Some(*a),
                        _ => None,
                    };
                    let all = |of: &[bool]| {
                        !uses.is_empty() && uses.iter().all(|u| of.get(at(*u)) == Some(&true))
                    };
                    let folds = matches!(inst, Inst::Const { .. }) || all(&constant);
                    // `fmul (fneg x) (fneg y)` is rewritten to `fmul x y`,
                    // which may be one already there.
                    let unnegates = matches!(
                        inst,
                        Inst::Bin {
                            op: BinOp::Fmul,
                            ..
                        }
                    ) && all(&negated);
                    let number = match alias {
                        Some(a) if n(&a) != NONE => n(&a),
                        _ if ops.contains(&NONE) => fresh(),
                        _ => {
                            let mut h = Fx::default().build_hasher();
                            std::mem::discriminant(inst).hash(&mut h);
                            immediates(inst, &mut h);
                            ops.hash(&mut h);
                            *seen.entry(h.finish()).or_insert_with(&mut fresh)
                        }
                    };
                    let start = operands.len() as u32;
                    operands.extend(uses.iter().copied());
                    for d in &dsts {
                        vn[at(*d)] = number;
                        reads[at(*d)] = (start, uses.len() as u32);
                        pure[at(*d)] = true;
                        counts[at(*d)] = alias.is_none() && !folds && !unnegates && counted(inst);
                        constant[at(*d)] = folds;
                        negated[at(*d)] = matches!(
                            inst,
                            Inst::Num1 {
                                op: NumOp1::Fneg,
                                ..
                            }
                        );
                    }
                }
            }
        }
        if let Term::Brif { cond, .. } = &b.term {
            work.push(*cond);
        }
    }
    // The pure values something kept reads, each number once.
    let mut live = vec![false; ids];
    let mut numbered = vec![false; next as usize];
    while let Some(v) = work.pop() {
        if at(v) >= ids || live[at(v)] {
            continue;
        }
        live[at(v)] = true;
        if same[at(v)] != v {
            work.push(same[at(v)]);
        } else if pure[at(v)] {
            if counts[at(v)] && !numbered[vn[at(v)] as usize] {
                numbered[vn[at(v)] as usize] = true;
                count += 1;
            }
            let (start, len) = reads[at(v)];
            work.extend_from_slice(&operands[start as usize..(start + len) as usize]);
        }
    }
    Floor {
        assigned: count,
        calls,
    }
}

/// What, beside its operands, makes two pure instructions of one kind
/// different.
fn immediates(inst: &Inst, h: &mut impl Hasher) {
    match inst {
        Inst::Const { lit, .. } => match lit {
            Lit::I1(x) => (0u8, u64::from(*x)).hash(h),
            Lit::I64(x) => (1u8, *x).hash(h),
            Lit::F64(x) => (2u8, x.to_bits()).hash(h),
            Lit::Str(x) => (3u8, x).hash(h),
            Lit::Dec(x, ps, sc) => (4u8, *x, *ps, *sc).hash(h),
        },
        Inst::Bin { op, .. } => std::mem::discriminant(op).hash(h),
        Inst::Cmp { pred, ty, .. } => (std::mem::discriminant(pred), ty).hash(h),
        Inst::Num1 { op, .. } => std::mem::discriminant(op).hash(h),
        Inst::Itof { narrow, .. } => narrow.hash(h),
        Inst::Dop { op, check, ty, .. } => (std::mem::discriminant(op), check, ty).hash(h),
        _ => {}
    }
}

/// FxHash (rustc's): the floor hashes every instruction of programs with
/// millions of them, where SipHash is most of its time.
type Fx = BuildHasherDefault<FxHasher>;

#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.write_u64(u64::from(*b));
        }
    }
    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::ir::{Col, ColTy, Ty};
    use super::super::cranelift::{self, ASSIGNED};
    use super::super::testutil::built;
    use super::vreg_floor;

    /// The floor of `text` (what Cranelift assigns before lowering), checked
    /// against what it assigns and calls.
    fn floor(text: &str) -> usize {
        let p = built(text);
        let f = vreg_floor(&p);
        cranelift::compile(&p, vec![]).expect("cranelift compile");
        let (assigned, calls) = ASSIGNED.with(|c| c.get());
        assert!(
            f.assigned <= assigned && f.calls <= calls,
            "floor {f:?}, assigned {assigned}, calls {calls}\n{text}"
        );
        f.assigned
    }

    #[test]
    fn a_call_counts_read_or_not_and_a_pure_value_only_when_read() {
        // Two loads (calls), one read multiply; the add is never read.
        let text = r#"
fn f(in: batch{x: f64, s: str}, out: batch{o: f64}) {
entry:
  %x = load in.x
  %s = load in.s
  %dead = fadd %x, %x
  %y = fmul %x, %x
  store out.o, %y
  emit
}
"#;
        assert_eq!(floor(text), 3);
    }

    #[test]
    fn equal_pure_values_count_once_commuted_or_not() {
        // a+b and b+a are one value; a-b and b-a are two.
        let text = r#"
fn f(in: batch{a: f64, b: f64}, out: batch{o: f64, p: f64, q: f64, r: f64}) {
entry:
  %a = load in.a
  %b = load in.b
  %s = fadd %a, %b
  %t = fadd %b, %a
  %u = fsub %a, %b
  %v = fsub %b, %a
  store out.o, %s
  store out.p, %t
  store out.q, %u
  store out.r, %v
  emit
}
"#;
        assert_eq!(floor(text), 2 + 1 + 2);
    }

    #[test]
    fn a_parameter_every_predecessor_passes_the_same_value_is_that_value() {
        // Values cross blocks only as branch args, so `l` and `r` take both
        // and pass one on: the same one (every parameter resolves to `x`),
        // or each its own (`j`'s parameter is a value of its own, and so
        // are the `l` and `r` ones it reads).
        let join = |left: &str, right: &str| {
            format!(
                r#"
fn f(in: batch{{x: f64, y: f64, c: i1}}, out: batch{{o: f64}}) {{
entry:
  %x = load in.x
  %y = load in.y
  %c = load in.c
  brif %c, l(%x, %y), r(%x, %y)
l(%lx: f64, %ly: f64):
  jump j(%{left})
r(%rx: f64, %ry: f64):
  jump j(%{right})
j(%v: f64):
  store out.o, %v
  emit
}}
"#
            )
        };
        assert_eq!(floor(&join("lx", "rx")), 3);
        assert_eq!(floor(&join("lx", "ry")), 3 + 1);
    }

    #[test]
    fn a_helper_that_can_trap_counts_the_flag_read_after_it() {
        let text = r#"
fn f(in: batch{x: f64}, out: batch{o: i64}) {
entry:
  %x = load in.x
  %i = ftoi.trunc %x
  store out.o, %i
  emit
}
"#;
        assert_eq!(floor(text), 1 + 2);
    }

    /// SQL shapes of every kind the floor counts, each checked against what
    /// Cranelift assigns (`floor`'s assertion; with debug assertions every
    /// compile checks it too). Measured on these at 10-100x the size
    /// (2026-10-05): Cranelift assigns 1.0-14x the floor, 1.35x on the
    /// native catalog's Normalizer, and allocates 1.7-2.1x what it assigns
    /// up front by the end of lowering.
    #[test]
    fn the_floor_holds_on_large_sql_shapes() {
        let col = |name: String, ty: Ty, nullable: bool| Col {
            name,
            ty: ColTy { ty, nullable },
        };
        let n = 12;
        let mut cols: Vec<Col> = (0..n)
            .map(|i| col(format!("x{i}"), Ty::F64, true))
            .collect();
        cols.extend((0..n).map(|i| col(format!("k{i}"), Ty::F64, false)));
        cols.extend((0..3).map(|i| col(format!("s{i}"), Ty::Str, true)));
        cols.push(col("d".into(), Ty::Dec(9, 2), true));
        // A fixed shuffle per column, so value numbering cannot line two up.
        let order = |j: usize| -> Vec<usize> {
            let mut o: Vec<usize> = (0..n).collect();
            let mut s = j as u64 * 0x9E37_79B9 + 1;
            for i in (1..n).rev() {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                o.swap(i, (s >> 33) as usize % (i + 1));
            }
            o
        };
        let sum = |p: &str, j: usize| {
            order(j)
                .iter()
                .map(|i| format!("{p}{i}"))
                .collect::<Vec<_>>()
                .join(" + ")
        };
        let norm = (0..n)
            .map(|i| format!("abs(coalesce(x{i}, 'nan'::DOUBLE))"))
            .collect::<Vec<_>>()
            .join(" + ");
        let shapes: Vec<String> = vec![
            (0..n)
                .map(|j| format!("{} AS o{j}", sum("x", j)))
                .collect::<Vec<_>>()
                .join(", "),
            (0..n)
                .map(|j| format!("{} AS o{j}", sum("k", j)))
                .collect::<Vec<_>>()
                .join(", "),
            (0..n)
                .map(|j| {
                    let terms: Vec<String> = (0..n)
                        .map(|i| format!("k{i} * {}.{j}{i}e0", i + 1))
                        .collect();
                    format!("{} AS o{j}", terms.join(" + "))
                })
                .collect::<Vec<_>>()
                .join(", "),
            (0..n)
                .map(|j| {
                    format!(
                        "coalesce(x{j}, 'nan'::DOUBLE) / CASE WHEN ({norm}) < 1e-15 \
                         THEN 1.0e0 ELSE ({norm}) END AS o{j}"
                    )
                })
                .collect::<Vec<_>>()
                .join(", "),
            (0..n)
                .map(|j| {
                    let arms: Vec<String> = (0..n)
                        .map(|i| format!("WHEN x{i} > {j}.{i}e0 THEN k{} * 0.5e0", (i + j) % n))
                        .collect();
                    format!("CASE {} ELSE 0.0e0 END AS o{j}", arms.join(" "))
                })
                .collect::<Vec<_>>()
                .join(", "),
            (0..n)
                .map(|j| format!("CAST(k{j} * 1.5e0 AS INTEGER) + CAST(x{j} AS BIGINT) AS o{j}"))
                .collect::<Vec<_>>()
                .join(", "),
            "length(upper(s0) || s1 || lower(s2)) AS a, s0 < s1 AS b, d * 1.25 + d AS c".into(),
        ];
        for items in shapes {
            let sql = format!("SELECT {items} FROM t");
            let p = crate::specializer::prepare(&sql, "t", &cols, &[])
                .unwrap_or_else(|e| panic!("{e}\n{sql}"))
                .program;
            let f = vreg_floor(&p);
            cranelift::compile(&p, vec![]).unwrap_or_else(|e| panic!("{e}\n{sql}"));
            let (assigned, calls) = ASSIGNED.with(|c| c.get());
            assert!(
                f.assigned > 0 && f.assigned <= assigned && f.calls <= calls,
                "floor {f:?}, assigned {assigned}, calls {calls}\n{sql}"
            );
        }
    }
}
