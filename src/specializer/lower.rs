//! Produce/consume lowering: relational IR -> imperative IR.
//!
//! The central mechanism is the env-threading function builder [`FB`]:
//! CASE and guarded CASTs split the CFG, and the IR's strict block-param SSA
//! (values never cross blocks except as branch args) means every live value
//! must ride the branch when a split happens. `emit` therefore carries a
//! live stack: recursive children push their partial lanes before a sibling
//! is evaluated, and any block transition rewrites the stack in place to the
//! target block's params. Columns are NOT threaded — loads are pure and each
//! block re-loads through its own cache.
//!
//! Null-lane discipline: an SExpr with `nullable == false` lowers to a bare
//! payload register (no flag anywhere); a nullable one carries an `i1` flag,
//! combined with `and` where NULL propagates. Kleene AND/OR are lowered
//! branchless from flag algebra. WHERE keeps a row iff the predicate is
//! TRUE — flag && value.
//!
//! CASE lowers to a condition chain with a join block; branch results are
//! evaluated only on their taken path (a guarded `1/0`-style branch must not
//! trap eagerly — SQL semantics, verified by test). CAST failure is a
//! conditional trap block; TRY_CAST folds the failure into the null lane.
//!
//! ponytail: blocks re-load columns instead of receiving them as branch args
//! — pure loads, identical semantics, simpler lowering. Thread them when the
//! Cranelift backend makes the extra reads measurable.

use std::collections::HashMap;

use super::frontend::PrepareError;
use super::ir::{
    BinOp, Block, BlockId, Builder, CmpPred, Col, Inst, Lit, NumOp1, Program, StaticTy, StrOp1,
    StrOp2, Term, Ty, Value,
};
use super::plan::{ArithOp, JoinKind, JoinSpec, Rel, SExpr, SKind, StaticTable};

pub fn lower(
    rel: &Rel,
    joins: &[JoinSpec],
    catalog: &[StaticTable],
    in_cols: &[Col],
    out_cols: Vec<Col>,
    name: &str,
) -> Result<Program, PrepareError> {
    let (exprs, filter_pred) = match rel {
        Rel::Project { input, exprs } => match input.as_ref() {
            Rel::Filter { input: scan, pred } => {
                debug_assert!(matches!(scan.as_ref(), Rel::Scan));
                (exprs, Some(pred))
            }
            Rel::Scan => (exprs, None),
            Rel::Project { .. } => {
                return Err(PrepareError::Internal("nested projection".to_string()))
            }
        },
        _ => {
            return Err(PrepareError::Internal(
                "plan root is not a projection".to_string(),
            ))
        }
    };

    let mut fb = FB::new(in_cols, joins);

    // Joins run before WHERE (SQL order), each in FROM order: probe, and for
    // INNER skip the row on a miss. A LEFT join's probe is also forced here
    // so its key expressions are evaluated (and can trap) for every row that
    // reaches it, exactly as the join would — even if no value column is
    // ever referenced. Later references re-probe per block (pure, cached),
    // same ponytail trade as column re-loads.
    for (j, spec) in joins.iter().enumerate() {
        let mut live = Vec::new();
        let (valid_hit, _) = fb.emit_probe(j as u32, &mut live)?;
        if spec.kind == JoinKind::Inner {
            let (keep, _) = fb.create_block(&[]);
            let (miss, _) = fb.create_block(&[]);
            fb.term(Term::Brif {
                cond: valid_hit,
                then_to: BlockId(keep as u32),
                then_args: vec![],
                else_to: BlockId(miss as u32),
                else_args: vec![],
            });
            fb.switch(miss);
            fb.term(Term::Skip);
            fb.switch(keep);
        }
    }

    if let Some(pred) = filter_pred {
        let mut live = Vec::new();
        let pl = fb.emit(pred, &mut live)?;
        let cond = fb.truthy(pl);
        let (keep, _) = fb.create_block(&[]);
        let (drop, _) = fb.create_block(&[]);
        fb.term(Term::Brif {
            cond,
            then_to: BlockId(keep as u32),
            then_args: vec![],
            else_to: BlockId(drop as u32),
            else_args: vec![],
        });
        fb.switch(drop);
        fb.term(Term::Skip);
        fb.switch(keep);
    }

    let mut live = Vec::new();
    for (ci, (_, e)) in exprs.iter().enumerate() {
        debug_assert!(live.is_empty());
        let lane = fb.emit(e, &mut live)?;
        let col = ci as u32;
        if out_cols[ci].ty.nullable {
            let flag = match lane.flag {
                Some(f) => f,
                // Nullability contract slack (e.g. an infallible TRY_CAST):
                // the column is declared nullable, the lane is provably
                // valid — store with a constant true flag.
                None => fb.const_i1(true),
            };
            fb.inst(Inst::StoreOpt {
                col,
                flag,
                val: lane.val,
            });
        } else {
            debug_assert!(lane.flag.is_none(), "non-nullable column with a flag lane");
            fb.inst(Inst::Store { col, val: lane.val });
        }
    }
    fb.term(Term::Emit);

    // Map static @j belongs to join j: keyed by the (already promoted) key
    // expression types, valued by the join's value columns.
    let statics = joins
        .iter()
        .map(|spec| StaticTy::Map {
            keys: spec.keys.iter().map(|k| k.ty).collect(),
            values: spec
                .val_cols
                .iter()
                .map(|&c| catalog[spec.table].cols[c as usize].ty.ty)
                .collect(),
        })
        .collect();

    fb.finish(name, statics, in_cols, out_cols)
}

/// A value in the null-lane representation: payload + optional validity.
/// `flag: None` means provably non-NULL (no flag register exists).
#[derive(Clone, Copy)]
struct Lane {
    flag: Option<Value>,
    val: Value,
}

/// The live stack: lanes (with their payload types) that must survive block
/// transitions triggered while a sibling expression is being emitted.
type Live = Vec<(Lane, Ty)>;

/// A block under construction.
struct PB {
    params: Vec<(Value, Ty)>,
    insts: Vec<Inst>,
    term: Option<Term>,
    cache: HashMap<u32, Lane>,
    /// Probes already emitted in this block: join index -> (valid hit —
    /// map hit AND every key flag — and the value-column registers).
    probes: HashMap<u32, (Value, Vec<Value>)>,
}

impl PB {
    fn new(params: Vec<(Value, Ty)>) -> PB {
        PB {
            params,
            insts: vec![],
            term: None,
            cache: HashMap::new(),
            probes: HashMap::new(),
        }
    }
}

/// Env-threading function builder over the strict block-param SSA IR.
struct FB<'a> {
    b: Builder,
    blocks: Vec<PB>,
    cur: usize,
    in_cols: &'a [Col],
    joins: &'a [JoinSpec],
}

impl<'a> FB<'a> {
    fn new(in_cols: &'a [Col], joins: &'a [JoinSpec]) -> FB<'a> {
        FB {
            b: Builder::new(),
            blocks: vec![PB::new(vec![])],
            cur: 0,
            in_cols,
            joins,
        }
    }

    fn fresh(&mut self) -> Value {
        self.b.fresh()
    }

    fn inst(&mut self, i: Inst) {
        self.blocks[self.cur].insts.push(i);
    }

    fn term(&mut self, t: Term) {
        let slot = &mut self.blocks[self.cur].term;
        debug_assert!(slot.is_none(), "block terminated twice");
        *slot = Some(t);
    }

    fn switch(&mut self, blk: usize) {
        self.cur = blk;
    }

    fn create_block(&mut self, param_tys: &[Ty]) -> (usize, Vec<Value>) {
        let params: Vec<(Value, Ty)> = param_tys.iter().map(|t| (self.b.fresh(), *t)).collect();
        let vals = params.iter().map(|(v, _)| *v).collect();
        self.blocks.push(PB::new(params));
        (self.blocks.len() - 1, vals)
    }

    fn finish(
        self,
        name: &str,
        statics: Vec<StaticTy>,
        in_cols: &[Col],
        out_cols: Vec<Col>,
    ) -> Result<Program, PrepareError> {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for (i, pb) in self.blocks.into_iter().enumerate() {
            let term = pb
                .term
                .ok_or_else(|| PrepareError::Internal(format!("block b{i} left unterminated")))?;
            blocks.push(Block {
                params: pb.params,
                insts: pb.insts,
                term,
            });
        }
        Ok(Program {
            statics,
            name: name.to_string(),
            in_cols: in_cols.to_vec(),
            out_cols,
            blocks,
        })
    }

    // ------------------------------------------------------ conveniences --

    fn const_lit(&mut self, lit: Lit) -> Value {
        let dst = self.fresh();
        self.inst(Inst::Const { dst, lit });
        dst
    }

    fn const_i1(&mut self, v: bool) -> Value {
        self.const_lit(Lit::I1(v))
    }

    /// Trapping instructions must never fire on the garbage payload under a
    /// false NULL flag — computed garbage is unbounded (`x + MAX + MAX` with
    /// x NULL overflows its payload). Mask nullable payloads to the type
    /// default right before any trapping instruction.
    fn masked(&mut self, l: Lane, ty: Ty) -> Value {
        match l.flag {
            None => l.val,
            Some(f) => {
                let d = self.default_of(ty);
                self.select_of(f, l.val, d)
            }
        }
    }

    /// Like `masked`, but to an op-specific SAFE constant: the type default
    /// (0.0) is itself in the trap domain of ln/logb, so those mask to a
    /// value the op accepts (the result is discarded under the false flag).
    fn masked_to(&mut self, l: Lane, safe: Lit) -> Value {
        match l.flag {
            None => l.val,
            Some(f) => {
                let d = self.const_lit(safe);
                self.select_of(f, l.val, d)
            }
        }
    }

    fn select_of(&mut self, cond: Value, a: Value, b: Value) -> Value {
        let dst = self.fresh();
        self.inst(Inst::Select { dst, cond, a, b });
        dst
    }

    fn default_of(&mut self, ty: Ty) -> Value {
        self.const_lit(match ty {
            Ty::I1 => Lit::I1(false),
            Ty::I64 => Lit::I64(0),
            Ty::F64 => Lit::F64(0.0),
            Ty::Str => Lit::Str(String::new()),
        })
    }

    fn bin(&mut self, op: BinOp, a: Value, b: Value) -> Value {
        let dst = self.fresh();
        self.inst(Inst::Bin { op, dst, a, b });
        dst
    }

    fn not(&mut self, a: Value) -> Value {
        let dst = self.fresh();
        self.inst(Inst::Not { dst, a });
        dst
    }

    /// The SQL truth of a boolean lane: TRUE iff valid AND value.
    fn truthy(&mut self, lane: Lane) -> Value {
        match lane.flag {
            None => lane.val,
            Some(f) => self.bin(BinOp::And, f, lane.val),
        }
    }

    // -------------------------------------------------- live threading --

    /// The flattened param type shape of a live stack: per lane, an i1 flag
    /// (when present) then the payload.
    fn live_types(live: &Live) -> Vec<Ty> {
        let mut tys = Vec::new();
        for (lane, ty) in live {
            if lane.flag.is_some() {
                tys.push(Ty::I1);
            }
            tys.push(*ty);
        }
        tys
    }

    fn live_args(live: &Live) -> Vec<Value> {
        let mut args = Vec::new();
        for (lane, _) in live {
            if let Some(f) = lane.flag {
                args.push(f);
            }
            args.push(lane.val);
        }
        args
    }

    /// Rebind the live stack to the params of the block just switched to.
    /// The shape (lane count + flag pattern) is invariant across transitions.
    fn rebind_live(live: &mut Live, params: &[Value]) {
        let mut i = 0;
        for (lane, _) in live.iter_mut() {
            if lane.flag.is_some() {
                lane.flag = Some(params[i]);
                i += 1;
            }
            lane.val = params[i];
            i += 1;
        }
        debug_assert_eq!(i, params.len());
    }

    // ------------------------------------------------------- expressions --

    fn emit(&mut self, e: &SExpr, live: &mut Live) -> Result<Lane, PrepareError> {
        match &e.kind {
            SKind::Col(idx) => {
                if let Some(lane) = self.blocks[self.cur].cache.get(idx) {
                    return Ok(*lane);
                }
                let col = &self.in_cols[*idx as usize];
                let lane = if col.ty.nullable {
                    let flag = self.fresh();
                    let val = self.fresh();
                    self.inst(Inst::LoadOpt {
                        flag,
                        dst: val,
                        col: *idx,
                    });
                    Lane {
                        flag: Some(flag),
                        val,
                    }
                } else {
                    let val = self.fresh();
                    self.inst(Inst::Load {
                        dst: val,
                        col: *idx,
                    });
                    Lane { flag: None, val }
                };
                self.blocks[self.cur].cache.insert(*idx, lane);
                Ok(lane)
            }
            SKind::StaticCol { join, col } => {
                let (valid_hit, dsts) = self.emit_probe(*join, live)?;
                let flag = match self.joins[*join as usize].kind {
                    // INNER: a miss already skipped the row before any
                    // expression could look — the lane is provably valid.
                    JoinKind::Inner => None,
                    JoinKind::Left => Some(valid_hit),
                };
                Ok(Lane {
                    flag,
                    val: dsts[*col as usize],
                })
            }
            SKind::Lit(lit) => Ok(Lane {
                flag: None,
                val: self.const_lit(lit.clone()),
            }),
            SKind::NullOf => {
                let flag = self.const_i1(false);
                let val = self.default_of(e.ty);
                Ok(Lane {
                    flag: Some(flag),
                    val,
                })
            }
            SKind::IntToFloat(inner) => {
                let l = self.emit(inner, live)?;
                let dst = self.fresh();
                self.inst(Inst::Itof { dst, a: l.val });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::Arith { op, a, b } => {
                let la = self.emit(a, live)?;
                live.push((la, a.ty));
                let lb = self.emit(b, live)?;
                let (la, _) = live.pop().expect("pushed above");
                let ir_op = match (op, e.ty) {
                    (ArithOp::Add, Ty::I64) => BinOp::Iadd,
                    (ArithOp::Sub, Ty::I64) => BinOp::Isub,
                    (ArithOp::Mul, Ty::I64) => BinOp::Imul,
                    (ArithOp::Rem, Ty::I64) => BinOp::Irem,
                    (ArithOp::IDiv, Ty::I64) => BinOp::Idiv,
                    (ArithOp::Add, Ty::F64) => BinOp::Fadd,
                    (ArithOp::Sub, Ty::F64) => BinOp::Fsub,
                    (ArithOp::Mul, Ty::F64) => BinOp::Fmul,
                    (ArithOp::Div, Ty::F64) => BinOp::Fdiv,
                    // `//` on doubles is PLAIN division (wave-3 pins); the
                    // zero-divisor NULL guard is the frontend's CASE wrap.
                    (ArithOp::IDiv, Ty::F64) => BinOp::Fdiv,
                    (ArithOp::Rem, Ty::F64) => BinOp::Frem,
                    (op, ty) => {
                        return Err(PrepareError::Internal(format!(
                            "arith {op:?} on {} escaped the frontend",
                            ty.name()
                        )))
                    }
                };
                // Integer arithmetic traps (overflow, % edge cases): mask
                // nullable payloads so garbage under a false flag can never
                // fire the trap. Float ops are total — no masking needed.
                let (va, vb) = if e.ty == Ty::I64 {
                    (self.masked(la, Ty::I64), self.masked(lb, Ty::I64))
                } else {
                    (la.val, lb.val)
                };
                let val = self.bin(ir_op, va, vb);
                Ok(Lane {
                    flag: self.combine_flags(la.flag, lb.flag),
                    val,
                })
            }
            SKind::Cmp { pred, a, b } => {
                let la = self.emit(a, live)?;
                live.push((la, a.ty));
                let lb = self.emit(b, live)?;
                let (la, _) = live.pop().expect("pushed above");
                let dst = self.fresh();
                self.inst(Inst::Cmp {
                    pred: *pred,
                    ty: a.ty,
                    dst,
                    a: la.val,
                    b: lb.val,
                });
                Ok(Lane {
                    flag: self.combine_flags(la.flag, lb.flag),
                    val: dst,
                })
            }
            SKind::Not(inner) => {
                let l = self.emit(inner, live)?;
                let val = self.not(l.val);
                Ok(Lane { flag: l.flag, val })
            }
            SKind::And { a, b } => self.kleene(a, b, live, true),
            SKind::Or { a, b } => self.kleene(a, b, live, false),
            SKind::IsNull { negated, inner } => {
                let l = self.emit(inner, live)?;
                let val = match l.flag {
                    None => self.const_i1(*negated),
                    Some(f) => {
                        if *negated {
                            f
                        } else {
                            self.not(f)
                        }
                    }
                };
                Ok(Lane { flag: None, val })
            }
            SKind::Case { arms, default } => self.case(e, arms, default.as_deref(), live),
            SKind::Cast { inner, trying } => self.cast(e, inner, *trying, live),
            SKind::StrCase { upper, a } => {
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                let op = if *upper { StrOp1::Upper } else { StrOp1::Lower };
                self.inst(Inst::Str1 { op, dst, a: l.val });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::Trim { side, a, chars } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let lc = self.emit(chars, live)?;
                let (la, _) = live.pop().expect("pushed above");
                let dst = self.fresh();
                self.inst(Inst::Strim {
                    side: *side,
                    dst,
                    a: la.val,
                    chars: lc.val,
                });
                Ok(Lane {
                    flag: self.combine_flags(la.flag, lc.flag),
                    val: dst,
                })
            }
            SKind::Substr { a, start, len } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let ls = self.emit(start, live)?;
                live.push((ls, Ty::I64));
                let ll = match len {
                    Some(l) => Some(self.emit(l, live)?),
                    None => None,
                };
                let (ls, _) = live.pop().expect("pushed above");
                let (la, _) = live.pop().expect("pushed above");
                // The range guards trap; mask nullable position payloads.
                let start_v = self.masked(ls, Ty::I64);
                let len_v = ll.map(|l| self.masked(l, Ty::I64));
                let dst = self.fresh();
                self.inst(Inst::Ssubstr {
                    dst,
                    a: la.val,
                    start: start_v,
                    len: len_v,
                });
                let flag = self.combine_flags(la.flag, ls.flag);
                Ok(Lane {
                    flag: self.combine_flags(flag, ll.and_then(|l| l.flag)),
                    val: dst,
                })
            }
            SKind::Abs(a) => {
                let l = self.emit(a, live)?;
                // iabs traps on i64::MIN; mask the nullable payload.
                let (op, av) = if e.ty == Ty::I64 {
                    (NumOp1::Iabs, self.masked(l, Ty::I64))
                } else {
                    (NumOp1::Fabs, l.val)
                };
                let dst = self.fresh();
                self.inst(Inst::Num1 { op, dst, a: av });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::Like { ci, a, p, esc } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let lp = self.emit(p, live)?;
                live.push((lp, Ty::Str));
                let le = match esc {
                    Some(e) => Some(self.emit(e, live)?),
                    None => None,
                };
                let (lp, _) = {
                    let x = live.pop().expect("pushed above");
                    let _ = &x;
                    x
                };
                let (la, _) = live.pop().expect("pushed above");
                // Trapping op: mask every nullable payload to "" — the
                // empty string/pattern/escape are all outside the trap
                // domain (like("", "", no-escape) is a clean false/true).
                let empty = self.const_lit(Lit::Str(String::new()));
                let mask = |fbb: &mut Self, l: Lane| match l.flag {
                    None => l.val,
                    Some(f) => fbb.select_of(f, l.val, empty),
                };
                let av = mask(self, la);
                let pv = mask(self, lp);
                let (ev, eflag) = match le {
                    Some(l) => (Some(mask(self, l)), l.flag),
                    None => (None, None),
                };
                let dst = self.fresh();
                self.inst(Inst::Slike {
                    ci: *ci,
                    dst,
                    a: av,
                    p: pv,
                    esc: ev,
                });
                let f1 = self.combine_flags(la.flag, lp.flag);
                Ok(Lane {
                    flag: self.combine_flags(f1, eflag),
                    val: dst,
                })
            }
            SKind::Round2 { trunc, a, n } => {
                let la = self.emit(a, live)?;
                live.push((la, a.ty));
                let ln = self.emit(n, live)?;
                let (la, _) = live.pop().expect("pushed above");
                let dst = self.fresh();
                if a.ty == Ty::F64 {
                    self.inst(Inst::Round2f {
                        trunc: *trunc,
                        dst,
                        a: la.val,
                        n: ln.val,
                    });
                } else {
                    self.inst(Inst::Round2i {
                        trunc: *trunc,
                        dst,
                        a: la.val,
                        n: ln.val,
                    });
                }
                Ok(Lane {
                    flag: self.combine_flags(la.flag, ln.flag),
                    val: dst,
                })
            }
            SKind::MathF1 { op, a } => {
                let l = self.emit(a, live)?;
                // Safe mask per op: the value must be OUTSIDE the trap
                // domain (1.0 for logs; 0.0 is fine for sqrt/trig; total
                // ops need no mask at all).
                let av = match op {
                    NumOp1::Ln | NumOp1::Log2 | NumOp1::Log10 => self.masked_to(l, Lit::F64(1.0)),
                    NumOp1::Fsqrt | NumOp1::Fsin | NumOp1::Fcos | NumOp1::Ftan => {
                        self.masked_to(l, Lit::F64(0.0))
                    }
                    _ => l.val,
                };
                let dst = self.fresh();
                self.inst(Inst::Num1 {
                    op: *op,
                    dst,
                    a: av,
                });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::MathF2 { op, a, b } => {
                let la = self.emit(a, live)?;
                live.push((la, a.ty));
                let lb = self.emit(b, live)?;
                let (la, _) = live.pop().expect("pushed above");
                // Flogb traps on base<=0 / base==1 / x<=0. NULL pre-empts
                // EVERY domain check (pinned: log(-2.0, NULL) is NULL, not
                // an error) — so both payloads mask under the COMBINED
                // flag: either side NULL makes both sides safe.
                let flag = self.combine_flags(la.flag, lb.flag);
                let (av, bv) = match op {
                    BinOp::Flogb => {
                        let a = Lane { flag, val: la.val };
                        let b = Lane { flag, val: lb.val };
                        (
                            self.masked_to(a, Lit::F64(10.0)),
                            self.masked_to(b, Lit::F64(1.0)),
                        )
                    }
                    _ => (la.val, lb.val),
                };
                let val = self.bin(*op, av, bv);
                Ok(Lane { flag, val })
            }
            SKind::Round(a) => {
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                self.inst(Inst::Num1 {
                    op: NumOp1::Fround,
                    dst,
                    a: l.val,
                });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::Str2 { op, a, b } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let lb = self.emit(b, live)?;
                let (la, _) = live.pop().expect("pushed above");
                let flag = self.combine_flags(la.flag, lb.flag);
                // Jaccard/Hamming trap on empty inputs / length mismatch —
                // and the "" mask default IS in the trap domain. NULL
                // pre-empts the checks (jaccard(NULL, '') is NULL, not an
                // error), so BOTH payloads mask to "a" under the COMBINED
                // flag, the Flogb pattern.
                let (av, bv) = match op {
                    StrOp2::Jaccard | StrOp2::Hamming => {
                        let a = Lane { flag, val: la.val };
                        let b = Lane { flag, val: lb.val };
                        (
                            self.masked_to(a, Lit::Str("a".into())),
                            self.masked_to(b, Lit::Str("a".into())),
                        )
                    }
                    _ => (la.val, lb.val),
                };
                let dst = self.fresh();
                self.inst(Inst::Str2 {
                    op: *op,
                    dst,
                    a: av,
                    b: bv,
                });
                Ok(Lane { flag, val: dst })
            }
            SKind::Str3 { op, a, b, c } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let lb = self.emit(b, live)?;
                live.push((lb, Ty::Str));
                let lc = self.emit(c, live)?;
                let (lb, _) = live.pop().expect("pushed above");
                let (la, _) = live.pop().expect("pushed above");
                let dst = self.fresh();
                self.inst(Inst::Str3 {
                    op: *op,
                    dst,
                    a: la.val,
                    b: lb.val,
                    c: lc.val,
                });
                let f = self.combine_flags(la.flag, lb.flag);
                Ok(Lane {
                    flag: self.combine_flags(f, lc.flag),
                    val: dst,
                })
            }
            SKind::Str2i { op, a, n } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let ln = self.emit(n, live)?;
                let (la, _) = live.pop().expect("pushed above");
                // Total under masked defaults: repeat("", any n) allocates
                // nothing (the cap guard sees 0 bytes) and extract is total.
                let dst = self.fresh();
                self.inst(Inst::Str2i {
                    op: *op,
                    dst,
                    a: la.val,
                    n: ln.val,
                });
                Ok(Lane {
                    flag: self.combine_flags(la.flag, ln.flag),
                    val: dst,
                })
            }
            SKind::Spad { left, a, len, pad } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let ll = self.emit(len, live)?;
                live.push((ll, Ty::I64));
                let lp = self.emit(pad, live)?;
                let (ll, _) = live.pop().expect("pushed above");
                let (la, _) = live.pop().expect("pushed above");
                // The empty-pad trap fires only when growth is needed; a
                // VALID len with a NULL pad must not trap (NULL pre-empts).
                // Mask len to 0 under the COMBINED flag — len 0 returns ''
                // before the pad is ever examined.
                let flag = {
                    let f = self.combine_flags(la.flag, ll.flag);
                    self.combine_flags(f, lp.flag)
                };
                let len_v = self.masked_to(Lane { flag, val: ll.val }, Lit::I64(0));
                let dst = self.fresh();
                self.inst(Inst::Spad {
                    left: *left,
                    dst,
                    a: la.val,
                    len: len_v,
                    pad: lp.val,
                });
                Ok(Lane { flag, val: dst })
            }
            SKind::Sslice { a, lo, hi } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let llo = self.emit(lo, live)?;
                live.push((llo, Ty::I64));
                let lhi = self.emit(hi, live)?;
                let (llo, _) = live.pop().expect("pushed above");
                let (la, _) = live.pop().expect("pushed above");
                let dst = self.fresh();
                self.inst(Inst::Sslice {
                    dst,
                    a: la.val,
                    lo: llo.val,
                    hi: lhi.val,
                });
                let f = self.combine_flags(la.flag, llo.flag);
                Ok(Lane {
                    flag: self.combine_flags(f, lhi.flag),
                    val: dst,
                })
            }
            SKind::Sord { empty_zero, a } => {
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                self.inst(Inst::Sord {
                    empty_zero: *empty_zero,
                    dst,
                    a: l.val,
                });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::StripAccents(a) => {
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                self.inst(Inst::Str1 {
                    op: StrOp1::StripAccents,
                    dst,
                    a: l.val,
                });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::SLen { bytes, a } => {
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                self.inst(Inst::SLen {
                    bytes: *bytes,
                    dst,
                    a: l.val,
                });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::Concat { a, b } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let lb = self.emit(b, live)?;
                let (la, _) = live.pop().expect("pushed above");
                let dst = self.fresh();
                self.inst(Inst::Sconcat {
                    dst,
                    a: la.val,
                    b: lb.val,
                });
                Ok(Lane {
                    flag: self.combine_flags(la.flag, lb.flag),
                    val: dst,
                })
            }
        }
    }

    /// Emit (or reuse) join `j`'s probe in the current block. The probe is
    /// pure — same keys, same row, same result — so blocks re-probe rather
    /// than thread probe lanes through branch args. Returns the valid-hit
    /// flag (map hit AND every nullable key's validity: a NULL key never
    /// matches, and a garbage payload under a false flag must not spuriously
    /// hit) plus the value-column registers.
    fn emit_probe(&mut self, j: u32, live: &mut Live) -> Result<(Value, Vec<Value>), PrepareError> {
        if let Some((valid_hit, dsts)) = self.blocks[self.cur].probes.get(&j) {
            return Ok((*valid_hit, dsts.clone()));
        }
        let spec = &self.joins[j as usize];
        let nkeys = spec.keys.len();
        for key in &spec.keys {
            let lane = self.emit(key, live)?;
            live.push((lane, key.ty));
        }
        let mut keys_valid: Option<Value> = None;
        let mut key_vals = Vec::with_capacity(nkeys);
        for (lane, _) in &live[live.len() - nkeys..] {
            key_vals.push(lane.val);
            if let Some(f) = lane.flag {
                keys_valid = self.combine_flags(keys_valid, Some(f));
            }
        }
        live.truncate(live.len() - nkeys);

        let spec = &self.joins[j as usize];
        let hit = self.fresh();
        let dsts: Vec<Value> = spec.val_cols.iter().map(|_| self.b.fresh()).collect();
        self.inst(Inst::Probe {
            static_id: j,
            hit,
            dsts: dsts.clone(),
            keys: key_vals,
        });
        let valid_hit = match keys_valid {
            None => hit,
            Some(f) => self.bin(BinOp::And, f, hit),
        };
        self.blocks[self.cur]
            .probes
            .insert(j, (valid_hit, dsts.clone()));
        Ok((valid_hit, dsts))
    }

    /// Branchless Kleene AND/OR from flag algebra. With the lane contract
    /// (payloads under a false flag may be garbage), every value read is
    /// guarded by its flag.
    fn kleene(
        &mut self,
        a: &SExpr,
        b: &SExpr,
        live: &mut Live,
        is_and: bool,
    ) -> Result<Lane, PrepareError> {
        let la = self.emit(a, live)?;
        live.push((la, Ty::I1));
        let lb = self.emit(b, live)?;
        let (la, _) = live.pop().expect("pushed above");

        let op = if is_and { BinOp::And } else { BinOp::Or };
        let val = self.bin(op, la.val, lb.val);
        let flag = match (la.flag, lb.flag) {
            (None, None) => None,
            // One side definite: the result is known when the definite side
            // decides (false for AND, true for OR), or when the other side
            // is valid.
            (Some(f), None) => {
                let decides = if is_and { self.not(lb.val) } else { lb.val };
                Some(self.bin(BinOp::Or, f, decides))
            }
            (None, Some(f)) => {
                let decides = if is_and { self.not(la.val) } else { la.val };
                Some(self.bin(BinOp::Or, f, decides))
            }
            (Some(fa), Some(fb)) => {
                // Known when: both valid, or either side validly decides.
                let both = self.bin(BinOp::And, fa, fb);
                let da = if is_and { self.not(la.val) } else { la.val };
                let da = self.bin(BinOp::And, fa, da);
                let db = if is_and { self.not(lb.val) } else { lb.val };
                let db = self.bin(BinOp::And, fb, db);
                let t = self.bin(BinOp::Or, both, da);
                Some(self.bin(BinOp::Or, t, db))
            }
        };
        Ok(Lane { flag, val })
    }

    /// CASE chain: one condition block per arm, results evaluated only on
    /// their taken path, all paths joining with the result lane as params.
    fn case(
        &mut self,
        e: &SExpr,
        arms: &[(SExpr, SExpr)],
        default: Option<&SExpr>,
        live: &mut Live,
    ) -> Result<Lane, PrepareError> {
        let res_ty = e.ty;
        let res_nullable = e.nullable;

        let mut join_tys = Self::live_types(live);
        if res_nullable {
            join_tys.push(Ty::I1);
        }
        join_tys.push(res_ty);
        let live_width = Self::live_types(live).len();
        let (join, join_params) = self.create_block(&join_tys);

        let finish_branch = |fb: &mut FB<'a>, lane: Lane, live: &Live| -> Term {
            let mut args = Self::live_args(live);
            if res_nullable {
                let flag = match lane.flag {
                    Some(f) => f,
                    None => fb.const_i1(true),
                };
                args.push(flag);
            }
            args.push(lane.val);
            Term::Jump {
                to: BlockId(join as u32),
                args,
            }
        };

        for (cond, result) in arms {
            let cl = self.emit(cond, live)?;
            let keep = self.truthy(cl);
            let shape = Self::live_types(live);
            let (then_b, then_p) = self.create_block(&shape);
            let (else_b, else_p) = self.create_block(&shape);
            let args = Self::live_args(live);
            self.term(Term::Brif {
                cond: keep,
                then_to: BlockId(then_b as u32),
                then_args: args.clone(),
                else_to: BlockId(else_b as u32),
                else_args: args,
            });

            self.switch(then_b);
            Self::rebind_live(live, &then_p);
            let rl = self.emit(result, live)?;
            let jump = finish_branch(self, rl, live);
            self.term(jump);

            self.switch(else_b);
            Self::rebind_live(live, &else_p);
        }

        let dl = match default {
            Some(d) => self.emit(d, live)?,
            None => {
                let flag = self.const_i1(false);
                let val = self.default_of(res_ty);
                Lane {
                    flag: Some(flag),
                    val,
                }
            }
        };
        let jump = finish_branch(self, dl, live);
        self.term(jump);

        self.switch(join);
        Self::rebind_live(live, &join_params[..live_width]);
        let tail = &join_params[live_width..];
        Ok(if res_nullable {
            Lane {
                flag: Some(tail[0]),
                val: tail[1],
            }
        } else {
            Lane {
                flag: None,
                val: tail[0],
            }
        })
    }

    /// CAST/TRY_CAST lowering. NULL input never traps; CAST failure traps
    /// with a conversion message; TRY_CAST failure becomes NULL.
    fn cast(
        &mut self,
        e: &SExpr,
        inner: &SExpr,
        trying: bool,
        live: &mut Live,
    ) -> Result<Lane, PrepareError> {
        let from = inner.ty;
        let to = e.ty;
        let l = self.emit(inner, live)?;

        // Branchless conversions first.
        let simple: Option<Lane> = match (from, to) {
            (a, b) if a == b => Some(l),
            (Ty::I64, Ty::F64) => {
                let dst = self.fresh();
                self.inst(Inst::Itof { dst, a: l.val });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::F64, Ty::I64) if !trying => {
                // ftoi.round matches DuckDB CAST rounding; its own range trap
                // stands in for DuckDB's conversion error. Nullable payloads
                // are masked: computed garbage under a false flag (x * 1e300
                // * 1e300 with x NULL) must not fire the range trap.
                let a = self.masked(l, Ty::F64);
                let dst = self.fresh();
                self.inst(Inst::Ftoi {
                    mode: super::ir::RoundMode::Round,
                    dst,
                    a,
                });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::I64, Ty::Str) => {
                let dst = self.fresh();
                self.inst(Inst::Itos { dst, a: l.val });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::F64, Ty::Str) => {
                let dst = self.fresh();
                self.inst(Inst::Ftos { dst, a: l.val });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::I1, Ty::Str) => {
                let t = self.const_lit(Lit::Str("true".to_string()));
                let f = self.const_lit(Lit::Str("false".to_string()));
                let dst = self.fresh();
                self.inst(Inst::Select {
                    dst,
                    cond: l.val,
                    a: t,
                    b: f,
                });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::I1, Ty::I64) => {
                let one = self.const_lit(Lit::I64(1));
                let zero = self.const_lit(Lit::I64(0));
                let dst = self.fresh();
                self.inst(Inst::Select {
                    dst,
                    cond: l.val,
                    a: one,
                    b: zero,
                });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::I1, Ty::F64) => {
                let one = self.const_lit(Lit::F64(1.0));
                let zero = self.const_lit(Lit::F64(0.0));
                let dst = self.fresh();
                self.inst(Inst::Select {
                    dst,
                    cond: l.val,
                    a: one,
                    b: zero,
                });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::I64, Ty::I1) => {
                let zero = self.const_lit(Lit::I64(0));
                let dst = self.fresh();
                self.inst(Inst::Cmp {
                    pred: CmpPred::Ne,
                    ty: Ty::I64,
                    dst,
                    a: l.val,
                    b: zero,
                });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::F64, Ty::I1) => {
                // DuckDB: nonzero -> true (measured 2.5::BOOLEAN = true).
                let zero = self.const_lit(Lit::F64(0.0));
                let dst = self.fresh();
                self.inst(Inst::Cmp {
                    pred: CmpPred::Ne,
                    ty: Ty::F64,
                    dst,
                    a: l.val,
                    b: zero,
                });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            _ => None,
        };
        if let Some(lane) = simple {
            // TRY_CAST of an infallible conversion: the declared nullability
            // is `true`, so surface a flag even though nothing can fail.
            if trying && lane.flag.is_none() {
                let t = self.const_i1(true);
                return Ok(Lane {
                    flag: Some(t),
                    val: lane.val,
                });
            }
            return Ok(lane);
        }

        match (from, to) {
            // String parses: shared shape, differing parse op.
            (Ty::Str, Ty::I64) | (Ty::Str, Ty::F64) => {
                let ok = self.fresh();
                let parsed = self.fresh();
                if to == Ty::I64 {
                    self.inst(Inst::StoiOpt {
                        flag: ok,
                        dst: parsed,
                        a: l.val,
                    });
                } else {
                    self.inst(Inst::StofOpt {
                        flag: ok,
                        dst: parsed,
                        a: l.val,
                    });
                }
                if trying {
                    let flag = match l.flag {
                        Some(f) => self.bin(BinOp::And, f, ok),
                        None => ok,
                    };
                    return Ok(Lane {
                        flag: Some(flag),
                        val: parsed,
                    });
                }
                // CAST: trap iff the input is a real (non-NULL) string that
                // does not parse.
                let not_ok = self.not(ok);
                let bad = match l.flag {
                    Some(f) => self.bin(BinOp::And, f, not_ok),
                    None => not_ok,
                };
                let (trap_b, _) = self.create_block(&[]);
                let mut cont_tys = Self::live_types(live);
                let flag_in_shape = l.flag.is_some();
                if flag_in_shape {
                    cont_tys.push(Ty::I1);
                }
                cont_tys.push(to);
                let live_width = Self::live_types(live).len();
                let (cont_b, cont_p) = self.create_block(&cont_tys);
                let mut args = Self::live_args(live);
                if let Some(f) = l.flag {
                    args.push(f);
                }
                args.push(parsed);
                self.term(Term::Brif {
                    cond: bad,
                    then_to: BlockId(trap_b as u32),
                    then_args: vec![],
                    else_to: BlockId(cont_b as u32),
                    else_args: args,
                });
                self.switch(trap_b);
                self.term(Term::Trap {
                    msg: format!(
                        "Conversion Error: could not cast VARCHAR to {}",
                        if to == Ty::I64 { "BIGINT" } else { "DOUBLE" }
                    ),
                });
                self.switch(cont_b);
                Self::rebind_live(live, &cont_p[..live_width]);
                let tail = &cont_p[live_width..];
                Ok(if flag_in_shape {
                    Lane {
                        flag: Some(tail[0]),
                        val: tail[1],
                    }
                } else {
                    Lane {
                        flag: None,
                        val: tail[0],
                    }
                })
            }
            // TRY_CAST(f64 -> i64): guard the range so ftoi cannot trap; the
            // payload rides the branch on the live stack.
            (Ty::F64, Ty::I64) => {
                debug_assert!(trying);
                // Exactly ±2^63; NaN fails both compares -> NULL. A NULL
                // input's default payload passes the range check but its
                // false flag forces the null path anyway.
                let min = self.const_lit(Lit::F64(-9223372036854775808.0));
                let max = self.const_lit(Lit::F64(9223372036854775808.0));
                let ge = self.fresh();
                self.inst(Inst::Cmp {
                    pred: CmpPred::Ge,
                    ty: Ty::F64,
                    dst: ge,
                    a: l.val,
                    b: min,
                });
                let lt = self.fresh();
                self.inst(Inst::Cmp {
                    pred: CmpPred::Lt,
                    ty: Ty::F64,
                    dst: lt,
                    a: l.val,
                    b: max,
                });
                let in_range = self.bin(BinOp::And, ge, lt);
                let ok = match l.flag {
                    Some(f) => self.bin(BinOp::And, f, in_range),
                    None => in_range,
                };

                live.push((l, Ty::F64));
                let live_width = Self::live_types(live).len();
                let mut join_tys = Self::live_types(live);
                join_tys.push(Ty::I1);
                join_tys.push(Ty::I64);
                let (join, join_p) = self.create_block(&join_tys);
                let shape = Self::live_types(live);
                let (conv_b, conv_p) = self.create_block(&shape);
                let (null_b, null_p) = self.create_block(&shape);
                let args = Self::live_args(live);
                self.term(Term::Brif {
                    cond: ok,
                    then_to: BlockId(conv_b as u32),
                    then_args: args.clone(),
                    else_to: BlockId(null_b as u32),
                    else_args: args,
                });

                self.switch(conv_b);
                Self::rebind_live(live, &conv_p);
                let payload = live.last().expect("pushed above").0.val;
                let r = self.fresh();
                self.inst(Inst::Ftoi {
                    mode: super::ir::RoundMode::Round,
                    dst: r,
                    a: payload,
                });
                let t = self.const_i1(true);
                let mut args = Self::live_args(live);
                args.push(t);
                args.push(r);
                self.term(Term::Jump {
                    to: BlockId(join as u32),
                    args,
                });

                self.switch(null_b);
                Self::rebind_live(live, &null_p);
                let f = self.const_i1(false);
                let d = self.const_lit(Lit::I64(0));
                let mut args = Self::live_args(live);
                args.push(f);
                args.push(d);
                self.term(Term::Jump {
                    to: BlockId(join as u32),
                    args,
                });

                self.switch(join);
                Self::rebind_live(live, &join_p[..live_width]);
                live.pop();
                let tail = &join_p[live_width..];
                Ok(Lane {
                    flag: Some(tail[0]),
                    val: tail[1],
                })
            }
            (from, to) => Err(PrepareError::Internal(format!(
                "cast {} -> {} escaped the frontend",
                from.name(),
                to.name()
            ))),
        }
    }

    /// NULL propagation: the result is valid iff every nullable input is.
    fn combine_flags(&mut self, a: Option<Value>, b: Option<Value>) -> Option<Value> {
        match (a, b) {
            (None, None) => None,
            (Some(f), None) | (None, Some(f)) => Some(f),
            (Some(fa), Some(fb)) => Some(self.bin(BinOp::And, fa, fb)),
        }
    }
}
