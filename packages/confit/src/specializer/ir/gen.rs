//! Deterministic random-program generator for round-trip and (later)
//! differential fuzzing. Every generated program is verifier-valid by
//! construction and built with dense definition-ordered value ids, so
//! `parse(print(p)) == p` must hold exactly — any divergence is a bug in the
//! printer, the parser, or this generator, and all three are worth knowing.
//!
//! Hand-rolled xorshift instead of a proptest dependency: round-trip failures
//! shrink trivially (the seed pins the program), and the generator doubles as
//! the input source for M-interp's interpreter-vs-codegen fuzzing.

use super::{
    BinOp, Block, BlockId, Builder, CmpPred, Col, ColTy, Inst, Lit, NumOp1, Program, RoundMode,
    StaticTy, StrOp1, StrOp2, StrOp2i, StrOp3, Term, TrimSide, Ty, Value,
};

/// Seed count for a fuzz loop: `default` in the gate, overridden by
/// `SPECIALIZER_FUZZ_SEEDS` for exploratory deep runs. Mirrors the regexp
/// fuzzer's `REGEXP_FUZZ_N` knob so a deep audit never needs a source edit.
/// Each seed keys a distinct xorshift stream, so raising this raises coverage
/// rather than repeating programs.
pub fn fuzz_seeds(default: u64) -> u64 {
    match std::env::var("SPECIALIZER_FUZZ_SEEDS") {
        Ok(s) => s.trim().parse().unwrap_or(default),
        Err(_) => default,
    }
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // xorshift64* must not be seeded with 0.
        Rng(seed.wrapping_mul(2685821657736338717).max(1))
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(2685821657736338717)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    pub fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

const TYS: [Ty; 4] = [Ty::I1, Ty::I64, Ty::F64, Ty::Str];

/// Column-name pool: mixes plain identifiers with names that force the
/// quoted form (SQL-derived output names contain arbitrary characters).
const NASTY_NAMES: [&str; 4] = ["COALESCE(a, b)", "weird name", "a\"quote", "tab\there"];

fn rand_ty(rng: &mut Rng) -> Ty {
    TYS[rng.below(4) as usize]
}

fn rand_col_ty(rng: &mut Rng) -> ColTy {
    ColTy {
        ty: rand_ty(rng),
        nullable: rng.chance(40),
    }
}

fn rand_lit(rng: &mut Rng, ty: Ty) -> Lit {
    match ty {
        Ty::I1 => Lit::I1(rng.chance(50)),
        Ty::I64 => Lit::I64(match rng.below(6) {
            0 => 0,
            1 => -1,
            2 => i64::MAX,
            3 => i64::MIN,
            _ => rng.next() as i64 % 1_000_000,
        }),
        Ty::F64 => Lit::F64(match rng.below(9) {
            0 => 0.0,
            1 => -0.0,
            2 => f64::NAN,
            3 => f64::INFINITY,
            4 => f64::NEG_INFINITY,
            5 => 1e300,
            6 => 1e-5,
            7 => 0.1,
            _ => (rng.next() as i64 % 1000) as f64 / 8.0,
        }),
        Ty::Str => Lit::Str(
            match rng.below(6) {
                0 => "",
                1 => "plain",
                2 => "quote\"backslash\\",
                3 => "line\nbreak\ttab",
                4 => "unicode é ✓",
                _ => "x",
            }
            .to_string(),
        ),
    }
}

/// Per-block generation state: the values in scope, by type.
struct Scope {
    avail: Vec<(Value, Ty)>,
}

impl Scope {
    fn new() -> Scope {
        Scope { avail: Vec::new() }
    }

    fn add(&mut self, v: Value, ty: Ty) {
        self.avail.push((v, ty));
    }

    fn pick(&self, rng: &mut Rng, ty: Ty) -> Option<Value> {
        let of_ty: Vec<Value> = self
            .avail
            .iter()
            .filter(|(_, t)| *t == ty)
            .map(|(v, _)| *v)
            .collect();
        if of_ty.is_empty() {
            None
        } else {
            Some(of_ty[rng.below(of_ty.len() as u64) as usize])
        }
    }
}

/// Get a value of `ty`, minting a const when none is in scope (or sometimes
/// just because — consts are where the literal edge cases enter).
fn ensure(
    rng: &mut Rng,
    b: &mut Builder,
    scope: &mut Scope,
    insts: &mut Vec<Inst>,
    ty: Ty,
) -> Value {
    if rng.chance(70) {
        if let Some(v) = scope.pick(rng, ty) {
            return v;
        }
    }
    let dst = b.fresh();
    insts.push(Inst::Const {
        dst,
        lit: rand_lit(rng, ty),
    });
    scope.add(dst, ty);
    dst
}

/// Loads for every input column plus reads of every static — the block's
/// raw material.
fn load_all(
    rng: &mut Rng,
    b: &mut Builder,
    scope: &mut Scope,
    insts: &mut Vec<Inst>,
    p_in: &[Col],
    statics: &[StaticTy],
) {
    for (ci, c) in p_in.iter().enumerate() {
        if c.ty.nullable {
            let flag = b.fresh();
            let dst = b.fresh();
            insts.push(Inst::LoadOpt {
                flag,
                dst,
                col: ci as u32,
            });
            scope.add(flag, Ty::I1);
            scope.add(dst, c.ty.ty);
        } else {
            let dst = b.fresh();
            insts.push(Inst::Load {
                dst,
                col: ci as u32,
            });
            scope.add(dst, c.ty.ty);
        }
    }
    for (si, st) in statics.iter().enumerate() {
        match st {
            StaticTy::Scalar(ct) if ct.nullable => {
                let flag = b.fresh();
                let dst = b.fresh();
                insts.push(Inst::SloadOpt {
                    static_id: si as u32,
                    flag,
                    dst,
                });
                scope.add(flag, Ty::I1);
                scope.add(dst, ct.ty);
            }
            StaticTy::Scalar(ct) => {
                let dst = b.fresh();
                insts.push(Inst::Sload {
                    static_id: si as u32,
                    dst,
                });
                scope.add(dst, ct.ty);
            }
            StaticTy::Map { keys, values } => {
                let key_vals: Vec<Value> = keys
                    .iter()
                    .map(|kt| ensure(rng, b, scope, insts, *kt))
                    .collect();
                let hit = b.fresh();
                scope.add(hit, Ty::I1);
                let mut dsts = Vec::with_capacity(values.len());
                for vt in values {
                    let d = b.fresh();
                    scope.add(d, *vt);
                    dsts.push(d);
                }
                insts.push(Inst::Probe {
                    static_id: si as u32,
                    hit,
                    dsts,
                    keys: key_vals,
                });
            }
            StaticTy::Model { n_features } => {
                // Model id 0 always exists (the data generator builds at
                // least one model per declaration), so the differential
                // exercises the traversal rather than degenerating into
                // "both backends trap". The out-of-range id is a pin.
                let id = b.fresh();
                insts.push(Inst::Const {
                    dst: id,
                    lit: Lit::I64(0),
                });
                scope.add(id, Ty::I64);
                let feats: Vec<Value> = (0..*n_features)
                    .map(|_| ensure(rng, b, scope, insts, Ty::F64))
                    .collect();
                let dst = b.fresh();
                scope.add(dst, Ty::F64);
                insts.push(Inst::Predict {
                    static_id: si as u32,
                    dst,
                    id,
                    feats,
                });
            }
            // The random-program generator doesn't emit multiplicity loops
            // (stage-B programs are exercised by targeted tests instead).
            StaticTy::MultiMap { .. } | StaticTy::BatchMap { .. } => {
            }
        }
    }
}

/// A small fresh i64 const in [-9, 9] — positions/counts for string ops,
/// keeping generated programs executable (range guards, alloc caps).
fn small(rng: &mut Rng, b: &mut Builder, insts: &mut Vec<Inst>) -> Value {
    let dst = b.fresh();
    insts.push(Inst::Const {
        dst,
        lit: Lit::I64(rng.below(19) as i64 - 9),
    });
    dst
}

/// A few random compute instructions over whatever is in scope.
fn compute(rng: &mut Rng, b: &mut Builder, scope: &mut Scope, insts: &mut Vec<Inst>) {
    let n = rng.below(7);
    for _ in 0..n {
        match rng.below(14) {
            0 => {
                let ops = [
                    BinOp::Iadd,
                    BinOp::Isub,
                    BinOp::Imul,
                    BinOp::Fadd,
                    BinOp::Fsub,
                    BinOp::Fmul,
                    BinOp::Fdiv,
                    BinOp::Frem,
                    BinOp::Fpow,
                    BinOp::Ffloordiv,
                    BinOp::Ffloormod,
                    BinOp::Fnextafter,
                    BinOp::And,
                    BinOp::Or,
                    BinOp::Xor,
                ];
                // idiv/irem excluded: they trap on zero, and generated
                // programs must stay executable for M-interp fuzzing.
                // Flogb joins at low weight: it traps on half the domain,
                // but identical-trap agreement is differential signal too.
                let op = if rng.chance(5) {
                    BinOp::Flogb
                } else {
                    ops[rng.below(ops.len() as u64) as usize]
                };
                let (ot, rt) = op.sig();
                let a = ensure(rng, b, scope, insts, ot);
                let rhs = ensure(rng, b, scope, insts, ot);
                let dst = b.fresh();
                insts.push(Inst::Bin { op, dst, a, b: rhs });
                scope.add(dst, rt);
            }
            1 => {
                let ty = [Ty::I64, Ty::F64, Ty::Str][rng.below(3) as usize];
                let preds = [
                    CmpPred::Eq,
                    CmpPred::Ne,
                    CmpPred::Lt,
                    CmpPred::Le,
                    CmpPred::Gt,
                    CmpPred::Ge,
                ];
                let pred = preds[rng.below(6) as usize];
                let a = ensure(rng, b, scope, insts, ty);
                let rhs = ensure(rng, b, scope, insts, ty);
                let dst = b.fresh();
                insts.push(Inst::Cmp {
                    pred,
                    ty,
                    dst,
                    a,
                    b: rhs,
                });
                scope.add(dst, Ty::I1);
            }
            2 => {
                let a = ensure(rng, b, scope, insts, Ty::I1);
                let dst = b.fresh();
                insts.push(Inst::Not { dst, a });
                scope.add(dst, Ty::I1);
            }
            3 => {
                let ty = rand_ty(rng);
                let cond = ensure(rng, b, scope, insts, Ty::I1);
                let a = ensure(rng, b, scope, insts, ty);
                let rhs = ensure(rng, b, scope, insts, ty);
                let dst = b.fresh();
                insts.push(Inst::Select {
                    dst,
                    cond,
                    a,
                    b: rhs,
                });
                scope.add(dst, ty);
            }
            4 => {
                let a = ensure(rng, b, scope, insts, Ty::I64);
                let dst = b.fresh();
                insts.push(Inst::Itof { dst, a });
                scope.add(dst, Ty::F64);
            }
            5 => {
                let mode = if rng.chance(50) {
                    RoundMode::Trunc
                } else {
                    RoundMode::Round
                };
                let a = ensure(rng, b, scope, insts, Ty::F64);
                let dst = b.fresh();
                insts.push(Inst::Ftoi { mode, dst, a });
                scope.add(dst, Ty::I64);
            }
            6 => {
                let a = ensure(rng, b, scope, insts, Ty::Str);
                let rhs = ensure(rng, b, scope, insts, Ty::Str);
                let dst = b.fresh();
                insts.push(Inst::Sconcat { dst, a, b: rhs });
                scope.add(dst, Ty::Str);
            }
            8 => {
                let op = match rng.below(3) {
                    0 => StrOp1::Upper,
                    1 => StrOp1::Lower,
                    _ => StrOp1::StripAccents,
                };
                let a = ensure(rng, b, scope, insts, Ty::Str);
                let dst = b.fresh();
                insts.push(Inst::Str1 { op, dst, a });
                scope.add(dst, Ty::Str);
            }
            9 => {
                let sides = [TrimSide::Both, TrimSide::Lead, TrimSide::Trail];
                let side = sides[rng.below(3) as usize];
                let a = ensure(rng, b, scope, insts, Ty::Str);
                let chars = ensure(rng, b, scope, insts, Ty::Str);
                let dst = b.fresh();
                insts.push(Inst::Strim {
                    side,
                    dst,
                    a,
                    chars,
                });
                scope.add(dst, Ty::Str);
            }
            10 => {
                let a = ensure(rng, b, scope, insts, Ty::Str);
                // Positions are small fresh consts, not arbitrary scope
                // values: the ±2^32 range guard traps, and generated
                // programs must stay executable for M-interp fuzzing.
                let start = small(rng, b, insts);
                scope.add(start, Ty::I64);
                let len = if rng.chance(50) {
                    let l = small(rng, b, insts);
                    scope.add(l, Ty::I64);
                    Some(l)
                } else {
                    None
                };
                let dst = b.fresh();
                insts.push(Inst::Ssubstr { dst, a, start, len });
                scope.add(dst, Ty::Str);
            }
            11 => {
                // iabs excluded: it traps on i64::MIN, and generated programs
                // must stay executable for M-interp fuzzing. The wave-1
                // trapping unaries join at low weight (trap-agreement is
                // signal, but programs should mostly run to completion).
                let total = [
                    NumOp1::Fabs,
                    NumOp1::Fround,
                    NumOp1::Ffloor,
                    NumOp1::Fceil,
                    NumOp1::Ftrunc,
                    NumOp1::Fexp,
                    NumOp1::Fcbrt,
                ];
                let trapping = [
                    NumOp1::Ln,
                    NumOp1::Log2,
                    NumOp1::Log10,
                    NumOp1::Fsqrt,
                    NumOp1::Fsin,
                    NumOp1::Fcos,
                    NumOp1::Ftan,
                ];
                let op = if rng.chance(20) {
                    trapping[rng.below(trapping.len() as u64) as usize]
                } else {
                    total[rng.below(total.len() as u64) as usize]
                };
                let a = ensure(rng, b, scope, insts, Ty::F64);
                let dst = b.fresh();
                insts.push(Inst::Num1 { op, dst, a });
                scope.add(dst, Ty::F64);
            }
            12 if rng.chance(35) => {
                // round/trunc with digits: total on both int and float.
                let trunc = rng.chance(50);
                let n = ensure(rng, b, scope, insts, Ty::I64);
                if rng.chance(50) {
                    let a = ensure(rng, b, scope, insts, Ty::F64);
                    let dst = b.fresh();
                    insts.push(Inst::Round2f { trunc, dst, a, n });
                    scope.add(dst, Ty::F64);
                } else {
                    let a = ensure(rng, b, scope, insts, Ty::I64);
                    let dst = b.fresh();
                    insts.push(Inst::Round2i { trunc, dst, a, n });
                    scope.add(dst, Ty::I64);
                }
            }
            12 if rng.chance(25) => {
                // LIKE without ESCAPE is total (no trap conditions).
                let a = ensure(rng, b, scope, insts, Ty::Str);
                let pat = ensure(rng, b, scope, insts, Ty::Str);
                let dst = b.fresh();
                insts.push(Inst::Slike {
                    ci: rng.chance(30),
                    dst,
                    a,
                    p: pat,
                    esc: None,
                });
                scope.add(dst, Ty::I1);
            }
            12 => {
                let a = ensure(rng, b, scope, insts, Ty::Str);
                if rng.chance(30) {
                    let dst = b.fresh();
                    insts.push(Inst::SLen {
                        bytes: rng.chance(50),
                        dst,
                        a,
                    });
                    scope.add(dst, Ty::I64);
                } else {
                    let n = ensure(rng, b, scope, insts, Ty::Str);
                    // Jaccard/Hamming trap (empty inputs / length mismatch)
                    // — low weight, same policy as Flogb: trap-agreement is
                    // differential signal, but programs should mostly run.
                    let op = if rng.chance(15) {
                        [StrOp2::Jaccard, StrOp2::Hamming][rng.below(2) as usize]
                    } else {
                        let ops = [
                            StrOp2::Find,
                            StrOp2::Contains,
                            StrOp2::Starts,
                            StrOp2::Ends,
                            StrOp2::Levenshtein,
                            StrOp2::Damerau,
                        ];
                        ops[rng.below(6) as usize]
                    };
                    let dst = b.fresh();
                    insts.push(Inst::Str2 { op, dst, a, b: n });
                    scope.add(dst, op.result_ty());
                }
            }
            13 => {
                let a = ensure(rng, b, scope, insts, Ty::Str);
                match rng.below(5) {
                    0 => {
                        let op = if rng.chance(50) {
                            StrOp3::Replace
                        } else {
                            StrOp3::Translate
                        };
                        let x = ensure(rng, b, scope, insts, Ty::Str);
                        let y = ensure(rng, b, scope, insts, Ty::Str);
                        let dst = b.fresh();
                        insts.push(Inst::Str3 {
                            op,
                            dst,
                            a,
                            b: x,
                            c: y,
                        });
                        scope.add(dst, Ty::Str);
                    }
                    1 => {
                        // Repeat with a SMALL fresh count: an arbitrary
                        // in-scope i64 would build multi-GiB strings.
                        let op = if rng.chance(50) {
                            StrOp2i::Repeat
                        } else {
                            StrOp2i::Extract
                        };
                        let n = small(rng, b, insts);
                        scope.add(n, Ty::I64);
                        let dst = b.fresh();
                        insts.push(Inst::Str2i { op, dst, a, n });
                        scope.add(dst, Ty::Str);
                    }
                    2 => {
                        // Spad traps on empty pad + growth — low weight via
                        // small len keeps most programs running while the
                        // trap row stays reachable for agreement checks.
                        let len = small(rng, b, insts);
                        scope.add(len, Ty::I64);
                        let pad = ensure(rng, b, scope, insts, Ty::Str);
                        let dst = b.fresh();
                        insts.push(Inst::Spad {
                            left: rng.chance(50),
                            dst,
                            a,
                            len,
                            pad,
                        });
                        scope.add(dst, Ty::Str);
                    }
                    3 => {
                        let lo = small(rng, b, insts);
                        scope.add(lo, Ty::I64);
                        let hi = small(rng, b, insts);
                        scope.add(hi, Ty::I64);
                        let dst = b.fresh();
                        insts.push(Inst::Sslice { dst, a, lo, hi });
                        scope.add(dst, Ty::Str);
                    }
                    _ => {
                        let dst = b.fresh();
                        insts.push(Inst::Sord {
                            empty_zero: rng.chance(50),
                            dst,
                            a,
                        });
                        scope.add(dst, Ty::I64);
                    }
                }
            }
            _ => {
                let (from, mk): (Ty, fn(Value, Value) -> Inst) = if rng.chance(50) {
                    (Ty::I64, |dst, a| Inst::Itos { dst, a })
                } else {
                    (Ty::F64, |dst, a| Inst::Ftos { dst, a })
                };
                let a = ensure(rng, b, scope, insts, from);
                let dst = b.fresh();
                insts.push(mk(dst, a));
                scope.add(dst, Ty::Str);
            }
        }
    }
}

/// One store per out column, then the given terminator's stores contract.
fn stores(
    rng: &mut Rng,
    b: &mut Builder,
    scope: &mut Scope,
    insts: &mut Vec<Inst>,
    out_cols: &[Col],
) {
    for (ci, c) in out_cols.iter().enumerate() {
        let val = ensure(rng, b, scope, insts, c.ty.ty);
        if c.ty.nullable {
            let flag = ensure(rng, b, scope, insts, Ty::I1);
            insts.push(Inst::StoreOpt {
                col: ci as u32,
                flag,
                val,
            });
        } else {
            insts.push(Inst::Store {
                col: ci as u32,
                val,
            });
        }
    }
}

pub fn gen_program(seed: u64) -> Program {
    let mut rng = Rng::new(seed);
    let mut b = Builder::new();

    let statics: Vec<StaticTy> = (0..rng.below(3))
        .map(|_| {
            if rng.chance(40) {
                StaticTy::Scalar(rand_col_ty(&mut rng))
            } else if rng.chance(66) {
                StaticTy::Map {
                    keys: (0..1 + rng.below(2)).map(|_| rand_ty(&mut rng)).collect(),
                    values: (0..1 + rng.below(2)).map(|_| rand_ty(&mut rng)).collect(),
                }
            } else {
                StaticTy::Model {
                    n_features: 1 + rng.below(3) as u32,
                }
            }
        })
        .collect();

    let name_of = |i: usize, prefix: &str, rng: &mut Rng| -> String {
        if rng.chance(20) {
            format!("{} {}", NASTY_NAMES[rng.below(4) as usize], i)
        } else {
            format!("{prefix}{i}")
        }
    };
    let in_cols: Vec<Col> = (0..1 + rng.below(3) as usize)
        .map(|i| Col {
            name: name_of(i, "c", &mut rng),
            ty: rand_col_ty(&mut rng),
        })
        .collect();
    let out_cols: Vec<Col> = (0..1 + rng.below(2) as usize)
        .map(|i| Col {
            name: name_of(i, "o", &mut rng),
            ty: rand_col_ty(&mut rng),
        })
        .collect();

    let shape = rng.below(3);
    let mut blocks = Vec::new();
    match shape {
        // Straight line: load, compute, store, emit.
        0 => {
            let mut scope = Scope::new();
            let mut insts = Vec::new();
            load_all(&mut rng, &mut b, &mut scope, &mut insts, &in_cols, &statics);
            compute(&mut rng, &mut b, &mut scope, &mut insts);
            stores(&mut rng, &mut b, &mut scope, &mut insts, &out_cols);
            blocks.push(Block {
                params: vec![],
                insts,
                term: Term::Emit,
            });
        }
        // Filter: entry branches to a storing block or a skip/trap block.
        1 => {
            let mut scope = Scope::new();
            let mut insts = Vec::new();
            load_all(&mut rng, &mut b, &mut scope, &mut insts, &in_cols, &statics);
            compute(&mut rng, &mut b, &mut scope, &mut insts);
            let cond = ensure(&mut rng, &mut b, &mut scope, &mut insts, Ty::I1);
            blocks.push(Block {
                params: vec![],
                insts,
                term: Term::Brif {
                    cond,
                    then_to: BlockId(1),
                    then_args: vec![],
                    else_to: BlockId(2),
                    else_args: vec![],
                },
            });
            let mut keep_scope = Scope::new();
            let mut keep_insts = Vec::new();
            load_all(
                &mut rng,
                &mut b,
                &mut keep_scope,
                &mut keep_insts,
                &in_cols,
                &statics,
            );
            stores(
                &mut rng,
                &mut b,
                &mut keep_scope,
                &mut keep_insts,
                &out_cols,
            );
            blocks.push(Block {
                params: vec![],
                insts: keep_insts,
                term: Term::Emit,
            });
            let drop_term = if rng.chance(80) {
                Term::Skip
            } else {
                Term::Trap {
                    msg: "generated trap".to_string(),
                }
            };
            blocks.push(Block {
                params: vec![],
                insts: vec![],
                term: drop_term,
            });
        }
        // Diamond: both arms feed a join param that lands in a store.
        _ => {
            let join_ty = rand_ty(&mut rng);
            let mut scope = Scope::new();
            let mut insts = Vec::new();
            load_all(&mut rng, &mut b, &mut scope, &mut insts, &in_cols, &statics);
            let cond = ensure(&mut rng, &mut b, &mut scope, &mut insts, Ty::I1);
            blocks.push(Block {
                params: vec![],
                insts,
                term: Term::Brif {
                    cond,
                    then_to: BlockId(1),
                    then_args: vec![],
                    else_to: BlockId(2),
                    else_args: vec![],
                },
            });
            for _ in 0..2 {
                let mut arm_scope = Scope::new();
                let mut arm_insts = Vec::new();
                let v = ensure(&mut rng, &mut b, &mut arm_scope, &mut arm_insts, join_ty);
                blocks.push(Block {
                    params: vec![],
                    insts: arm_insts,
                    term: Term::Jump {
                        to: BlockId(3),
                        args: vec![v],
                    },
                });
            }
            let param = b.fresh();
            let mut join_scope = Scope::new();
            join_scope.add(param, join_ty);
            let mut join_insts = Vec::new();
            load_all(
                &mut rng,
                &mut b,
                &mut join_scope,
                &mut join_insts,
                &in_cols,
                &statics,
            );
            compute(&mut rng, &mut b, &mut join_scope, &mut join_insts);
            stores(
                &mut rng,
                &mut b,
                &mut join_scope,
                &mut join_insts,
                &out_cols,
            );
            blocks.push(Block {
                params: vec![(param, join_ty)],
                insts: join_insts,
                term: Term::Emit,
            });
        }
    }

    Program {
        statics,
        regexes: Vec::new(),
        externs: Vec::new(),
        name: "fuzzed".to_string(),
        in_cols,
        out_cols,
        blocks,
    }
}
