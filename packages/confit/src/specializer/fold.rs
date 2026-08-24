//! Prepare-time scalar folding: the expression half of binding-time
//! analysis. Every all-constant subtree collapses to `Lit`/`NullOf` before
//! lowering, so a static computation never reaches the per-row IR.
//!
//! The folder is deliberately conservative — it folds ONLY when the result
//! provably equals what the interpreter would compute at run time:
//! * every operand must itself be a constant (`Lit`/`NullOf`); a constant
//!   that merely dominates (`FALSE AND dynamic`) is never folded, because
//!   dropping the dynamic side could drop its trap;
//! * an operation that would trap at run time (integer overflow, `% 0`) is
//!   left unfolded — the trap stays a run-time trap, same timing as before;
//! * f64 arithmetic and comparisons mirror exec/interp.rs exactly (IEEE:
//!   `x/0 = inf`, NaN compares false except `!=`);
//! * CASE and CAST nodes fold their children but never themselves — CAST
//!   can trap and CASE guards branch traps, both wrong to evaluate eagerly.

use super::ir::{CmpPred, Lit, Ty};
use super::plan::{ArithOp, SExpr, SKind};

/// A constant operand: a payload or a typed NULL.
enum K {
    Val(Lit),
    Null,
}

fn as_const(e: &SExpr) -> Option<K> {
    match &e.kind {
        SKind::Lit(l) => Some(K::Val(l.clone())),
        SKind::NullOf => Some(K::Null),
        _ => None,
    }
}

fn lit(l: Lit, ty: Ty) -> SExpr {
    SExpr {
        kind: SKind::Lit(l),
        ty,
        nullable: false,
    }
}

fn null(ty: Ty) -> SExpr {
    SExpr {
        kind: SKind::NullOf,
        ty,
        nullable: true,
    }
}

/// Fold every all-constant subtree of `e`, bottom-up.
pub fn fold(e: SExpr) -> SExpr {
    let SExpr { kind, ty, nullable } = e;
    let e = |kind| SExpr { kind, ty, nullable };
    match kind {
        SKind::Col(_) | SKind::StaticCol { .. } | SKind::Lit(_) | SKind::NullOf
        | SKind::JoinHit(_) => e(kind),
        // Opaque call: fold the args, never the call itself.
        SKind::ExternCall {
            site,
            ext,
            args,
            ret,
            whole,
        } => e(SKind::ExternCall {
            site,
            ext,
            args: args.into_iter().map(fold).collect(),
            ret,
            whole,
        }),
        // Fold the operands, never the scoring — a model is prepare-time
        // data, not a constant the folder can evaluate. Note a NULL feature
        // does NOT collapse the call to NULL: that is the whole point of the
        // missing-value rule.
        SKind::TreePredict { model, id, feats } => e(SKind::TreePredict {
            model,
            id: Box::new(fold(*id)),
            feats: feats.into_iter().map(fold).collect(),
        }),
        SKind::IntToFloat(inner) => {
            let inner = fold(*inner);
            match as_const(&inner) {
                Some(K::Val(Lit::I64(i))) => lit(Lit::F64(i as f64), ty),
                Some(K::Null) => null(ty),
                _ => e(SKind::IntToFloat(Box::new(inner))),
            }
        }
        // Same shape, but the constant folds through f32 — otherwise a
        // LITERAL feature would keep the double rounding this node exists
        // to remove (TASK-77).
        SKind::IntToFloat32(inner) => {
            let inner = fold(*inner);
            match as_const(&inner) {
                Some(K::Val(Lit::I64(i))) => lit(Lit::F64(i as f32 as f64), ty),
                Some(K::Null) => null(ty),
                _ => e(SKind::IntToFloat32(Box::new(inner))),
            }
        }
        // Wave-1 math: fold children only — the ops themselves stay
        // runtime so constant domain errors trap per row exactly like the
        // vectorized path we pin against (no fold/vector divergence).
        SKind::Str2 { op, a, b } => {
            let a = fold(*a);
            let b = fold(*b);
            e(SKind::Str2 {
                op,
                a: Box::new(a),
                b: Box::new(b),
            })
        }
        SKind::SLen { bytes, a } => {
            let a = fold(*a);
            e(SKind::SLen {
                bytes,
                a: Box::new(a),
            })
        }
        SKind::ReMatch { re, a } => e(SKind::ReMatch {
            re,
            a: Box::new(fold(*a)),
        }),
        SKind::ReExtract { re, group, a } => e(SKind::ReExtract {
            re,
            group,
            a: Box::new(fold(*a)),
        }),
        SKind::ReReplace { re, global, a } => e(SKind::ReReplace {
            re,
            global,
            a: Box::new(fold(*a)),
        }),
        SKind::Like { ci, a, p, esc } => {
            let a = fold(*a);
            let p = fold(*p);
            let esc = esc.map(|e| Box::new(fold(*e)));
            e(SKind::Like {
                ci,
                a: Box::new(a),
                p: Box::new(p),
                esc,
            })
        }
        SKind::Round2 { trunc, a, n } => {
            let a = fold(*a);
            let n = fold(*n);
            e(SKind::Round2 {
                trunc,
                a: Box::new(a),
                n: Box::new(n),
            })
        }
        SKind::MathF1 { op, a } => {
            let a = fold(*a);
            e(SKind::MathF1 { op, a: Box::new(a) })
        }
        SKind::MathF2 { op, a, b } => {
            let a = fold(*a);
            let b = fold(*b);
            e(SKind::MathF2 {
                op,
                a: Box::new(a),
                b: Box::new(b),
            })
        }
        // Wave-3 string ops: fold children only, same policy as Str2 —
        // the ops stay runtime so constant trap rows keep their timing.
        SKind::Str3 { op, a, b, c } => {
            let a = fold(*a);
            let b = fold(*b);
            let c = fold(*c);
            e(SKind::Str3 {
                op,
                a: Box::new(a),
                b: Box::new(b),
                c: Box::new(c),
            })
        }
        SKind::Str2i { op, a, n } => {
            let a = fold(*a);
            let n = fold(*n);
            e(SKind::Str2i {
                op,
                a: Box::new(a),
                n: Box::new(n),
            })
        }
        SKind::Spad { left, a, len, pad } => {
            let a = fold(*a);
            let len = fold(*len);
            let pad = fold(*pad);
            e(SKind::Spad {
                left,
                a: Box::new(a),
                len: Box::new(len),
                pad: Box::new(pad),
            })
        }
        SKind::Sslice { a, lo, hi } => {
            let a = fold(*a);
            let lo = fold(*lo);
            let hi = fold(*hi);
            e(SKind::Sslice {
                a: Box::new(a),
                lo: Box::new(lo),
                hi: Box::new(hi),
            })
        }
        SKind::Sord { empty_zero, a } => {
            let a = fold(*a);
            e(SKind::Sord {
                empty_zero,
                a: Box::new(a),
            })
        }
        SKind::StripAccents(a) => {
            let a = fold(*a);
            e(SKind::StripAccents(Box::new(a)))
        }
        SKind::Reverse(a) => {
            let a = fold(*a);
            e(SKind::Reverse(Box::new(a)))
        }
        SKind::Not(inner) => {
            let inner = fold(*inner);
            match as_const(&inner) {
                Some(K::Val(Lit::I1(b))) => lit(Lit::I1(!b), ty),
                Some(K::Null) => null(ty),
                _ => e(SKind::Not(Box::new(inner))),
            }
        }
        SKind::IsNull { negated, inner } => {
            let inner = fold(*inner);
            match as_const(&inner) {
                Some(K::Val(_)) => lit(Lit::I1(negated), ty),
                Some(K::Null) => lit(Lit::I1(!negated), ty),
                None => e(SKind::IsNull {
                    negated,
                    inner: Box::new(inner),
                }),
            }
        }
        SKind::Arith { op, a, b } => {
            let (a, b) = (fold(*a), fold(*b));
            match (as_const(&a), as_const(&b)) {
                (Some(K::Null), Some(_)) | (Some(_), Some(K::Null)) => null(ty),
                // TASK-122: MIN % -1 at a NARROW width overflows DuckDB's
                // checked division even though the i64 value (0) is fine, so
                // the fold must not hide it from the runtime guard.
                (Some(K::Val(Lit::I64(x))), Some(K::Val(Lit::I64(y))))
                    if op == ArithOp::Rem
                        && y == -1
                        && ty.int_range().is_some_and(|(lo, _)| x == lo) =>
                {
                    e(SKind::Arith {
                        op,
                        a: Box::new(a),
                        b: Box::new(b),
                    })
                }
                (Some(K::Val(x)), Some(K::Val(y))) => match arith(op, &x, &y) {
                    Some(l) => lit(l, ty),
                    // Would trap at run time — keep the node, keep the trap.
                    None => e(SKind::Arith {
                        op,
                        a: Box::new(a),
                        b: Box::new(b),
                    }),
                },
                _ => e(SKind::Arith {
                    op,
                    a: Box::new(a),
                    b: Box::new(b),
                }),
            }
        }
        SKind::Cmp { pred, a, b } => {
            let (a, b) = (fold(*a), fold(*b));
            match (as_const(&a), as_const(&b)) {
                (Some(K::Null), Some(_)) | (Some(_), Some(K::Null)) => null(ty),
                (Some(K::Val(x)), Some(K::Val(y))) => lit(Lit::I1(cmp(pred, &x, &y)), ty),
                _ => e(SKind::Cmp {
                    pred,
                    a: Box::new(a),
                    b: Box::new(b),
                }),
            }
        }
        SKind::And { a, b } => kleene(true, *a, *b, ty, nullable),
        SKind::Or { a, b } => kleene(false, *a, *b, ty, nullable),
        SKind::Case { arms, default } => {
            let arms: Vec<_> = arms
                .into_iter()
                .map(|(c, r)| (fold(c), fold(r)))
                .collect();
            let default = default.map(|d| Box::new(fold(*d)));
            // TASK-87 face D: constant CONDITIONS select at fold time,
            // exactly the runtime walk — a FALSE/NULL condition drops its
            // arm, the first TRUE condition commits to its (already
            // folded, never evaluated-early) arm. DuckDB folds this and
            // then elides trapping siblings of a NULL result; leaving the
            // CASE unfolded hid that NULL from the TASK-85 strict-op
            // check. Arm VALUES still never fold themselves here — only
            // the selection runs, which is what the runtime does anyway.
            // An arm escaping the CASE carries the node's unified width
            // (fleet 2026-08-13: the arm's own ty re-narrowed enclosers).
            let retype = |mut r: SExpr| {
                if r.ty != ty && r.ty.is_int() && ty.is_int() {
                    r.ty = ty;
                }
                r
            };
            // TASK-124: an arm is VALUE context and the CASE's surroundings
            // may be SELECTION context (`WHERE CASE WHEN TRUE THEN (b AND
            // trap) END` traps on DuckDB -- the taken arm is eager). Letting
            // a non-constant arm escape the CASE bare would move it across
            // that boundary and lend it a laziness DuckDB never gives it, so
            // only context-independent expressions escape; anything else
            // keeps a one-arm CASE as its value-context wrapper (the
            // condition is constant TRUE, so the branch costs nothing after
            // lowering).
            let escape = |r: SExpr| -> SExpr {
                if matches!(r.kind, SKind::Lit(_) | SKind::NullOf | SKind::Col(_)) {
                    return retype(r);
                }
                let cond = SExpr {
                    kind: SKind::Lit(Lit::I1(true)),
                    ty: Ty::I1,
                    nullable: false,
                };
                e(SKind::Case {
                    arms: vec![(cond, retype(r))],
                    default: None,
                })
            };
            let mut kept = Vec::new();
            for (c, r) in arms {
                match &c.kind {
                    SKind::Lit(Lit::I1(true)) => {
                        if kept.is_empty() {
                            return escape(r); // first arm wins
                        }
                        // TRUE after dynamic arms: it IS the default now
                        return e(SKind::Case {
                            arms: kept,
                            default: Some(Box::new(r)),
                        });
                    }
                    SKind::Lit(Lit::I1(false)) | SKind::NullOf => continue,
                    _ => kept.push((c, r)),
                }
            }
            if kept.is_empty() {
                // every arm dropped: the default, or SQL's implicit NULL.
                // The default is an ARM for context purposes -- same
                // boundary, same wrapper (TASK-124).
                return match default {
                    Some(d) => escape(*d),
                    None => null(ty),
                };
            }
            e(SKind::Case {
                arms: kept,
                default,
            })
        }
        SKind::Cast { inner, trying } => {
            let inner = fold(*inner);
            // A width-only cast of a folded integer constant collapses:
            // the node existed as the NON-literal provenance mark for the
            // value-fits promotion (frontend::int_literal_value), and by
            // fold time the hints are already taken. Out-of-range
            // constants were refused or NULLed at bind, so `fits` here is
            // belt-and-braces.
            if !trying && ty.is_int() && inner.ty.is_int() {
                if let SKind::Lit(Lit::I64(v)) = inner.kind {
                    let fits = ty.int_range().map_or(true, |(lo, hi)| (lo..=hi).contains(&v));
                    if fits {
                        return lit(Lit::I64(v), ty);
                    }
                }
            }
            e(SKind::Cast {
                inner: Box::new(inner),
                trying,
            })
        }
        // TASK-103: a corpus query DID care -- upper() over a baked pure
        // extern must finish for the || SQLNULL collapse to see through it.
        // Same casemap kernel as Inst::Str1, so fold and runtime agree.
        SKind::StrCase { upper, a } => {
            let a = fold(*a);
            match &a.kind {
                SKind::Lit(Lit::Str(s)) => {
                    let f = if upper {
                        super::exec::casemap::simple_upper
                    } else {
                        super::exec::casemap::simple_lower
                    };
                    lit(Lit::Str(s.chars().map(f).collect()), ty)
                }
                SKind::NullOf => null(ty),
                _ => e(SKind::StrCase {
                    upper,
                    a: Box::new(a),
                }),
            }
        }
        SKind::Trim { side, a, chars } => e(SKind::Trim {
            side,
            a: Box::new(fold(*a)),
            chars: Box::new(fold(*chars)),
        }),
        SKind::Substr { a, start, len } => e(SKind::Substr {
            a: Box::new(fold(*a)),
            start: Box::new(fold(*start)),
            len: len.map(|l| Box::new(fold(*l))),
        }),
        SKind::Abs(a) => {
            // TASK-103: DuckDB's binder folds abs over a constant, and an
            // extern argument like abs(-3) must be a finished Lit for the
            // pure-udf bake to see it. i64::MIN stays unfolded -- the
            // runtime trap owns it, same doctrine as the shifts above.
            let a = fold(*a);
            match &a.kind {
                SKind::Lit(Lit::I64(v)) if *v != i64::MIN => lit(Lit::I64(v.abs()), ty),
                SKind::Lit(Lit::F64(v)) => lit(Lit::F64(v.abs()), ty),
                SKind::NullOf => null(ty),
                _ => e(SKind::Abs(Box::new(a))),
            }
        }
        SKind::Round(a) => e(SKind::Round(Box::new(fold(*a)))),
        SKind::Concat { a, b } => e(SKind::Concat {
            a: Box::new(fold(*a)),
            b: Box::new(fold(*b)),
        }),
    }
}

/// Kleene AND/OR, folded only when BOTH sides are constants — the full
/// three-valued table, including `FALSE AND NULL = FALSE`.
fn kleene(is_and: bool, a: SExpr, b: SExpr, ty: Ty, nullable: bool) -> SExpr {
    let (a, b) = (fold(a), fold(b));
    let tri = |e: &SExpr| match as_const(e) {
        Some(K::Val(Lit::I1(v))) => Some(Some(v)),
        Some(K::Null) => Some(None),
        _ => None,
    };
    if let (Some(x), Some(y)) = (tri(&a), tri(&b)) {
        let r = if is_and {
            match (x, y) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            }
        } else {
            match (x, y) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            }
        };
        return match r {
            Some(v) => lit(Lit::I1(v), ty),
            None => null(ty),
        };
    }
    let (a, b) = (Box::new(a), Box::new(b));
    let kind = if is_and {
        SKind::And { a, b }
    } else {
        SKind::Or { a, b }
    };
    SExpr { kind, ty, nullable }
}

/// `None` = the interpreter would trap on this — do not fold.
fn arith(op: ArithOp, a: &Lit, b: &Lit) -> Option<Lit> {
    match (a, b) {
        (Lit::I64(x), Lit::I64(y)) => match op {
            ArithOp::Add => x.checked_add(*y).map(Lit::I64),
            ArithOp::Sub => x.checked_sub(*y).map(Lit::I64),
            ArithOp::Mul => x.checked_mul(*y).map(Lit::I64),
            ArithOp::Rem => x.checked_rem(*y).map(Lit::I64),
            // Zero/MIN//-1 stay unfolded; the frontend's CASE guard turns
            // the zero row into NULL at runtime, never reaching the fold.
            ArithOp::IDiv => x.checked_div(*y).map(Lit::I64),
            ArithOp::Div => unreachable!("/ is promoted to f64 by the frontend"),
            // Trapping shifts stay unfolded (None) exactly when the
            // interpreter would trap — same kernel decides both.
            ArithOp::Shl => super::exec::kernels::duck_shl(*x, *y).ok().map(Lit::I64),
            ArithOp::Shr => Some(Lit::I64(super::exec::kernels::duck_shr(*x, *y))),
            ArithOp::BitAnd => Some(Lit::I64(x & y)),
            ArithOp::BitOr => Some(Lit::I64(x | y)),
            ArithOp::BitXor => Some(Lit::I64(x ^ y)),
        },
        (Lit::F64(x), Lit::F64(y)) => Some(Lit::F64(match op {
            ArithOp::Add => x + y,
            ArithOp::Sub => x - y,
            ArithOp::Mul => x * y,
            ArithOp::Div => x / y,
            // `//` on doubles is plain division (wave-3 pins); the zero-
            // divisor NULL comes from the frontend's CASE guard, which is
            // never folded — this arm only sees the guarded default.
            ArithOp::IDiv => x / y,
            // IEEE, exactly as exec/interp.rs: x % 0.0 is NaN, never traps.
            ArithOp::Rem => x % y,
            ArithOp::Shl | ArithOp::Shr | ArithOp::BitAnd | ArithOp::BitOr | ArithOp::BitXor => {
                unreachable!("bitwise is BIGINT-only at bind")
            }
        })),
        _ => unreachable!("operands are promoted to a common type at bind"),
    }
}

fn cmp(pred: CmpPred, a: &Lit, b: &Lit) -> bool {
    use std::cmp::Ordering;
    let ord = |o: Ordering| match pred {
        CmpPred::Eq => o == Ordering::Equal,
        CmpPred::Ne => o != Ordering::Equal,
        CmpPred::Lt => o == Ordering::Less,
        CmpPred::Le => o != Ordering::Greater,
        CmpPred::Gt => o == Ordering::Greater,
        CmpPred::Ge => o != Ordering::Less,
    };
    match (a, b) {
        (Lit::I64(x), Lit::I64(y)) => ord(x.cmp(y)),
        (Lit::Str(x), Lit::Str(y)) => ord(x.cmp(y)),
        // DuckDB DOUBLE order, exactly as exec/interp.rs computes it.
        (Lit::F64(x), Lit::F64(y)) => ord(super::exec::duck_fcmp(*x, *y)),
        _ => unreachable!("cmp operands share a type; i1 cmp rejected at bind"),
    }
}
